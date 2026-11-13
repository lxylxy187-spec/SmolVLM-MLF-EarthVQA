"""Minimal fixed bottleneck Adapter for the B-group capacity control.

The module is attached to the native Idefics3 connector output.  It processes
each post-connector visual token independently and leaves the native image
splitting, tile filtering, mask handling, placeholder merge, connector and
language model unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


IMPLEMENTATION = "post_connector_bottleneck_residual_v1"
FORMAT_VERSION = 1


class BottleneckVisualAdapter(nn.Module):
    """H + W2(GELU(W1(H))) with a zero-initialized output projection."""

    def __init__(self, hidden_size: int = 960, bottleneck_size: int = 615):
        super().__init__()
        if hidden_size <= 0 or bottleneck_size <= 0:
            raise ValueError("Adapter dimensions must be positive")
        if bottleneck_size >= hidden_size:
            raise ValueError("Bottleneck size must be smaller than hidden size")
        self.hidden_size = int(hidden_size)
        self.bottleneck_size = int(bottleneck_size)
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=True)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_size, hidden_size, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Adapter expected last dim {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        return hidden_states + self.up(self.activation(self.down(hidden_states)))


def get_core(model):
    if getattr(model.config, "model_type", None) != "idefics3":
        raise ValueError("This adapter requires model_type='idefics3'")
    core = model.model if hasattr(model, "model") else model
    if not all(hasattr(core, name) for name in ("vision_model", "connector", "text_model")):
        raise ValueError("Expected native Idefics3 vision_model/connector/text_model structure")
    return core


class AdapterHandle:
    """Own the connector hook; registered parameters live on the model core."""

    def __init__(self, model, hidden_size: int = 960, bottleneck_size: int = 615):
        self.core = get_core(model)
        if hasattr(self.core, "visual_adapter"):
            raise ValueError("Visual adapter already attached; retain the original handle")
        projection = self.core.connector.modality_projection.proj
        if projection.weight.device.type == "meta":
            raise ValueError("Materialize checkpoint weights before attaching adapter")
        connector_hidden = int(projection.out_features)
        if connector_hidden != hidden_size:
            raise ValueError(
                f"Connector output width {connector_hidden} != requested {hidden_size}"
            )
        self.adapter = BottleneckVisualAdapter(hidden_size, bottleneck_size).to(
            device=projection.weight.device,
            dtype=projection.weight.dtype,
        )
        self.core.add_module("visual_adapter", self.adapter)
        self.enabled = True
        self._hook = self.core.connector.register_forward_hook(
            self._adapt_connector_output,
            always_call=True,
        )

    def _adapt_connector_output(self, module, args, output):
        if output is None or not self.enabled:
            return output
        if not torch.is_tensor(output):
            raise RuntimeError(
                "Native Idefics3 connector returned a non-tensor output; "
                "the fixed B-group adapter cannot be applied safely"
            )
        if output.device != self.adapter.up.weight.device:
            raise RuntimeError("Connector and adapter must be on the same device")
        return self.adapter(output)

    def remove(self):
        self._hook.remove()
        if getattr(self.core, "visual_adapter", None) is self.adapter:
            del self.core.visual_adapter

    def metadata(self):
        return {
            "format_version": FORMAT_VERSION,
            "implementation": IMPLEMENTATION,
            "model_type": "idefics3",
            "location": "connector_output_before_inputs_merger",
            "hidden_size": self.adapter.hidden_size,
            "bottleneck_size": self.adapter.bottleneck_size,
            "activation": "gelu",
            "down_bias": True,
            "up_bias": True,
            "up_zero_initialized_at_attach": True,
            "token_count_changed": False,
            "added_parameters": sum(p.numel() for p in self.adapter.parameters()),
        }


def attach_visual_adapter(model, hidden_size: int = 960, bottleneck_size: int = 615):
    """Attach after the baseline freeze policy and before optimizer creation."""
    return AdapterHandle(model, hidden_size=hidden_size, bottleneck_size=bottleneck_size)


def save_adapter_pretrained(model, handle, directory, **save_kwargs):
    """Save native HF weights and the Adapter as an explicit sidecar."""
    if get_core(model) is not handle.core:
        raise ValueError("Adapter handle belongs to a different model")
    if "state_dict" in save_kwargs:
        raise ValueError("This helper constructs its own native-only state_dict")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "model.visual_adapter." if model is not handle.core else "visual_adapter."
    native_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    }
    model.save_pretrained(directory, state_dict=native_state, **save_kwargs)
    torch.save(
        {key: value.detach().cpu() for key, value in handle.adapter.state_dict().items()},
        directory / "visual_adapter.pt",
    )
    (directory / "visual_adapter_config.json").write_text(
        json.dumps(handle.metadata(), indent=2) + "\n",
        encoding="utf-8",
    )


def load_visual_adapter(model, directory):
    """Load Adapter sidecars after native from_pretrained."""
    directory = Path(directory)
    metadata = json.loads(
        (directory / "visual_adapter_config.json").read_text(encoding="utf-8")
    )
    if (metadata.get("format_version"), metadata.get("implementation")) != (
        FORMAT_VERSION,
        IMPLEMENTATION,
    ):
        raise ValueError("Unsupported visual adapter checkpoint format")
    if metadata.get("activation") != "gelu":
        raise ValueError("B-group checkpoint must use GELU")
    if metadata.get("down_bias") is not True or metadata.get("up_bias") is not True:
        raise ValueError("B-group checkpoint must use bias in both linear layers")
    state = torch.load(
        directory / "visual_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    handle = attach_visual_adapter(
        model,
        hidden_size=int(metadata["hidden_size"]),
        bottleneck_size=int(metadata["bottleneck_size"]),
    )
    try:
        handle.adapter.load_state_dict(state, strict=True)
    except Exception:
        handle.remove()
        raise
    if handle.metadata()["added_parameters"] != int(metadata["added_parameters"]):
        handle.remove()
        raise ValueError("Adapter parameter-count mismatch")
    return handle
