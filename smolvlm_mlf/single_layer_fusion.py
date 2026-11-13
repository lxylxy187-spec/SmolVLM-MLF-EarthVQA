"""One-branch MLF ablation on the native Idefics3 vision-to-connector path.

This module deliberately leaves the full H4+H8 implementation untouched.
Only one intermediate block is hooked and only one LN/projection is created.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from smolvlm_mlf.visual_fusion import _last, get_core


class SingleLayerVisualFusion(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=eps)
        self.proj = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, selected: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
        if selected.shape != final.shape:
            raise RuntimeError("Single-layer fusion requires aligned real-tile visual tokens")
        return final + self.proj(self.norm(selected))


class SingleLayerFusionHandle:
    def __init__(self, model, selected_depth: int):
        self.core = get_core(model)
        self.vision = self.core.vision_model
        blocks = self.vision.encoder.layers
        if selected_depth not in (4, 8) or len(blocks) != 12:
            raise ValueError("Use only the audited H4 or H8 of the 12-block vision encoder")
        if any(p.requires_grad for p in self.vision.parameters()):
            raise ValueError("Freeze vision_model before attaching fusion")
        if hasattr(self.core, "visual_fusion"):
            raise ValueError("Another visual fusion module is already attached")
        weight = self.core.connector.modality_projection.proj.weight
        if weight.device.type == "meta":
            raise ValueError("Materialize the original checkpoint before attaching fusion")
        self.selected_depth = selected_depth
        width = self.core.config.vision_config.hidden_size
        self.fusion = SingleLayerVisualFusion(width, self.vision.config.layer_norm_eps).to(
            device=weight.device, dtype=weight.dtype
        )
        self.core.add_module("visual_fusion", self.fusion)
        self._hooks = []
        self.clear()
        self._hooks.append(self.vision.register_forward_pre_hook(self._begin))
        self._hooks.append(blocks[selected_depth - 1].register_forward_hook(self._capture))
        self._hooks.append(self.vision.register_forward_hook(self._finish, always_call=True))
        self._hooks.append(self.core.connector.register_forward_pre_hook(self._fuse, with_kwargs=True))

    def clear(self):
        self._selected = None
        self._expected = None
        self._running = False

    def _begin(self, module, args):
        if self._running or self._expected is not None:
            raise RuntimeError("Overlapping or unconsumed vision forward")
        if any(p.requires_grad for p in self.vision.parameters()):
            raise RuntimeError("Vision was unfrozen after fusion attachment")
        self._running = True
        self._selected = None

    def _capture(self, module, args, output):
        if self._running:
            self._selected = _last(output).detach()

    def _finish(self, module, args, output):
        self._running = False
        if output is None:
            self.clear()
        else:
            self._expected = _last(output)

    def _fuse(self, module, args, kwargs):
        final = args[0] if args else kwargs.get("image_hidden_states")
        if final is not self._expected:
            self.clear()
            raise RuntimeError("Connector input is not the native final post-layernorm vision output")
        selected = self._selected
        self.clear()
        if selected is None:
            raise RuntimeError("Missing selected vision block output")
        if final.device != self.fusion.proj.weight.device:
            raise RuntimeError("Fusion and native vision/connector must be on the same device")
        selected = selected.to(device=final.device, dtype=final.dtype)
        fused = self.fusion(selected, final)
        if args:
            return (fused, *args[1:]), kwargs
        return args, {**kwargs, "image_hidden_states": fused}

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.clear()
        if getattr(self.core, "visual_fusion", None) is self.fusion:
            del self.core.visual_fusion

    def metadata(self):
        return {
            "format_version": 1,
            "implementation": "pre_connector_single_layer_zero_projection_residual",
            "model_type": "idefics3",
            "selected_depth_one_based": self.selected_depth,
            "final_depth_one_based": 12,
            "vision_width": self.core.config.vision_config.hidden_size,
            "layer_norm_eps": self.vision.config.layer_norm_eps,
            "added_parameters": sum(p.numel() for p in self.fusion.parameters()),
        }


def attach_single_layer_fusion(model, selected_depth: int):
    return SingleLayerFusionHandle(model, selected_depth)


def save_single_layer_fusion_pretrained(model, handle, directory, **save_kwargs):
    if get_core(model) is not handle.core:
        raise ValueError("Handle belongs to a different model")
    if "state_dict" in save_kwargs:
        raise ValueError("This helper constructs its own native-only state_dict")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "model.visual_fusion." if model is not handle.core else "visual_fusion."
    native_state = {k: v for k, v in model.state_dict().items() if not k.startswith(prefix)}
    model.save_pretrained(directory, state_dict=native_state, **save_kwargs)
    torch.save(
        {k: v.detach().cpu() for k, v in handle.fusion.state_dict().items()},
        directory / "visual_fusion.pt",
    )
    (directory / "visual_fusion_config.json").write_text(
        json.dumps(handle.metadata(), indent=2) + "\n", encoding="utf-8"
    )


def load_single_layer_fusion(model, directory, expected_depth: int | None = None):
    directory = Path(directory)
    metadata = json.loads((directory / "visual_fusion_config.json").read_text(encoding="utf-8"))
    if (metadata.get("format_version"), metadata.get("implementation")) != (
        1, "pre_connector_single_layer_zero_projection_residual"
    ):
        raise ValueError("Unsupported single-layer fusion sidecar format")
    depth = metadata.get("selected_depth_one_based")
    if expected_depth is not None and depth != expected_depth:
        raise ValueError(f"Selected depth mismatch: expected {expected_depth}, found {depth}")
    core = get_core(model)
    if metadata.get("vision_width") != core.config.vision_config.hidden_size:
        raise ValueError("Vision width mismatch")
    if metadata.get("layer_norm_eps") != core.vision_model.config.layer_norm_eps:
        raise ValueError("LayerNorm epsilon mismatch")
    state = torch.load(directory / "visual_fusion.pt", map_location="cpu", weights_only=True)
    handle = attach_single_layer_fusion(model, depth)
    try:
        handle.fusion.load_state_dict(state, strict=True)
        if handle.metadata()["added_parameters"] != metadata["added_parameters"]:
            raise ValueError("Added-parameter count mismatch")
    except Exception:
        handle.remove()
        raise
    return handle
