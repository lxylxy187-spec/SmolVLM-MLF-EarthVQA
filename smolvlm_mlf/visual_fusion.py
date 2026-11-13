"""Minimal pre-connector fusion for the frozen Idefics3 vision backbone.

No Transformers files are modified. One model instance must execute serially;
Python hooks are intended for eager, single-device training (DDP is unverified).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class DirectVisualFusion(nn.Module):
    """A: final + r4 + r8; B: final + alpha4*r4 + alpha8*r8.

    The original final post-LN tensor is the residual anchor. Projection
    weights start at zero; optional scalar scales start at one, never zero.
    """

    def __init__(self, width: int, eps: float = 1e-6, learnable_scaling: bool = False):
        super().__init__()
        if not isinstance(learnable_scaling, bool):
            raise TypeError("learnable_scaling must be a bool.")
        self.learnable_scaling = learnable_scaling
        self.early_norm = nn.LayerNorm(width, eps=eps)
        self.middle_norm = nn.LayerNorm(width, eps=eps)
        self.early_proj = nn.Linear(width, width, bias=False)
        self.middle_proj = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.early_proj.weight)
        nn.init.zeros_(self.middle_proj.weight)
        if learnable_scaling:
            self.alpha4 = nn.Parameter(torch.ones(()))
            self.alpha8 = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("alpha4", None)
            self.register_parameter("alpha8", None)

    def forward(self, early, middle, final):
        if early.shape != middle.shape or early.shape != final.shape:
            raise RuntimeError("Fusion requires aligned [real_tiles, patches, width] tensors.")
        r4 = self.early_proj(self.early_norm(early))
        r8 = self.middle_proj(self.middle_norm(middle))
        if self.learnable_scaling:
            r4 = self.alpha4 * r4
            r8 = self.alpha8 * r8
        return final + r4 + r8


def get_core(model):
    """Accept native Idefics3 conditional generation or its native core only."""
    if getattr(model.config, "model_type", None) != "idefics3":
        raise ValueError("This implementation requires model_type='idefics3'. Run the audit first.")
    core = model.model if hasattr(model, "model") else model
    if not all(hasattr(core, name) for name in ("vision_model", "connector", "text_model")):
        raise ValueError("Expected native Idefics3 vision_model/connector/text_model structure.")
    return core


def _last(output):
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class FusionHandle:
    """Own transient activations and hook lifetimes; parameters live on core."""

    def __init__(self, model, layers=(4, 8, 12), learnable_scaling=False):
        self.core = get_core(model)
        self.vision = self.core.vision_model
        blocks = self.vision.encoder.layers
        self.layers = tuple(layers)
        if len(self.layers) != 3 or not (1 <= self.layers[0] < self.layers[1] < self.layers[2] == len(blocks)):
            raise ValueError("Use three one-based depths, with the final depth equal to vision depth.")
        if any(p.requires_grad for p in self.vision.parameters()):
            raise ValueError("Freeze vision_model before attaching fusion; trainable vision is unsupported.")
        if hasattr(self.core, "visual_fusion"):
            raise ValueError("Fusion already attached; retain the original handle.")
        weight = self.core.connector.modality_projection.proj.weight
        if weight.device.type == "meta":
            raise ValueError("Materialize checkpoint weights before attaching fusion.")
        width = self.core.config.vision_config.hidden_size
        self.fusion = DirectVisualFusion(
            width, self.vision.config.layer_norm_eps, learnable_scaling=learnable_scaling
        ).to(
            device=weight.device, dtype=weight.dtype
        )
        self.core.add_module("visual_fusion", self.fusion)
        self.enabled = True
        self._hooks = []
        self.clear()
        self._hooks.append(self.vision.register_forward_pre_hook(self._begin))
        for depth in self.layers[:2]:
            self._hooks.append(blocks[depth - 1].register_forward_hook(self._capture(depth)))
        self._hooks.append(self.vision.register_forward_hook(self._finish, always_call=True))
        self._hooks.append(self.core.connector.register_forward_pre_hook(self._fuse, with_kwargs=True))

    def clear(self):
        """Drop pending activations, including after an interrupted forward."""
        self._features = {}
        self._expected = None
        self._running = False

    def _begin(self, module, args):
        if self._running or self._expected is not None:
            raise RuntimeError("Overlapping or unconsumed vision forward; call handle.clear() after an audit/error.")
        if any(p.requires_grad for p in self.vision.parameters()):
            raise RuntimeError("Vision was unfrozen after fusion attachment.")
        self._features = {}
        self._running = True

    def _capture(self, depth):
        def hook(module, args, output):
            if self._running and self.enabled:
                self._features[depth] = _last(output).detach()
        return hook

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
            raise RuntimeError("Connector input did not come from the pending native vision forward.")
        features = self._features
        self.clear()
        if not self.enabled:
            return args, kwargs
        if any(depth not in features for depth in self.layers[:2]):
            raise RuntimeError("Missing selected vision block outputs; do not bypass the native vision path.")
        target = self.fusion.early_proj.weight
        if final.device != target.device:
            raise RuntimeError("Place vision, connector and fusion on one device for this minimal implementation.")
        early, middle = (features[d].to(device=final.device, dtype=final.dtype) for d in self.layers[:2])
        fused = self.fusion(early, middle, final)
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
            "format_version": 2,
            "implementation": "pre_connector_zero_projection_residual_v2",
            "model_type": "idefics3",
            "layers_one_based": list(self.layers),
            "vision_width": self.core.config.vision_config.hidden_size,
            "layer_norm_eps": self.vision.config.layer_norm_eps,
            "learnable_scaling": self.fusion.learnable_scaling,
            "added_parameters": sum(p.numel() for p in self.fusion.parameters()),
        }


def attach_visual_fusion(model, layers=(4, 8, 12), learnable_scaling=False):
    """Call after baseline freeze policy, before optimizer/DDP construction."""
    return FusionHandle(model, layers, learnable_scaling=learnable_scaling)


def save_fusion_pretrained(model, handle, directory, **save_kwargs):
    """Save native HF weights plus explicit fusion sidecars. No optimizer state."""
    if get_core(model) is not handle.core:
        raise ValueError("Handle belongs to a different model.")
    if "state_dict" in save_kwargs:
        raise ValueError("This helper constructs its own native-only state_dict.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "model.visual_fusion." if model is not handle.core else "visual_fusion."
    native_state = {k: v for k, v in model.state_dict().items() if not k.startswith(prefix)}
    model.save_pretrained(directory, state_dict=native_state, **save_kwargs)
    torch.save({k: v.detach().cpu() for k, v in handle.fusion.state_dict().items()}, directory / "visual_fusion.pt")
    (directory / "visual_fusion_config.json").write_text(
        json.dumps(handle.metadata(), indent=2), encoding="utf-8"
    )


def load_visual_fusion(model, directory):
    """Load sidecars AFTER native from_pretrained and freezing the vision model."""
    directory = Path(directory)
    metadata = json.loads((directory / "visual_fusion_config.json").read_text(encoding="utf-8"))
    checkpoint_format = (metadata.get("format_version"), metadata.get("implementation"))
    if checkpoint_format == (1, "pre_connector_zero_projection_residual_v1"):
        learnable_scaling = False  # Original A sidecars contain no scalar keys.
    elif checkpoint_format == (2, "pre_connector_zero_projection_residual_v2"):
        learnable_scaling = metadata.get("learnable_scaling")
        if not isinstance(learnable_scaling, bool):
            raise ValueError("Fusion v2 metadata must contain boolean learnable_scaling.")
        if metadata.get("layer_norm_eps") != get_core(model).vision_model.config.layer_norm_eps:
            raise ValueError("Fusion checkpoint LayerNorm epsilon mismatch.")
    else:
        raise ValueError("Unsupported fusion checkpoint format.")
    if get_core(model).config.vision_config.hidden_size != metadata["vision_width"]:
        raise ValueError("Fusion checkpoint vision width mismatch.")
    state = torch.load(directory / "visual_fusion.pt", map_location="cpu", weights_only=True)
    handle = attach_visual_fusion(
        model, tuple(metadata["layers_one_based"]), learnable_scaling=learnable_scaling
    )
    try:
        handle.fusion.load_state_dict(state, strict=True)
    except Exception:
        handle.remove()
        raise
    return handle
