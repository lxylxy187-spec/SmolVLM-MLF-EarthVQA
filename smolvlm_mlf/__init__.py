"""Minimal MLF for the native SmolVLM-500M / Idefics3 vision path."""

from .visual_fusion import (
    DirectVisualFusion,
    attach_visual_fusion,
    load_visual_fusion,
    save_fusion_pretrained,
)

__all__ = [
    "DirectVisualFusion",
    "attach_visual_fusion",
    "load_visual_fusion",
    "save_fusion_pretrained",
]
