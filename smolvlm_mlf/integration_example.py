"""Copy the relevant calls into the existing trainer/evaluator.

Nothing loads a model or starts training on import. Keep the baseline's
processor, data pipeline, freeze policy, optimizer settings and decoding.
"""
from .visual_fusion import (
    attach_visual_fusion,
    load_visual_fusion,
    save_fusion_pretrained,
)


def attach_before_optimizer(model, variant="A"):
    """Call after native weights are loaded/placed and baseline freezing is set."""
    if variant not in ("A", "B"):
        raise ValueError("Choose A (main) or B (backup).")
    return attach_visual_fusion(
        model, layers=(4, 8, 12), learnable_scaling=(variant == "B")
    )


def assert_optimizer_contains_mlf(optimizer, handle):
    """Detect name-filtered optimizers that accidentally omit fusion parameters."""
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [
        name for name, p in handle.fusion.named_parameters()
        if not p.requires_grad or id(p) not in optimizer_ids
    ]
    if missing:
        raise RuntimeError(f"MLF parameters frozen or missing from optimizer: {missing}")


def save_for_evaluation(model, processor, handle, output_dir):
    """Save native weights plus MLF sidecars; this is NOT a full trainer resume."""
    save_fusion_pretrained(model, handle, output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)


def restore_after_native_loading(model, model_dir):
    """Call after evaluator's native from_pretrained and baseline freeze setup."""
    handle = load_visual_fusion(model, model_dir)
    model.eval()
    return handle
