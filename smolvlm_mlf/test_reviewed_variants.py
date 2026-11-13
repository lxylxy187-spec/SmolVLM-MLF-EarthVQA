"""Offline checks for the reviewed A/B MLF variants and versioned sidecars.

No checkpoint downloads, datasets, or VQA training are used. Random local
Idefics3 models exercise attachment, native serialization, and gradient flow.
The default target is outputs/smolvlm_mlf; MLF_TEST_PACKAGE_DIR can override it.
This file also works unchanged when copied beside visual_fusion.py.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import Idefics3ForConditionalGeneration


_HERE = Path(__file__).resolve().parent
if os.environ.get("MLF_TEST_PACKAGE_DIR"):
    _PACKAGE = Path(os.environ["MLF_TEST_PACKAGE_DIR"]).resolve()
elif (_HERE / "visual_fusion.py").is_file():
    _PACKAGE = _HERE
else:
    _PACKAGE = _HERE.parent.parent / "outputs" / "smolvlm_mlf"
if not (_PACKAGE / "visual_fusion.py").is_file():
    raise ImportError(f"Reviewed MLF package has not been created at {_PACKAGE}")
sys.path.insert(0, str(_PACKAGE))

from visual_fusion import (  # noqa: E402
    DirectVisualFusion,
    attach_visual_fusion,
    load_visual_fusion,
    save_fusion_pretrained,
)
from test_visual_fusion import multitile_batch, tiny_model  # noqa: E402


def _nonzero_projection(fusion):
    """Give both random-model branches a deterministic nonzero signal."""
    with torch.no_grad():
        for scale, projection in (
            (0.05, fusion.early_proj),
            (0.025, fusion.middle_proj),
        ):
            projection.weight.copy_(
                torch.eye(
                    projection.weight.shape[0],
                    device=projection.weight.device,
                    dtype=projection.weight.dtype,
                ) * scale
            )


class ReviewedVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(29)

    def test_real_width_parameter_counts_and_initial_identity_for_a_and_b(self):
        inputs = [torch.randn(2, 7, 768) for _ in range(3)]
        for scaling, expected_count in ((False, 1_182_720), (True, 1_182_722)):
            with self.subTest(learnable_scaling=scaling):
                fusion = DirectVisualFusion(768, learnable_scaling=scaling)
                self.assertEqual(sum(p.numel() for p in fusion.parameters()), expected_count)
                named = dict(fusion.named_parameters())
                if scaling:
                    self.assertIn("alpha4", named)
                    self.assertIn("alpha8", named)
                    self.assertEqual(fusion.alpha4.item(), 1.0)
                    self.assertEqual(fusion.alpha8.item(), 1.0)
                else:
                    self.assertNotIn("alpha4", named)
                    self.assertNotIn("alpha8", named)
                for projection in (fusion.early_proj, fusion.middle_proj):
                    self.assertIsNone(projection.bias)
                    self.assertEqual(torch.count_nonzero(projection.weight).item(), 0)
                torch.testing.assert_close(fusion(*inputs), inputs[2], rtol=0, atol=0)

    def test_b_starts_without_double_zero_and_alpha_receives_later_gradient(self):
        fusion = DirectVisualFusion(16, learnable_scaling=True)
        early, middle, final, upstream = [torch.randn(2, 7, 16) for _ in range(4)]
        # A zero projector with alpha=1 preserves the baseline yet lets the
        # projector receive task gradients on the first backward pass.
        self.assertEqual(fusion.alpha4.item(), 1.0)
        self.assertEqual(fusion.alpha8.item(), 1.0)
        (fusion(early, middle, final) * upstream).sum().backward()
        for projection in (fusion.early_proj, fusion.middle_proj):
            self.assertIsNotNone(projection.weight.grad)
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0.0)
        for alpha in (fusion.alpha4, fusion.alpha8):
            self.assertIsNotNone(alpha.grad)
            self.assertEqual(alpha.grad.item(), 0.0)
        for norm in (fusion.early_norm, fusion.middle_norm):
            for parameter in norm.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

        fusion.zero_grad(set_to_none=True)
        _nonzero_projection(fusion)
        (fusion(early, middle, final) * upstream).sum().backward()
        for alpha in (fusion.alpha4, fusion.alpha8):
            self.assertIsNotNone(alpha.grad)
            self.assertTrue(torch.isfinite(alpha.grad))
            self.assertGreater(abs(alpha.grad.item()), 1e-8)

    def test_b_sidecar_restores_variant_weights_logits_and_optimizer_membership(self):
        model = tiny_model()
        batch = multitile_batch()
        handle = attach_visual_fusion(model, learnable_scaling=True)
        try:
            self.assertIs(model.model.visual_fusion, handle.fusion)
            # The model's native parameters are frozen in tiny_model(). Building
            # an optimizer after attachment must include the newly registered alphas.
            optimizer = torch.optim.SGD(
                (p for p in model.parameters() if p.requires_grad), lr=0.1
            )
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertIn(id(handle.fusion.alpha4), optimizer_ids)
            self.assertIn(id(handle.fusion.alpha8), optimizer_ids)
            self.assertTrue(all(id(p) in optimizer_ids for p in handle.fusion.parameters()))
            self.assertTrue(all(not p.requires_grad for p in model.model.vision_model.parameters()))

            _nonzero_projection(handle.fusion)
            with torch.no_grad():
                handle.fusion.alpha4.fill_(1.25)
                handle.fusion.alpha8.fill_(-0.75)
                expected_logits = model(**batch, use_cache=False).logits.detach().clone()
            expected_state = copy.deepcopy(handle.fusion.state_dict())
            with tempfile.TemporaryDirectory(prefix="mlf-reviewed-b-") as folder:
                save_fusion_pretrained(model, handle, folder, safe_serialization=True)
                metadata = json.loads(
                    (Path(folder) / "visual_fusion_config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["format_version"], 2)
                self.assertIs(metadata["learnable_scaling"], True)
                self.assertEqual(metadata["layers_one_based"], [4, 8, 12])
                self.assertEqual(metadata["vision_width"], 8)
                self.assertEqual(metadata["added_parameters"], 162)

                restored = Idefics3ForConditionalGeneration.from_pretrained(
                    folder, local_files_only=True, attn_implementation="eager"
                ).eval()
                self.assertFalse(hasattr(restored.model, "visual_fusion"))
                restored.requires_grad_(False)
                loaded = load_visual_fusion(restored, folder)
                try:
                    self.assertIs(loaded.metadata()["learnable_scaling"], True)
                    self.assertEqual(loaded.metadata()["format_version"], 2)
                    self.assertEqual(loaded.fusion.alpha4.item(), 1.25)
                    self.assertEqual(loaded.fusion.alpha8.item(), -0.75)
                    for name, value in expected_state.items():
                        torch.testing.assert_close(loaded.fusion.state_dict()[name], value, rtol=0, atol=0)
                    with torch.no_grad():
                        actual_logits = restored(**batch, use_cache=False).logits
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
                    restored_optimizer = torch.optim.SGD(
                        (p for p in restored.parameters() if p.requires_grad), lr=0.1
                    )
                    restored_ids = {
                        id(p) for group in restored_optimizer.param_groups for p in group["params"]
                    }
                    self.assertIn(id(loaded.fusion.alpha4), restored_ids)
                    self.assertIn(id(loaded.fusion.alpha8), restored_ids)
                    self.assertEqual(loaded._features, {})
                    self.assertIsNone(loaded._expected)
                finally:
                    loaded.remove()
        finally:
            handle.remove()

    def test_legacy_v1_a_sidecar_remains_loadable_without_alpha_parameters(self):
        model = tiny_model()
        handle = attach_visual_fusion(model)
        try:
            with tempfile.TemporaryDirectory(prefix="mlf-reviewed-v1-a-") as folder:
                save_fusion_pretrained(model, handle, folder, safe_serialization=True)
                config_path = Path(folder) / "visual_fusion_config.json"
                metadata = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["format_version"], 2)
                self.assertIs(metadata["learnable_scaling"], False)
                # Reproduce the exact old A sidecar schema, preserving its
                # compatible parameter state. Only temporary fixture files change.
                metadata["format_version"] = 1
                metadata["implementation"] = "pre_connector_zero_projection_residual_v1"
                metadata.pop("learnable_scaling")
                config_path.write_text(json.dumps(metadata), encoding="utf-8")
                restored = Idefics3ForConditionalGeneration.from_pretrained(
                    folder, local_files_only=True, attn_implementation="eager"
                ).eval()
                restored.requires_grad_(False)
                loaded = load_visual_fusion(restored, folder)
                try:
                    named = dict(loaded.fusion.named_parameters())
                    self.assertNotIn("alpha4", named)
                    self.assertNotIn("alpha8", named)
                    self.assertIs(loaded.metadata()["learnable_scaling"], False)
                    for name, value in handle.fusion.state_dict().items():
                        torch.testing.assert_close(loaded.fusion.state_dict()[name], value, rtol=0, atol=0)
                finally:
                    loaded.remove()
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main(verbosity=2)
