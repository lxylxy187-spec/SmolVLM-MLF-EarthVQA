"""Offline integration smoke tests against native Transformers Idefics3.

Run with: python -m unittest discover -s . -p test_visual_fusion.py -v
These random tiny models test wiring and checkpoint behavior, not VQA quality.
The visual encoder retains 12 real blocks so the selected depths are 4/8/12.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Idefics3Config, Idefics3ForConditionalGeneration

from visual_fusion import attach_visual_fusion, load_visual_fusion, save_fusion_pretrained


def tiny_model():
    """Construct a real architecture locally; no model/tokenizer downloads."""
    config = Idefics3Config(
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 12,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 2,
            "attention_dropout": 0.0,
        },
        text_config={
            "model_type": "llama",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "attention_dropout": 0.0,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
        },
        scale_factor=2,
        image_token_id=63,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Idefics3ForConditionalGeneration(config)
    model.requires_grad_(False)
    return model.eval()


def multitile_batch():
    """Two samples, three real tiles, three all-zero tile padding slots.

    Sample zero has two real tiles; its second tile is only half valid.
    Sample one has one real tile and left text padding. Every tile produces
    four connector tokens (16 patches / scale_factor**2).
    """
    generator = torch.Generator().manual_seed(23)
    pixels = torch.zeros(2, 3, 3, 8, 8)
    pixels[0, :2] = torch.randn(2, 3, 8, 8, generator=generator)
    pixels[1, 0] = torch.randn(3, 8, 8, generator=generator)
    pixels[0, 1, :, 4:, :] = 0
    pixel_mask = torch.zeros(2, 3, 8, 8, dtype=torch.bool)
    pixel_mask[0, 0] = True
    pixel_mask[0, 1, :4, :] = True
    pixel_mask[1, 0] = True
    input_ids = torch.tensor(
        [
            [1] + [63] * 8 + [10, 11, 12, 13, 14, 15, 16],
            [0] * 4 + [1] + [63] * 4 + [10, 11, 12, 13, 14, 15, 16],
        ],
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "pixel_values": pixels,
        "pixel_attention_mask": pixel_mask,
    }


class VisualFusionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Very small CPU matrices are substantially faster without thread pools.
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.model = tiny_model()
        self.batch = multitile_batch()
        self.handle = None

    def tearDown(self):
        if self.handle is not None:
            self.handle.remove()

    def attach(self):
        self.handle = attach_visual_fusion(self.model, layers=(4, 8, 12))
        return self.handle

    def logits(self, model=None):
        with torch.no_grad():
            return (model or self.model)(**self.batch, use_cache=False).logits.detach().clone()

    def assert_handle_clear(self, handle=None):
        handle = handle or self.handle
        self.assertEqual(handle._features, {})
        self.assertIsNone(handle._expected)
        self.assertFalse(handle._running)

    def optimize_fusion_once(self):
        """One actual language-model loss update, with every native weight frozen."""
        self.model.train()
        labels = self.batch["input_ids"].clone()
        labels[(labels == 63) | (labels == 0)] = -100
        optimizer = torch.optim.SGD(self.handle.fusion.parameters(), lr=1.5)
        optimizer.zero_grad(set_to_none=True)
        result = self.model(**self.batch, labels=labels, use_cache=False)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        for projection in (self.handle.fusion.early_proj, self.handle.fusion.middle_proj):
            self.assertIsNotNone(projection.weight.grad)
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0.0)
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
        self.assertTrue(all(not p.requires_grad for p in self.model.model.vision_model.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.model.vision_model.parameters()))
        optimizer.step()
        self.model.eval()
        self.assert_handle_clear()

    def test_zero_init_preserves_logits_generation_and_native_tile_masking(self):
        baseline_logits = self.logits()
        with torch.no_grad():
            baseline_generation = self.model.generate(
                **self.batch, max_new_tokens=3, do_sample=False, use_cache=True
            )
        self.attach()
        observed = {}

        def vision_inputs(module, args, kwargs):
            observed["vision_pixels"] = tuple(kwargs["pixel_values"].shape)
            observed["patch_mask"] = kwargs["patch_attention_mask"].detach().clone()

        def fusion_inputs(module, args):
            observed["fusion_shapes"] = [tuple(t.shape) for t in args]

        vision_hook = self.model.model.vision_model.register_forward_pre_hook(vision_inputs, with_kwargs=True)
        fusion_hook = self.handle.fusion.register_forward_pre_hook(fusion_inputs)
        try:
            fused_logits = self.logits()
            with torch.no_grad():
                fused_generation = self.model.generate(
                    **self.batch, max_new_tokens=3, do_sample=False, use_cache=True
                )
        finally:
            vision_hook.remove()
            fusion_hook.remove()
        torch.testing.assert_close(fused_logits, baseline_logits, rtol=0, atol=0)
        self.assertTrue(torch.equal(fused_generation, baseline_generation))
        self.assertEqual(observed["vision_pixels"], (3, 3, 8, 8))
        self.assertEqual(observed["fusion_shapes"], [(3, 16, 8)] * 3)
        self.assertEqual(observed["patch_mask"].sum(dim=(1, 2)).tolist(), [16, 8, 16])
        self.assert_handle_clear()

    def test_loss_backpropagates_to_fusion_and_optimizer_changes_outputs(self):
        self.attach()
        before = self.logits()
        self.optimize_fusion_once()
        after = self.logits()
        self.assertGreater((after - before).abs().max().item(), 1e-8)
        self.assertFalse(torch.equal(after, before))

    def test_sidecar_roundtrip_reproduces_weights_outputs_and_requires_explicit_load(self):
        self.attach()
        baseline_logits = self.logits()
        self.optimize_fusion_once()
        trained_logits = self.logits()
        expected_state = copy.deepcopy(self.handle.fusion.state_dict())
        with tempfile.TemporaryDirectory(prefix="smolvlm-fusion-test-") as folder:
            save_fusion_pretrained(self.model, self.handle, folder, safe_serialization=True)
            self.assertTrue((Path(folder) / "visual_fusion.pt").is_file())
            self.assertTrue((Path(folder) / "visual_fusion_config.json").is_file())
            native_only = Idefics3ForConditionalGeneration.from_pretrained(
                folder, local_files_only=True, attn_implementation="eager"
            ).eval()
            # HF from_pretrained alone intentionally does not reconstruct hooks
            # or the fusion module. Deployments must then load the sidecars.
            self.assertFalse(hasattr(native_only.model, "visual_fusion"))
            torch.testing.assert_close(self.logits(native_only), baseline_logits, rtol=0, atol=0)
            self.assertFalse(torch.equal(self.logits(native_only), trained_logits))
            native_only.requires_grad_(False)
            loaded_handle = load_visual_fusion(native_only, folder)
            try:
                for name, weight in expected_state.items():
                    torch.testing.assert_close(loaded_handle.fusion.state_dict()[name], weight, rtol=0, atol=0)
                torch.testing.assert_close(self.logits(native_only), trained_logits, rtol=0, atol=0)
                self.assert_handle_clear(loaded_handle)
            finally:
                loaded_handle.remove()

    def test_repeated_cached_generation_does_not_reencode_or_reuse_stale_features(self):
        self.attach()
        calls = []
        hook = self.model.model.vision_model.register_forward_pre_hook(lambda module, args: calls.append(1))
        try:
            with torch.no_grad():
                first = self.model.generate(
                    **self.batch, max_new_tokens=4, min_new_tokens=4, do_sample=False, use_cache=True
                )
                self.assert_handle_clear()
                second = self.model.generate(
                    **self.batch, max_new_tokens=4, min_new_tokens=4, do_sample=False, use_cache=True
                )
        finally:
            hook.remove()
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(calls), 2, "Each generate should run vision once during prefill.")
        self.assert_handle_clear()

    def test_failed_vision_forward_cleans_activations_and_clear_recovers_pending_audit(self):
        self.attach()

        def fail(module, args):
            raise RuntimeError("injected vision failure")

        # The first selected layer has already been captured when layer six fails.
        hook = self.model.model.vision_model.encoder.layers[5].register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected vision failure"):
                self.logits()
        finally:
            hook.remove()
        self.assert_handle_clear()
        recovered = self.logits()
        self.assertTrue(torch.isfinite(recovered).all())
        # A standalone encoder audit deliberately leaves output pending; callers
        # must clear it before returning to ordinary model execution.
        with torch.no_grad():
            self.model.model.vision_model(pixel_values=self.batch["pixel_values"][0, :1])
        self.assertIsNotNone(self.handle._expected)
        self.handle.clear()
        torch.testing.assert_close(self.logits(), recovered, rtol=0, atol=0)
        self.assert_handle_clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
