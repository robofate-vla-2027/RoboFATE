"""BF16 arithmetic on reduced-width native CPU modules, without model weights."""
import unittest

import torch

from test_native_integration import tiny_model, synthetic_batch
from test_pipeline_integration import cache_features, history_context


class PrecisionIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)

    def test_dual_forward_backward_under_bf16_autocast(self):
        model = tiny_model().train()
        samples, decisions = synthetic_batch()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses = model(samples, decisions, original=False)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in losses.values()))
        losses['loss'].backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertGreater(float(model.mo_out.weight.grad.norm()), 0)
        self.assertGreater(float(model.controller.gru.weight_ih_l0.grad.norm()), 0)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))

    def test_cached_visual_and_history_features_under_bf16(self):
        model = tiny_model().train()
        model.dm.model.visual.to(dtype=torch.bfloat16)
        samples, decisions = synthetic_batch(original=True)
        for i, sample in enumerate(samples):
            current = history_context(i, i+2)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                sample['context'] = cache_features(model, current)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses = model(samples, decisions, original=True)
        self.assertTrue(bool(torch.isfinite(losses['loss'])))
        losses['loss'].backward()
        model.discard_inactive_gradients(True)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        projection = model.dm.model.vlm.model.multi_modal_projector.mm_input_projection_weight
        self.assertGreater(float(projection.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.controller.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
