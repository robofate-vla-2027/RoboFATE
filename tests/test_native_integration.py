"""Randomly initialized, reduced-width native DM0.5 CPU integration tests."""
import copy
import unittest

import torch
from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig
from opendm.model.dm05 import dm05_arch as arch
from robofate.config import PAPER
from robofate.data import collate_contexts
from robofate.model import RoboFATE, install_visual_dispatch, project_history_with_grad
from robofate.train import optimizer_for


def tiny_model():
    text = Gemma3TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                           vocab_size=64, max_position_embeddings=1024,
                           layer_types=["full_attention", "full_attention"],
                           rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
                           query_pre_attn_scalar=8, pad_token_id=0, bos_token_id=2, eos_token_id=1)
    vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=32, patch_size=8, vision_use_head=False)
    vlm = Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                      image_token_index=63, boi_token_index=62, eoi_token_index=61, mm_tokens_per_image=4)
    native = arch.DM05ForConditionalGeneration(arch.DM05Config(
        vlm_config=vlm.to_dict(), action_config=text.to_dict(), action_dim=32, chunk_size=50))
    # The native scratch projector starts at zero; use a nondegenerate fixture.
    torch.nn.init.normal_(native.model.vlm.model.multi_modal_projector.mm_input_projection_weight, std=0.02)
    native.set_attention_implementation(llm_attn_implementation="sdpa", vision_attn_implementation="sdpa", action_attn_implementation="sdpa")
    config = copy.deepcopy(PAPER["model"])
    config.update(summary_width=16, risk_width=16, queries=4, attention_heads=2, gru_hidden=8,
                  action_width=16, action_layers=2, action_ffn_width=32, visual_tokens_per_view=4)
    model = RoboFATE(native, config)
    install_visual_dispatch(model.dm.model.visual)
    arch._project_history_image_features = project_history_with_grad
    model.dm.enable_gradient_checkpointing(vlm_gradient_checkpointing=True, ae_gradient_checkpointing=True)
    model.moae.gradient_checkpointing_enable()
    return model


def native_context(history=False, length=2):
    tokens = [2] + ([7] * 320 if history else []) + [63] * 12 + [10 + i for i in range(length)] + [1]
    ids = torch.tensor([tokens])
    types = (ids == 63).long()
    state = torch.zeros_like(ids, dtype=torch.bool)
    state[0, -2] = True
    result = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": types,
              "state_token_mask": state, "pixel_values": torch.randn(3, 3, 32, 32), "visual_features": False}
    if history:
        result["history_mask"] = torch.zeros_like(ids, dtype=torch.bool)
    return result


def synthetic_batch(original=False):
    samples, decisions = [], []
    for i in range(4):
        mask = torch.zeros(50, 32, dtype=torch.bool); mask[:50-i, :14] = True
        sample = {"action": torch.randn(50, 32), "mask": mask,
                  "context": native_context(history=True, length=i+2)}
        if not original:
            sample["history"] = [native_context(length=i+2) for _ in range(8)]
            decisions.append({"history": [native_context(length=3) for _ in range(8)], "label": 0.0})
        samples.append(sample)
    return samples, decisions


class NativeIntegrationTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(71)
    def test_batched_forward_backward_on_real_native_modules(self):
        model = tiny_model().train()
        losses = model(*synthetic_batch(), original=False)
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        losses["loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.dm.model.visual.parameters()))
        self.assertGreater(model.dm.model.vlm.model.multi_modal_projector.mm_input_projection_weight.grad.norm(), 0)
        self.assertGreater(model.controller.trigger[-1].weight.grad.norm(), 0)
        self.assertGreater(model.mo_out.weight.grad.norm(), 0)
    def test_original_optimizer_updates_only_permitted_parameters(self):
        model = tiny_model().train()
        optimizer = optimizer_for(model, PAPER)
        model(*synthetic_batch(), original=False)["loss"].backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n,p in model.named_parameters()}
        model(*synthetic_batch(original=True), original=True)["loss"].backward()
        model.discard_inactive_gradients(True)
        optimizer.step()
        changed = []
        for name, parameter in model.named_parameters():
            if not torch.equal(parameter, before[name]): changed.append(name)
            if not model.original_update_allowed(name):
                self.assertTrue(torch.equal(parameter, before[name]), name)
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(any(".mlp.shared." in name for name in changed))
        self.assertTrue(any(name.startswith("dm.") for name in changed))
    def test_batched_current_encoding_matches_single_samples(self):
        model = tiny_model().eval()
        contexts = [native_context(length=i+2) for i in range(4)]
        together = model.summarize(collate_contexts(contexts))
        separate = torch.cat([model.summarize(c) for c in contexts])
        torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-5)
    def test_ten_step_bae_and_moae_generation(self):
        model = tiny_model().eval()
        context = collate_contexts([native_context(history=True, length=i+2) for i in range(4)])
        risk = torch.randn(4, 16)
        for decoder in ("bae", "moae"):
            generated = model.generate(context, risk, decoder=decoder)
            self.assertEqual(generated.shape, (4, 50, 32))
            self.assertTrue(bool(torch.isfinite(generated).all()))


if __name__ == "__main__": unittest.main(verbosity=2)
