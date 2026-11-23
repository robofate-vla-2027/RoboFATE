"""Synthetic CPU fixtures; no pretrained weights, devices, API, or actuation."""
import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from robofate.config import PAPER, TASKS, TASK_GROUPS, load_config, update_kind, decision_class, lr_ratio
from robofate.data import (assign_families, validate_records, query_history, choose_window,
                           collate_contexts, to_device, safe_path, Corpus)
from robofate.model import MoFFN, PerceiverController, RoboFATE, masked_mse, flow_inputs
from robofate.prepare import action_window
from robofate.collection import validate_annotation, compare_restored_state
from robofate.evaluation import aggregate_sources, is_better
from robofate.train import optimizer_for, set_learning_rates, retained_checkpoints
from robofate.policy import TakeoverPolicy
from robofate.metrics import main_study_metrics


def record(task, kind, i, family=None):
    return {"task": task, "kind": kind, "source_id": f"synthetic_{task}_{kind}_{i}",
            "family_id": family or f"synthetic_family_{task}_{kind}_{i}"}


def full_records():
    return [record(task, kind, i) for task in TASKS for kind in ("dual_fate", "native_success") for i in range(100)]


def context(length=6, value=1):
    return {"input_ids": torch.full((1, length), value, dtype=torch.long),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "token_type_ids": torch.zeros(1, length, dtype=torch.long),
            "state_token_mask": torch.ones(1, length, dtype=torch.bool),
            "pixel_values": torch.zeros(3, 4, 8), "visual_features": True}


class ConfigurationTests(unittest.TestCase):
    def test_paper_defaults(self):
        c = load_config()
        self.assertEqual(c["training"]["world_size"], 16)
        self.assertEqual(c["training"]["microbatch"], 4)
        self.assertEqual(c["training"]["global_batch"], 64)
        self.assertEqual(c["training"]["max_steps"], 300000)
        self.assertEqual(c["data"]["collected_records"], 3400)
        self.assertEqual(c["training"]["action_exposure"], 300000 * 64)
    def test_configuration_file_equals_code(self):
        self.assertEqual(load_config(Path(__file__).parents[1]/"configs/robofate_dm05.json"), PAPER)
    def test_original_then_dual_schedule(self):
        self.assertEqual([update_kind(i) for i in range(1, 5)], ["original", "dual", "original", "dual"])
    def test_homogeneous_decision_class_alternation(self):
        self.assertEqual([decision_class(i) for i in (2, 4, 6, 8)], ["dual_fate", "native_success"] * 2)
        with self.assertRaises(ValueError): decision_class(1)
    def test_learning_rate_endpoints(self):
        self.assertEqual(lr_ratio(1, PAPER), 1 / 2000)
        self.assertEqual(lr_ratio(2000, PAPER), 1)
        self.assertAlmostEqual(lr_ratio(300000, PAPER), 0.05)
        self.assertGreater(lr_ratio(100000, PAPER), lr_ratio(200000, PAPER))
    def test_three_seeds_fixed_split(self):
        self.assertEqual(PAPER["training"]["seeds"], [191126, 191127, 191128])
        self.assertEqual(PAPER["data"]["split_seed"], 191126)
    def test_invalid_step(self):
        with self.assertRaises(ValueError): update_kind(0)
        with self.assertRaises(ValueError): lr_ratio(300001, PAPER)


class DataTests(unittest.TestCase):
    def test_3400_collected_sources(self):
        self.assertEqual(sum(validate_records(full_records()).values()), 3400)
    def test_missing_source_is_not_padded(self):
        with self.assertRaises(ValueError): validate_records(full_records()[:-1])
    def test_duplicate_source_rejected(self):
        rows = full_records()
        rows[-1] = rows[0]
        with self.assertRaises(ValueError): validate_records(rows)
    def test_family_partition_80_10_10(self):
        rows = full_records()
        assignments = assign_families(rows)
        self.assertEqual([sum(v == s for v in assignments.values()) for s in ("train", "development", "test")], [2720, 340, 340])
    def test_original_inherits_family(self):
        rows = full_records()
        extra = record(TASKS[0], "original", 0, rows[0]["family_id"])
        self.assertEqual(assign_families(rows), assign_families(rows + [extra]))
    def test_order_independent_split(self):
        rows = full_records()
        self.assertEqual(assign_families(rows), assign_families(list(reversed(rows))))
    def test_single_family_stays_unsplit(self):
        rows = [record(TASKS[0], "dual_fate", i, "one_family") for i in range(100)]
        self.assertEqual(assign_families(rows), {"one_family": "train"})
    def test_three_families_keep_distinct_holdouts(self):
        rows = [record(TASKS[0], "dual_fate", i) for i in range(3)]
        self.assertEqual(set(assign_families(rows).values()), {"train", "development", "test"})
    def test_query_history_first_check(self):
        self.assertEqual(query_history(175), list(range(0, 176, 25)))
        self.assertEqual(query_history(200), list(range(25, 201, 25)))
    def test_no_off_grid_or_early_history(self):
        for n in (30, 174, 176, -25):
            with self.assertRaises(ValueError): query_history(n)
    def test_early_late_equal_mass(self):
        windows = [{"suffix_offset": i * 50} for i in range(10)]
        rng = random.Random(1)
        early = sum(choose_window(windows, rng)["suffix_offset"] < 100 for _ in range(10000))
        self.assertTrue(4800 < early < 5200)
    def test_short_suffix_sampler(self):
        self.assertEqual(choose_window([{"suffix_offset": 0}], random.Random(1)), {"suffix_offset": 0})
    def test_four_sample_collation(self):
        batch = collate_contexts([context(i) for i in (5, 8, 6, 7)])
        self.assertEqual(batch["input_ids"].shape, (4, 8))
        self.assertEqual(batch["pixel_values"].shape, (12, 4, 8))
        self.assertEqual(batch["attention_mask"].sum().item(), 26)
        self.assertEqual(batch["state_token_mask"].dtype, torch.bool)
    def test_history_collation_varied_available_slots(self):
        a, b = context(5), context(8)
        for c in (a, b): c["history_mask"] = torch.zeros_like(c["input_ids"], dtype=torch.bool)
        a["history_pixel_values"] = torch.zeros(1, 4, 8)
        batch = collate_contexts([a, b])
        self.assertEqual(batch["history_mask"].shape, (2, 8))
        self.assertEqual(batch["history_pixel_values"].shape, (1, 4, 8))
    def test_tuple_device_transfer(self):
        values = to_device(([{"x": torch.ones(1)}], []), torch.device("cpu"))
        self.assertIsInstance(values, tuple)
        self.assertEqual(values[0][0]["x"].device.type, "cpu")
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError): safe_path("/a", "../b")
        with self.assertRaises(ValueError): safe_path("/a", "/b")
    def test_action_normalization_is_unclipped(self):
        commands = np.full((2, 14), 4.0, dtype=np.float32)
        action, mask = action_window(commands, 0, np.zeros(14), np.ones(14))
        self.assertGreater(float(action.max()), 1)
        self.assertEqual(mask.sum(), 28)
        self.assertFalse(mask[2:].any())
        restored = (action[:2, :14].numpy() + 1) * (1 + 1e-6) / 2
        self.assertTrue(np.allclose(restored, commands, atol=1e-6))


class ModuleTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(8)
    def mixture(self): return MoFFN(nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)), 8, 12)
    def test_upcycling_identity_four_samples(self):
        module = self.mixture()
        module.risk = torch.randn(4, 12)
        x = torch.randn(4, 50, 8)
        torch.testing.assert_close(module(x), module.shared(x))
        self.assertEqual(len({p.data_ptr() for p in module.parameters()}), len(list(module.parameters())))
    def test_top_two_balance_not_divided_by_k(self):
        module = self.mixture()
        nn.init.zeros_(module.router_x.weight); nn.init.zeros_(module.router_c.weight)
        module.risk = torch.zeros(4, 12)
        module(torch.randn(4, 50, 8))
        self.assertAlmostEqual(float(module.balance.detach()), 2.0)
    def test_original_unscaled_no_router(self):
        module = self.mixture(); module.original = True
        x = torch.randn(4, 50, 8)
        torch.testing.assert_close(module(x), module.shared(x))
        module(x).sum().backward()
        self.assertIsNone(module.balance)
        self.assertIsNone(module.router_x.weight.grad)
        self.assertTrue(all(p.grad is None for e in module.experts for p in e.parameters()))
    def test_action_gradient_reaches_risk(self):
        module = self.mixture()
        with torch.no_grad():
            for i, expert in enumerate(module.experts): expert[-1].bias.add_(i * 0.1)
        module.risk = torch.randn(4, 12, requires_grad=True)
        module(torch.randn(4, 50, 8)).square().mean().backward()
        self.assertGreater(module.risk.grad.norm(), 0)
        self.assertGreater(module.router_c.weight.grad.norm(), 0)
    def test_controller_batch_four_and_all_history_gradients(self):
        c = PerceiverController(width=16, queries=4, heads=4, hidden=8, window=8)
        contexts = [(torch.randn(4, 7, 16, requires_grad=True), torch.zeros(4, 7, dtype=torch.long), None) for _ in range(8)]
        logit, risk = c(contexts)
        self.assertEqual(logit.shape, (4,)); self.assertEqual(risk.shape, (4, 16))
        (logit.square().mean() + risk.square().mean()).backward()
        self.assertTrue(all(x[0].grad.norm() > 0 for x in contexts))
        self.assertGreater(c.queries.grad.norm(), 0)
    def test_gru_fp32_under_bf16(self):
        c = PerceiverController(width=16, queries=4, heads=4, hidden=8)
        seen = []
        hook = c.gru.register_forward_hook(lambda m, inputs, out: seen.append((inputs[0].dtype, out[1].dtype)))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            c.predict(torch.randn(4, 8, 16))
        hook.remove()
        self.assertEqual(seen, [(torch.float32, torch.float32)])
    def test_streaming_summary_equivalence(self):
        c = PerceiverController(width=16, queries=4, heads=4, hidden=8).eval()
        contexts = [(torch.randn(4, 7, 16), torch.zeros(4, 7, dtype=torch.long), None) for _ in range(8)]
        a = c(contexts)
        b = c.predict(torch.stack([c.summarize(*x) for x in contexts], dim=1))
        for x, y in zip(a, b): torch.testing.assert_close(x, y, rtol=0, atol=0)
    def test_padding_does_not_change_summary(self):
        c = PerceiverController(width=16, queries=4, heads=4, hidden=8).eval()
        x = torch.randn(1, 5, 16); sources = torch.zeros(1, 5, dtype=torch.long)
        a = c.summarize(x, sources)
        padded = torch.cat([x, torch.randn(1, 3, 16)], dim=1)
        b = c.summarize(padded, torch.zeros(1, 8, dtype=torch.long), torch.tensor([[False]*5 + [True]*3]))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    def test_mask_excludes_tail_and_coordinates(self):
        target = torch.randn(4, 50, 32); prediction = torch.randn_like(target)
        mask = torch.zeros_like(target, dtype=torch.bool); mask[:, :3, :14] = True
        first = masked_mse(prediction, target, mask)
        prediction[~mask] = 1e6
        self.assertEqual(first, masked_mse(prediction, target, mask))
    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError): masked_mse(torch.zeros(1, 50, 32), torch.zeros(1, 50, 32), torch.zeros(1, 50, 32, dtype=torch.bool))
    def test_flow_unobserved_coordinates_are_noise(self):
        action = torch.randn(4, 50, 32); mask = torch.zeros_like(action, dtype=torch.bool)
        mask[:, :, :14] = True
        x, target, time = flow_inputs(action, mask)
        torch.testing.assert_close(x[~mask], (target + action)[~mask])
        self.assertTrue(bool(((time >= 0.001) & (time <= 1)).all()))
    def test_original_update_allowlist(self):
        self.assertTrue(RoboFATE.original_update_allowed("dm.model.vlm.weight"))
        self.assertTrue(RoboFATE.original_update_allowed("moae.layers.1.mlp.shared.weight"))
        for name in ("mo_out.weight", "moae.layers.0.mlp.weight", "controller.gru.weight_ih_l0", "moae.layers.1.mlp.router_c.weight"):
            self.assertFalse(RoboFATE.original_update_allowed(name))
    def test_optimizer_none_skips_state_and_parameters(self):
        p = nn.Parameter(torch.ones(2, 2))
        optimizer = torch.optim.Adafactor([p], lr=0.01, foreach=False)
        p.grad = torch.ones_like(p); optimizer.step()
        value = p.detach().clone(); state = copy.deepcopy(optimizer.state_dict())
        p.grad = None; optimizer.step()
        torch.testing.assert_close(p, value, rtol=0, atol=0)
        for key, item in state["state"][0].items(): torch.testing.assert_close(optimizer.state_dict()["state"][0][key], item)


class SelectionTests(unittest.TestCase):
    def test_task_class_and_source_balancing(self):
        rows = [dict(task="a", kind="dual_fate", decision_bce=2., action_mse=4., recall=1.),
                dict(task="a", kind="native_success", decision_bce=6., false_positive_rate=0.),
                dict(task="b", kind="dual_fate", decision_bce=1., action_mse=1., recall=1.),
                dict(task="b", kind="native_success", decision_bce=1., false_positive_rate=0.)]
        self.assertEqual(aggregate_sources(rows, tasks=("a", "b"))["selection_score"], 5.)
    def test_tie_uses_earlier_checkpoint(self):
        self.assertFalse(is_better(1., 20000, 1., 10000))
        self.assertTrue(is_better(1., 10000, 1., 20000))
    def test_retention_latest_three_and_best(self):
        self.assertEqual(retained_checkpoints([10000,20000,30000,40000,50000],10000), {10000,30000,40000,50000})
    def test_selection_nan_rejected(self):
        with self.assertRaises(ValueError): is_better(float("nan"), 10000, None, None)


class PolicyTests(unittest.TestCase):
    def fake(self):
        class Fake:
            def __init__(self): self.summaries = 0; self.checks = 0; self.decoders = []; self.risks = []; self.controller = self
            def eval(self): return self
            def summarize(self, context): self.summaries += 1; return torch.ones(1, 4)
            def predict(self, x): self.checks += 1; return torch.tensor([1.]), torch.arange(4.)[None]
            def generate(self, context, risk, **kwargs):
                self.decoders.append(kwargs["decoder"]); self.risks.append(risk)
                return torch.zeros(1, 50, 32)
        return Fake()
    def test_one_time_takeover_at_query_eight(self):
        m = self.fake(); p = TakeoverPolicy(m, np.zeros(14), np.ones(14))
        for n in range(0, 251, 25): p.query(n, {}, {}, 500 - n)
        self.assertEqual(m.decoders[:7], ["bae"] * 7)
        self.assertEqual(m.decoders[7:], ["moae"] * 4)
        self.assertEqual(m.checks, 1); self.assertEqual(m.summaries, 8)
        self.assertTrue(all(r is p.fixed_risk for r in m.risks[7:]))
    def test_remaining_budget_caps_commands(self):
        p = TakeoverPolicy(self.fake(), np.zeros(14), np.ones(14))
        self.assertEqual(p.query(0, {}, {}, 3)["commands_available"], 3)
        for _ in range(3): self.assertEqual(p.next_command().shape, (14,))
        with self.assertRaises(RuntimeError): p.next_command()
    def test_stop_clears_queue(self):
        p = TakeoverPolicy(self.fake(), np.zeros(14), np.ones(14)); p.query(0, {}, {}, 500); p.stop()
        with self.assertRaises(RuntimeError): p.next_command()
    def test_off_grid_or_skipped_query_rejected(self):
        for n in (1, 50):
            p = TakeoverPolicy(self.fake(), np.zeros(14), np.ones(14)); p.query(0, {}, {}, 500)
            with self.assertRaises(ValueError): p.query(n, {}, {}, 500)
    def test_failed_query_and_stop_require_reset(self):
        p = TakeoverPolicy(self.fake(), np.zeros(14), np.ones(14)); p.query(0, {}, {}, 500)
        with self.assertRaises(ValueError): p.query(1, {}, {}, 499)
        with self.assertRaises(RuntimeError): p.next_command()
        with self.assertRaises(RuntimeError): p.query(25, {}, {}, 475)
        p.reset(); self.assertEqual(p.query(0, {}, {}, 500)["decoder"], "bae")
    def test_restored_boundary_prime(self):
        m = self.fake(); p = TakeoverPolicy(m, np.zeros(14), np.ones(14)); p.prime([{}] * 7, 175)
        self.assertEqual(p.query(175, {}, {}, 100)["decoder"], "moae")
    def test_reset_removes_episode_local_risk(self):
        m = self.fake(); p = TakeoverPolicy(m, np.zeros(14), np.ones(14)); p.prime([{}] * 7, 175); p.query(175, {}, {}, 100)
        p.reset(); self.assertFalse(p.takeover); self.assertIsNone(p.fixed_risk)


class AnnotationTests(unittest.TestCase):
    def annotation(self):
        return {"accept": True, "boundary_action": 175, "history_indices": list(range(0, 176, 25)),
                "risk_description": "synthetic tilted support", "causal_evidence_indices": [175],
                "retrospective_evidence_indices": [250], "remaining_actions": 325, "decision_reason": "synthetic boundary"}
    def test_eligible_annotation(self):
        self.assertTrue(validate_annotation(self.annotation(), [175], range(501), 250, 500))
    def test_future_leakage_rejected(self):
        a = self.annotation(); a["causal_evidence_indices"] = [250]
        with self.assertRaises(ValueError): validate_annotation(a, [175], range(501), 250, 500)
    def test_missing_query_history_rejected(self):
        with self.assertRaises(ValueError): validate_annotation(self.annotation(), [175], [175,250], 250, 500)
    def test_rejection_is_not_success(self):
        a = self.annotation(); a.update(accept=False, boundary_action=None, history_indices=[])
        self.assertFalse(validate_annotation(a, [175], range(501), 250, 500))
    def test_restoration_requires_solver_state(self):
        self.assertFalse(compare_restored_state({"rgb": {}}, {"rgb": {}})["qualified"])


class MetricTests(unittest.TestCase):
    def rows(self):
        return [{"seed": seed, "task": task, "variant": variant, "trial_id": f"synthetic_{i}", "success": True, "score": 1.0}
                for seed in (0,1,2) for task in TASKS
                for variant in (("standard", "randomized") if task in TASK_GROUPS["Generalization"] else ("standard",))
                for i in range(25 if task in TASK_GROUPS["Generalization"] else 50)]
    def test_native_trial_count_and_aggregation(self):
        report = main_study_metrics(self.rows())
        self.assertEqual(report["admitted_trials"], 2550); self.assertEqual(report["success_rate"], 100.)
    def test_missing_trials_not_removed_from_denominator(self):
        with self.assertRaises(ValueError): main_study_metrics(self.rows()[:-1])
    def test_timeout_cannot_count_as_success(self):
        rows = self.rows(); rows[0]["outcome"] = "timeout"
        with self.assertRaises(ValueError): main_study_metrics(rows)


if __name__ == "__main__": unittest.main(verbosity=2)
