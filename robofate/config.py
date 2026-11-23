"""Paper configuration and invariant checks (Table 6 and Appendix B.1)."""
import copy
import json
from pathlib import Path

TASK_GROUPS = {
    "Generalization": ["make_toast", "pack_objects_into_box", "sweep_blocks",
                       "pour_liquid_into_cup", "sort_nesting_dolls_by_size", "arrange_largest_number"],
    "Precision": ["plug_in_charger", "build_tower", "insert_tubes"],
    "Long-Horizon": ["fill_pen_holder", "make_kong", "play_stacking_toy"],
    "Memory": ["swap_blocks", "imitate_sorting_sequence", "press_by_number"],
    "Open": ["classify_objects_by_language", "stack_blocks_by_language"],
}
TASKS = tuple(task for group in TASK_GROUPS.values() for task in group)
PAPER = {
    "model": {
        "summary_width": 2560, "risk_width": 2560, "queries": 64, "qformer_blocks": 2,
        "attention_heads": 20, "ffn_multiplier": 4, "gru_hidden": 512, "window": 8,
        "action_width": 1024, "action_layers": 34, "action_ffn_width": 4096,
        "experts": 8, "top_k": 2, "shared_coefficient": 0.5,
        "horizon": 50, "action_dim": 32, "supervised_dim": 14,
        "visual_tokens_per_view": 256, "max_context_tokens": 1536,
    },
    "data": {
        "tasks": list(TASKS), "positive_per_task": 100, "negative_per_task": 100,
        "collected_records": 3400, "family_split": [0.8, 0.1, 0.1], "split_seed": 191126,
        "action_stride": 50, "early_suffix_actions": 100,
        "history_slots": 20, "history_interval_actions": 25, "normalization_epsilon": 1e-6,
    },
    "training": {
        "world_size": 16, "microbatch": 4, "accumulation_steps": 1, "global_batch": 64,
        "decision_batch": 64, "max_steps": 300000, "warmup_steps": 2000,
        "minimum_lr_ratio": 0.05, "backbone_lr": 0.001, "other_lr": 0.005,
        "optimizer": "Adafactor", "beta2_decay": -0.8, "update_rms_clip": 1.0,
        "weight_decay": 0.0, "gradient_clip": 1.0, "balance_weight": 0.001,
        "checkpoint_every": 10000, "keep_latest": 3,
        "seeds": [191126, 191127, 191128], "parameter_dtype": "float32",
        "autocast_dtype": "bfloat16", "gru_dtype": "float32",
        "action_exposure": 19200000,
    },
    "inference": {"executed_prefix": 25, "control_hz": 25, "first_check": 8,
                  "check_interval": 1, "trigger_threshold": 0.5, "euler_steps": 10},
    "evaluation": {"noise_seeds": [31001, 31002, 31003], "trial_seeds": [0, 1, 2]},
}


def load_config(path=None):
    config = copy.deepcopy(PAPER) if path is None else json.loads(Path(path).read_text())
    if config != PAPER:
        raise ValueError("The main-study configuration must match configs/robofate_dm05.json")
    t = config["training"]
    if t["world_size"] * t["microbatch"] * t["accumulation_steps"] != t["global_batch"]:
        raise ValueError("Global batch must equal workers x microbatch x accumulation")
    return config


def update_kind(step):
    if step < 1:
        raise ValueError("Optimizer steps are one-based")
    return "original" if step % 2 else "dual"


def decision_class(step):
    if update_kind(step) != "dual":
        raise ValueError("Original updates do not have a decision batch")
    return "dual_fate" if (step // 2) % 2 else "native_success"


def lr_ratio(step, config):
    import math
    t = config["training"]
    if not 1 <= step <= t["max_steps"]:
        raise ValueError("Step is outside the training schedule")
    if step <= t["warmup_steps"]:
        return step / t["warmup_steps"]
    progress = (step - t["warmup_steps"]) / (t["max_steps"] - t["warmup_steps"])
    return t["minimum_lr_ratio"] + (1 - t["minimum_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
