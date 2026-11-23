"""Family-isolated corpora and stateless, rank-aware batch sampling."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import torch
from .config import TASKS, decision_class, update_kind

KINDS = ("dual_fate", "native_success", "original")
SPLITS = ("train", "development", "test")


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Corpus paths must be relative to their manifest directory")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Corpus reference escapes its root")
    return path


def validate_records(records, require_counts=True):
    ids = set()
    counts = Counter()
    for r in records:
        for key in ("source_id", "family_id", "task", "kind"):
            if not isinstance(r.get(key), str) or not r[key]:
                raise ValueError("Missing source identity: " + key)
        if r["source_id"] in ids:
            raise ValueError("Duplicate source_id: " + r["source_id"])
        ids.add(r["source_id"])
        if r["task"] not in TASKS or r["kind"] not in KINDS:
            raise ValueError("Task or source type is not in the main-study corpus")
        counts[(r["task"], r["kind"])] += 1
    if require_counts:
        for task in TASKS:
            for kind in ("dual_fate", "native_success"):
                if counts[(task, kind)] != 100:
                    raise ValueError(f"{task}/{kind}: expected 100 collected sources, got {counts[(task, kind)]}")
    return counts


def assign_families(records, seed=191126):
    """Split complete families, stratifying their task/class membership."""
    validate_records(records, require_counts=False)
    families = defaultdict(list)
    for r in records:
        families[r["family_id"]].append(r)
    strata = defaultdict(list)
    for family, rows in families.items():
        # Original demonstrations sharing a collected family inherit its split.
        collected = [(r["task"], r["kind"]) for r in rows if r["kind"] != "original"]
        signature = tuple(sorted(set(collected or [(r["task"], "original") for r in rows])))
        strata[signature].append(family)
    result = {}
    for signature, members in sorted(strata.items()):
        members.sort(key=lambda f: hashlib.sha256(f"{seed}:{f}".encode()).hexdigest())
        n = len(members)
        if n < 3:
            sizes = [n, 0, 0]
        else:
            # Largest-remainder allocation, retaining one family per held-out split.
            sizes = [int(n * x) for x in (0.8, 0.1, 0.1)]
            order = sorted(range(3), key=lambda i: (-(n * (0.8, 0.1, 0.1)[i] - sizes[i]), i))
            for i in order[:n - sum(sizes)]:
                sizes[i] += 1
            for i in (1, 2):
                if sizes[i] == 0:
                    sizes[0] -= 1
                    sizes[i] = 1
        offset = 0
        for split, size in zip(SPLITS, sizes):
            for family in members[offset:offset + size]:
                result[family] = split
            offset += size
    return result


def query_history(boundary, prefix=25, window=8):
    if type(boundary) is not int or boundary % prefix or boundary < (window - 1) * prefix:
        raise ValueError("Boundary must be an eligible policy query with eight causal queries")
    return list(range(boundary - (window - 1) * prefix, boundary + 1, prefix))


def action_starts(first, length, stride=50):
    if not 0 <= first < length:
        raise ValueError("An action suffix must contain at least one command")
    return list(range(first, length, stride))


def choose_window(windows, rng, early_actions=100):
    early = [w for w in windows if w["suffix_offset"] < early_actions]
    later = [w for w in windows if w["suffix_offset"] >= early_actions]
    pool = (early if rng.random() < 0.5 else later) if early and later else (early or later)
    if not pool:
        raise ValueError("Action source has no valid target windows")
    return rng.choice(pool)


class Corpus:
    def __init__(self, manifest, require_counts=True):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("format") != "robofate-prepared-v1":
            raise ValueError("Expected a robofate-prepared-v1 manifest")
        if self.manifest.get("split_seed") != 191126:
            raise ValueError("The source-family partition seed is fixed at 191126")
        self.records = self.manifest["records"]
        validate_records(self.records, require_counts)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        expected = assign_families(self.records, self.manifest["split_seed"])
        self.pools = defaultdict(lambda: defaultdict(list))
        for r in self.records:
            if r["split"] != expected[r["family_id"]]:
                raise ValueError("Source-family partition differs from the fixed split")
            if r.get("admitted") is True:
                self.pools[(r["split"], r["kind"])][r["task"]].append(r)
        for pools in self.pools.values():
            for rows in pools.values():
                rows.sort(key=lambda r: r["source_id"])

    def require_coverage(self, split, kinds=KINDS):
        for kind in kinds:
            absent = set(TASKS) - set(self.pools[(split, kind)])
            if absent:
                raise ValueError(f"Missing admitted {split}/{kind} sources for: {sorted(absent)}")

    @lru_cache(maxsize=32)
    def source(self, relative):
        return torch.load(safe_path(self.root, relative), map_location="cpu", weights_only=True)

    @lru_cache(maxsize=32)
    def context(self, relative):
        value = torch.load(safe_path(self.root, relative), map_location="cpu", weights_only=True)
        if not isinstance(value, dict):
            raise ValueError("Context must contain tensors and a feature-mode flag")
        return value

    def history(self, refs):
        if len(refs) != 8:
            raise ValueError("Controller requires exactly eight causal query observations")
        return [self.context(ref) for ref in refs]

    def sample_record(self, kind, rng, split="train"):
        pools = self.pools[(split, kind)]
        task = rng.choice(sorted(pools))
        return rng.choice(pools[task])

    def batch(self, step, rank, config, seed):
        t = config["training"]
        original = update_kind(step) == "original"
        action_kind = "original" if original else "dual_fate"
        actions, decisions = [], []
        for slot in range(t["microbatch"]):
            global_slot = rank * t["microbatch"] + slot
            action_rng = random.Random(f"{seed}:{step}:{global_slot}:action")
            record = self.sample_record(action_kind, action_rng)
            payload = self.source(record["prepared"])
            windows = payload["windows"]
            window = action_rng.choice(windows) if original else choose_window(windows, action_rng)
            sample = {**window, "context": self.context(window["context"]), "source_id": record["source_id"]}
            if not original:
                sample["history"] = self.history(payload["intervention_history"])
            actions.append(sample)
            if not original:
                decision_rng = random.Random(f"{seed}:{step}:{global_slot}:decision")
                kind = decision_class(step)
                record = self.sample_record(kind, decision_rng)
                payload = self.source(record["prepared"])
                history = decision_rng.choice(payload["decision_histories"])
                decisions.append({"history": self.history(history), "label": float(kind == "dual_fate"),
                                  "source_id": record["source_id"]})
        return actions, decisions


def collate_contexts(contexts, pad_token_id=0):
    if not contexts:
        raise ValueError("Cannot collate an empty context batch")
    modes = {c.get("visual_features", False) for c in contexts}
    if len(modes) != 1:
        raise ValueError("A batch cannot mix cached features and raw images")
    length = max(c["input_ids"].shape[1] for c in contexts)
    batch = {"visual_features": modes.pop()}
    for key, fill in (("input_ids", pad_token_id), ("attention_mask", 0), ("token_type_ids", 0), ("state_token_mask", 0)):
        tensors = []
        for c in contexts:
            value = c[key]
            if value.shape[0] != 1:
                raise ValueError("Stored contexts are single-observation records")
            tensors.append(torch.nn.functional.pad(value, (0, length - value.shape[1]), value=fill))
        batch[key] = torch.cat(tensors)
    batch["pixel_values"] = torch.cat([c["pixel_values"] for c in contexts])
    history_masks, history_pixels = [], []
    for c in contexts:
        mask = c.get("history_mask")
        if mask is None:
            mask = torch.zeros_like(c["input_ids"], dtype=torch.bool)
        history_masks.append(torch.nn.functional.pad(mask, (0, length - mask.shape[1]), value=False))
        if c.get("history_pixel_values") is not None:
            history_pixels.append(c["history_pixel_values"])
    if any(c.get("history_mask") is not None for c in contexts):
        batch["history_mask"] = torch.cat(history_masks)
        batch["history_pixel_values"] = torch.cat(history_pixels) if history_pixels else None
    return batch


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value
