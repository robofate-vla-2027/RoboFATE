"""Source-balanced reconstruction and task-balanced checkpoint selection."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import TASKS, load_config
from .data import Corpus, collate_contexts, to_device
from .model import RoboFATE


def aggregate_sources(rows, tasks=TASKS):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["task"]][row["kind"]].append(row)
    result = {}
    for task in tasks:
        positives = grouped[task]["dual_fate"]
        negatives = grouped[task]["native_success"]
        if not positives or not negatives:
            raise ValueError("Development/test task requires admitted positive and negative sources: " + task)
        positive_bce = sum(r["decision_bce"] for r in positives) / len(positives)
        negative_bce = sum(r["decision_bce"] for r in negatives) / len(negatives)
        bce = 0.5 * (positive_bce + negative_bce)
        mse = sum(r["action_mse"] for r in positives) / len(positives)
        recall = sum(r["recall"] for r in positives) / len(positives)
        fpr = sum(r["false_positive_rate"] for r in negatives) / len(negatives)
        result[task] = {"balanced_decision_bce": bce, "source_balanced_action_mse": mse,
                        "score": bce + mse, "positive_recall": recall, "negative_false_positive_rate": fpr,
                        "positive_sources": len(positives), "negative_sources": len(negatives)}
    return {"selection_score": sum(r["score"] for r in result.values()) / len(result), "tasks": result}


@torch.no_grad()
def evaluate(model, corpus, split, config, device, distributed=True):
    if split not in ("development", "test"):
        raise ValueError("Checkpoint evaluation uses development or test families")
    corpus.require_coverage(split, ("dual_fate", "native_success"))
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    rank = dist.get_rank() if distributed and dist.is_initialized() else 0
    records = sorted((r for r in corpus.records if r["split"] == split and r.get("admitted")
                      and r["kind"] in ("dual_fate", "native_success")), key=lambda r: r["source_id"])
    previous_mode = model.training
    model.eval()
    rows = []
    for index, record in enumerate(records):
        if index % world != rank:
            continue
        source = corpus.source(record["prepared"])
        losses, probabilities = [], []
        risk = None
        for refs in source["decision_histories"]:
            histories = to_device([corpus.history(refs)], device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit, current_risk = model.decision(histories)
            target = logit.new_full(logit.shape, float(record["kind"] == "dual_fate"))
            losses.append(float(F.binary_cross_entropy_with_logits(logit.float(), target)))
            probabilities.append(float(logit.float().sigmoid()[0]))
            if record["kind"] == "dual_fate":
                risk = current_risk
        row = {"source_id": record["source_id"], "family_id": record["family_id"],
               "task": record["task"], "kind": record["kind"],
               "decision_bce": sum(losses) / len(losses), "probabilities": probabilities}
        if record["kind"] == "dual_fate":
            error, count, windows = 0.0, 0, []
            for window in source["windows"]:
                context = to_device(collate_contexts([corpus.context(window["context"])]), device)
                target = window["action"].to(device)[None]
                mask = window["mask"].to(device)[None]
                for seed in config["evaluation"]["noise_seeds"]:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        prediction = model.generate(context, risk, seed=seed, steps=10)
                    squared = ((prediction.float() - target.float()).square() * mask).sum().item()
                    elements = int(mask.sum())
                    error += squared
                    count += elements
                    windows.append({"start": window["start"], "seed": seed,
                                    "valid_elements": elements, "squared_error": squared})
            if not count:
                raise ValueError("Evaluation source has no supervised teacher actions")
            row.update(action_mse=error / count, valid_elements_across_seeds=count,
                       recall=sum(p >= 0.5 for p in probabilities) / len(probabilities), windows=windows)
        else:
            row["false_positive_rate"] = sum(p >= 0.5 for p in probabilities) / len(probabilities)
        rows.append(row)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [r for part in gathered for r in part]
    model.train(previous_mode)
    report = aggregate_sources(rows)
    report.update(split=split, sources=rows, noise_seeds=config["evaluation"]["noise_seeds"],
                  solver_steps=10, threshold=0.5, manifest_sha256=corpus.digest)
    return report


def is_better(score, step, best_score, best_step):
    import math
    if not math.isfinite(score):
        raise ValueError("Checkpoint selection score must be finite")
    return best_score is None or score < best_score or (score == best_score and step < best_step)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the selected model once on held-out test families")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = Path(args.run)
    output = run / "test_evaluation.json"
    lock = run / "test_evaluation.started.json"
    config = load_config()
    corpus = Corpus(args.manifest)
    best = json.loads((run / "best.json").read_text())
    state = torch.load(run / best["checkpoint"], map_location="cpu", weights_only=True)
    if state["manifest_sha256"] != corpus.digest or state["config"] != config:
        raise ValueError("Checkpoint and evaluation corpus/configuration differ")
    if output.exists() or lock.exists():
        raise FileExistsError("The selected checkpoint already has a test evaluation receipt")
    with lock.open("x") as stream:
        json.dump({"checkpoint": best["checkpoint"], "step": best["step"], "split": "test"}, stream)
    device = torch.device(args.device)
    model = RoboFATE.from_pretrained(args.pretrained, config).to(device)
    model.load_state_dict(state["model"], strict=True)
    del state
    report = evaluate(model, corpus, "test", config, device, distributed=False)
    with output.open("x") as stream:
        json.dump({"checkpoint": best, **report}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
