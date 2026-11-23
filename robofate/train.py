"""Sixteen-worker main-study training with reproducible resume and development selection."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import re

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import load_config, lr_ratio, update_kind
from .data import Corpus, to_device
from .evaluation import evaluate, is_better
from .model import RoboFATE


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def optimizer_for(model, config):
    t = config["training"]
    backbone, others = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith("dm.model.vlm") else others).append(parameter)
    return torch.optim.Adafactor(
        [{"params": backbone, "lr": t["backbone_lr"], "label": "backbone"},
         {"params": others, "lr": t["other_lr"], "label": "action_and_added"}],
        beta2_decay=t["beta2_decay"], d=t["update_rms_clip"], weight_decay=t["weight_decay"], foreach=False)


def set_learning_rates(optimizer, step, config):
    ratio = lr_ratio(step, config)
    for group, key in zip(optimizer.param_groups, ("backbone_lr", "other_lr")):
        group["lr"] = config["training"][key] * ratio


def retained_checkpoints(steps, best_step, latest=3):
    return set(sorted(steps)[-latest:]) | ({best_step} if best_step is not None else set())


def checkpoint_save(model, optimizer, step, config, corpus, output, seed, best):
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state().cpu(),
                 "python": random.getstate()}
    states = [None] * world
    dist.all_gather_object(states, local_rng)
    if rank == 0:
        name = f"checkpoints/step_{step:06d}.pt"
        path = output / name
        temporary = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": config, "manifest_sha256": corpus.digest,
                    "seed": seed, "rank_rng": states, "best": best}, temporary)
        temporary.replace(path)
        write_json(output / "latest.json", {"step": step, "checkpoint": name})
        write_json(output / "best.json", best)
        candidates = {}
        for candidate in (output / "checkpoints").glob("step_*.pt"):
            match = re.fullmatch(r"step_(\d{6})\.pt", candidate.name)
            if match:
                candidates[int(match[1])] = candidate
        keep = retained_checkpoints(candidates, best["step"], config["training"]["keep_latest"])
        for candidate_step, candidate in candidates.items():
            if candidate_step not in keep:
                candidate.unlink()
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robofate_dm05.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=191126, choices=(191126, 191127, 191128))
    parser.add_argument("--resume", help="Checkpoint to continue in a new output directory")
    args = parser.parse_args()
    config = load_config(args.config)
    t = config["training"]
    world, rank, local_rank = (int(os.environ[name]) for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if world != t["world_size"]:
        raise ValueError("The main-study run requires exactly sixteen synchronized workers")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("CUDA_VISIBLE_DEVICES must name the externally allocated devices")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    output = Path(args.output).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        (output / "development").mkdir()
    dist.barrier()
    corpus = Corpus(args.manifest)
    corpus.require_coverage("train")
    corpus.require_coverage("development", ("dual_fate", "native_success"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = RoboFATE.from_pretrained(args.pretrained, config).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in trainable):
        raise ValueError("All trainable parameters must be FP32")
    optimizer = optimizer_for(model, config)
    start, best = 0, {"score": None, "step": None, "checkpoint": None}
    resume_rng = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["config"] != config or state["manifest_sha256"] != corpus.digest or state["seed"] != args.seed:
            raise ValueError("Resume configuration, corpus hash, and seed must match")
        if len(state["rank_rng"]) != world:
            raise ValueError("Resume requires the same distributed world size")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start, best, resume_rng = state["step"], state["best"], state["rank_rng"][rank]
        if not 0 <= start < t["max_steps"]:
            raise ValueError("Resume checkpoint is outside the active training schedule")
        # Preserve the earlier best model when continuing into a new run directory.
        if rank == 0 and best["checkpoint"]:
            import shutil
            original_root = Path(args.resume).resolve().parent.parent
            source = original_root / best["checkpoint"]
            if not source.is_file():
                raise FileNotFoundError("Resume run's selected checkpoint is missing")
            shutil.copy2(source, output / best["checkpoint"])
        del state
    ddp = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True,
                                  gradient_as_bucket_view=True, bucket_cap_mb=32)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    if resume_rng:
        torch.set_rng_state(resume_rng["cpu"])
        torch.cuda.set_rng_state(resume_rng["cuda"], local_rank)
        random.setstate(resume_rng["python"])
    model.train()
    if rank == 0:
        write_json(output / "run.json", {"config": config, "seed": args.seed,
                   "manifest_sha256": corpus.digest, "resume_step": start,
                   "total_parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in trainable)})
    log = (output / "training.jsonl").open("x") if rank == 0 else None
    for step in range(start + 1, t["max_steps"] + 1):
        original = update_kind(step) == "original"
        samples, decisions = to_device(corpus.batch(step, rank, config, args.seed), device)
        optimizer.zero_grad(set_to_none=True)
        set_learning_rates(optimizer, step, config)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = ddp(samples, decisions, original=original)
        if not bool(torch.isfinite(losses["loss"])):
            raise FloatingPointError("Nonfinite training objective")
        losses["loss"].backward()
        model.discard_inactive_gradients(original)
        norm = torch.nn.utils.clip_grad_norm_(trainable, t["gradient_clip"], error_if_nonfinite=True,
                                             foreach=False)
        optimizer.step()
        metrics = torch.stack([losses[k].detach().float() for k in ("loss", "moae", "bae", "bce", "balance")])
        dist.all_reduce(metrics)
        metrics /= world
        if rank == 0:
            row = {"step": step, "kind": update_kind(step), "global_action_batch": t["global_batch"],
                   **dict(zip(("loss", "moae", "bae", "bce", "balance"), metrics.cpu().tolist())),
                   "gradient_norm_before_clip": float(norm), "learning_rate_caps": [g["lr"] for g in optimizer.param_groups]}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if step % 100 == 0:
                print(json.dumps(row), flush=True)
        del losses, metrics, samples, decisions
        if step % t["checkpoint_every"] == 0:
            report = evaluate(model, corpus, "development", config, device)
            if is_better(report["selection_score"], step, best["score"], best["step"]):
                best = {"score": report["selection_score"], "step": step,
                        "checkpoint": f"checkpoints/step_{step:06d}.pt"}
            if rank == 0:
                write_json(output / "development" / f"step_{step:06d}.json", report)
            checkpoint_save(model, optimizer, step, config, corpus, output, args.seed, best)
    if rank == 0:
        log.close()
        write_json(output / "terminal.json", {"status": "completed", "updates": t["max_steps"], "best": best})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
