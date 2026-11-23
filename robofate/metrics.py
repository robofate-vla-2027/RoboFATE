"""Native trial accounting and capability-weighted RoboDojo aggregation."""
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path

from .config import TASK_GROUPS, TASKS


def main_study_metrics(rows):
    groups = defaultdict(list)
    identities = set()
    generalization = set(TASK_GROUPS["Generalization"])
    for row in rows:
        identity = (row["seed"], row["task"], row["variant"], row["trial_id"])
        if identity in identities:
            raise ValueError("Duplicate admitted trial")
        identities.add(identity)
        if row["task"] not in TASKS or row["seed"] not in (0, 1, 2):
            raise ValueError("Trial is outside the selected task/seed scope")
        allowed = ("standard", "randomized") if row["task"] in generalization else ("standard",)
        if row["variant"] not in allowed or type(row["success"]) is not bool:
            raise ValueError("Invalid native trial variant or terminal success flag")
        if row.get("outcome") in ("timeout", "early_failure", "safety_stop") and row["success"]:
            raise ValueError("Unsuccessful termination cannot count as success")
        if not isinstance(row["score"], (int, float)) or not math.isfinite(row["score"]):
            raise ValueError("Native partial-progress score must be finite")
        groups[(row["seed"], row["task"], row["variant"])].append(row)
    seeds = {}
    per_configuration = {}
    for seed in (0, 1, 2):
        tasks = {}
        for task in TASKS:
            variants = ("standard", "randomized") if task in generalization else ("standard",)
            rates, scores = [], []
            for variant in variants:
                cohort = groups[(seed, task, variant)]
                quota = 25 if task in generalization else 50
                if len(cohort) != quota:
                    raise ValueError(f"Expected {quota} admitted trials for {seed}/{task}/{variant}")
                rate = 100 * sum(r["success"] for r in cohort) / quota
                score = sum(r["score"] for r in cohort) / quota
                rates.append(rate)
                scores.append(score)
                per_configuration[f"{seed}/{task}/{variant}"] = {"success_rate": rate, "score": score, "trials": quota}
            tasks[task] = {"success_rate": sum(rates) / len(rates), "score": sum(scores) / len(scores)}
        capabilities = {
            name: {key: sum(tasks[task][key] for task in task_names) / len(task_names)
                   for key in ("success_rate", "score")}
            for name, task_names in TASK_GROUPS.items()
        }
        seeds[str(seed)] = {"tasks": tasks, "capabilities": capabilities,
                            **{key: sum(v[key] for v in capabilities.values()) / 5 for key in ("success_rate", "score")}}
    return {"admitted_trials": len(rows), "seeds": seeds, "configurations": per_configuration,
            **{key: sum(v[key] for v in seeds.values()) / 3 for key in ("success_rate", "score")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.trials).read_text().splitlines() if line.strip()]
    report = main_study_metrics(rows)
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
