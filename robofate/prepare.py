"""Convert recorded applied commands and native observations to query-aligned data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import load_config
from .data import action_starts, assign_families, query_history, safe_path, validate_records


def action_window(commands, start, q01, q99, horizon=50, dimension=32):
    targets = np.asarray(commands[start:start + horizon], dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 14 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("Applied command targets must be finite 14-coordinate vectors")
    normalized = 2 * (targets - q01) / (q99 - q01 + 1e-6) - 1
    action = torch.zeros(horizon, dimension)
    mask = torch.zeros(horizon, dimension, dtype=torch.bool)
    action[:len(targets), :14] = torch.from_numpy(normalized)
    mask[:len(targets), :14] = True
    return action, mask


class FrozenVision:
    def __init__(self, checkpoint, device):
        from safetensors import safe_open
        from transformers import SiglipVisionModel
        from opendm.model.dm05.dm05_arch import DM05Config
        checkpoint = Path(checkpoint)
        config = DM05Config.from_pretrained(checkpoint, local_files_only=True)
        self.model = SiglipVisionModel(config.vlm_config.vision_config)
        prefix = "model.vlm.model.vision_tower."
        state = {}
        for path in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = handle.get_tensor(key)
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
        self.device = device
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        self.sha256 = digest.hexdigest()

    @torch.no_grad()
    def __call__(self, pixels):
        pieces = []
        for chunk in pixels.split(8):
            pieces.append(self.model(chunk.to(self.device, dtype=torch.bfloat16)).last_hidden_state.cpu())
        return torch.cat(pieces)


class NativeProcessor:
    def __init__(self, checkpoint, visual=None):
        from transformers import AutoProcessor
        from opendm.data.transforms import ChatTokenization, Normalize
        self.checkpoint = Path(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.state_normalizer = Normalize(str(self.checkpoint / "norm_stats.json"), ["state"], pad_to_stats=True)
        self.tokenizers = {
            history: ChatTokenization(self.processor, image_keys=["images_1", "images_2", "images_3"],
                                      max_length=1536, is_history=history)
            for history in (False, True)
        }
        statistics = json.loads((self.checkpoint / "norm_stats.json").read_text())["norm_stats"]["action"]
        self.q01 = np.asarray(statistics["q01"], np.float32)
        self.q99 = np.asarray(statistics["q99"], np.float32)
        if self.q01.shape != (14,) or self.q99.shape != (14,) or not np.all(self.q99 > self.q01):
            raise ValueError("DM0.5 normalization must specify fourteen valid action coordinates")
        self.visual = visual
        tokenizer = self.processor.tokenizer
        self.state_marker = tokenizer.encode("States:", add_special_tokens=False)
        self.end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")

    def encode(self, raw, observation_root, index, with_history):
        from opendm.constants.robot import RobotStateDesc
        observations = {item["applied_actions"]: item for item in raw["observations"]}
        if len(observations) != len(raw["observations"]):
            raise ValueError("Repeated observation indices")
        current = observations[index]
        cameras = ("head", "left_wrist", "right_wrist")
        def image(observation, camera):
            with Image.open(safe_path(observation_root, observation["images"][camera])) as source:
                return source.convert("RGB")
        payload = {
            "prompt": raw["instruction"], "state": np.asarray(current["state"], np.float32),
            "meta_data": {"robot_type": "Dual ARX5", "speed": "0.5",
                          "state_desc": ([RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]) * 2},
            **{f"images_{i+1}": image(current, c) for i, c in enumerate(cameras)},
        }
        if payload["state"].shape != (14,) or not np.isfinite(payload["state"]).all():
            raise ValueError("Robot state must contain fourteen finite coordinates")
        payload = self.state_normalizer(payload)
        if with_history:
            past = list(range(index - 500, index, 25))
            payload["history_images"] = [image(observations[n], "head") if n in observations else None for n in past]
        value = self.tokenizers[with_history](payload)
        value = {k: v for k, v in value.items() if v is not None and k not in ("action", "action_mask")}
        ids = value["input_ids"][0].tolist()
        matches = [i for i in range(len(ids) - len(self.state_marker) + 1)
                   if ids[i:i + len(self.state_marker)] == self.state_marker]
        if len(matches) != 1:
            raise ValueError("Expected one native States: span")
        start = matches[0]
        stop = ids.index(self.end_of_turn, start)
        state_mask = torch.zeros_like(value["input_ids"], dtype=torch.bool)
        state_mask[:, start:stop] = True
        value["state_token_mask"] = state_mask
        if self.visual is not None:
            for key in ("pixel_values", "history_pixel_values"):
                if key in value:
                    value[key] = self.visual(value[key])
        value["visual_features"] = self.visual is not None
        return value


def prepare_record(record, raw_root, output, processor):
    path = safe_path(raw_root, record["episode"])
    raw = json.loads(path.read_text())
    if raw.get("action_interface") != "dm05_absolute_joint_command":
        raise ValueError("Expected pre-interpolation applied joint-command targets")
    if raw.get("native_success") is not True:
        raise ValueError("Training records must terminate in native task success")
    actions = raw["actions"]
    if [x["index"] for x in actions] != list(range(len(actions))):
        raise ValueError("Applied command record contains a gap or duplicate")
    observations = raw["observations"]
    if any(type(x["applied_actions"]) is not int for x in observations):
        raise ValueError("Observation indices count applied commands")
    indices = [x["applied_actions"] for x in observations]
    if indices != sorted(set(indices)):
        raise ValueError("Observation indices must be unique and ordered")
    if not raw.get("instruction") or not len(actions):
        raise ValueError("An episode requires a native instruction and applied commands")
    commands = np.asarray([x["applied_joint_command"] for x in actions], dtype=np.float32)
    if commands.shape != (len(actions), 14) or not np.isfinite(commands).all():
        raise ValueError("Malformed applied joint-command array")
    kind = record["kind"]
    first = record["boundary"] if kind == "dual_fate" else 0
    if kind == "dual_fate":
        query_history(first)
        if not first < raw["first_adverse_action"] or not first < raw["action_budget"]:
            raise ValueError("Intervention must precede the adverse event and retain budget")
        if raw.get("teacher_start") != first or len(actions) > raw["action_budget"]:
            raise ValueError("Teacher continuation must start at the annotated boundary within the native budget")
    name = hashlib.sha256(record["source_id"].encode()).hexdigest()[:24]
    directory = output / "sources" / name
    directory.mkdir(parents=True, exist_ok=False)
    context_refs = {}
    def context(index, history):
        key = (index, history)
        if key not in context_refs:
            value = processor.encode(raw, path.parent, index, history)
            target = directory / f"query_{index:06d}_{'action' if history else 'current'}.pt"
            torch.save(value, target)
            context_refs[key] = str(target.relative_to(output))
        return context_refs[key]
    def history(boundary):
        return [context(index, False) for index in query_history(boundary)]
    windows = []
    if kind != "native_success":
        for start in action_starts(first, len(commands)):
            action, mask = action_window(commands, start, processor.q01, processor.q99)
            windows.append({"start": start, "suffix_offset": start - first,
                            "context": context(start, True), "action": action, "mask": mask})
    histories = []
    if kind == "dual_fate":
        histories = [history(first)]
    elif kind == "native_success":
        for boundary in range(175, len(commands), 25):
            histories.append(history(boundary))
        if not histories:
            raise ValueError("Native-success source has no complete eligible decision history")
    payload = {"source_id": record["source_id"], "windows": windows,
               "decision_histories": histories,
               "intervention_history": histories[0] if kind == "dual_fate" else [],
               "raw_episode_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    target = directory / "source.pt"
    torch.save(payload, target)
    return str(target.relative_to(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-vision", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config()
    source = Path(args.manifest).resolve()
    manifest = json.loads(source.read_text())
    if manifest.get("format") != "robofate-recorded-v1":
        raise ValueError("Expected a robofate-recorded-v1 manifest")
    records = manifest["records"]
    validate_records(records)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    assignments = assign_families(records, config["data"]["split_seed"])
    visual = FrozenVision(args.checkpoint, args.device) if args.cache_vision else None
    processor = NativeProcessor(args.checkpoint, visual)
    converted = []
    for record in records:
        result = {k: record[k] for k in ("source_id", "family_id", "task", "kind")}
        result["split"] = assignments[record["family_id"]]
        try:
            result["prepared"] = prepare_record(record, source.parent, output, processor)
            result["admitted"] = True
        except (ValueError, KeyError, FileNotFoundError) as error:
            result.update(admitted=False, rejection=str(error))
        converted.append(result)
        print(json.dumps({"source_id": record["source_id"], "admitted": result["admitted"]}), flush=True)
    norm = Path(args.checkpoint) / "norm_stats.json"
    result = {"format": "robofate-prepared-v1", "split_seed": config["data"]["split_seed"],
              "collected_records": 3400, "records": converted,
              "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "normalization_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
              "vision_cache": "frozen_SigLIP" if visual else "raw_pixels",
              "vision_sha256": visual.sha256 if visual else None}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
