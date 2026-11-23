# RoboFATE

Risk anticipation and corrective action for long-horizon manipulation.
This implementation follows the DM0.5 architecture, training objectives, and
main-study configuration in Sections 3.4-3.5 and Appendices A-D.

## Main Configuration

| Setting | Value |
| --- | --- |
| Collected corpus | 17 tasks, 100 dual-fate positives and 100 native-success negatives per task; 3,400 records |
| Family partition | 80/10/10 train/development/test; split seed 191126 |
| Training | 300,000 optimizer updates on 16 H200 workers |
| Action batch | 4 windows per worker, 1 accumulation step, global batch 64 |
| Decision batch | 64 histories of one class; positive and negative batches alternate across dual-fate updates |
| Action schedule | Original-only and dual-fate-only updates alternate 1:1 |
| Optimizer | Adafactor; backbone cap 0.001, other-module cap 0.005 |
| Learning-rate schedule | 2,000-step linear warmup, cosine decay to 5% at step 300,000 |
| Precision | FP32 trainable parameters, BF16 autocast, FP32 GRU |
| Checkpoints | Every 10,000 updates; retain latest three and best |
| Seeds | 191126, 191127, 191128 with the same family split |

Only SigLIP is frozen. The visual projection, language backbone, BAE,
Perceiver, controller, and MoAE are trained. On original-data updates,
MoAE uses unscaled shared FFNs; only those FFNs are updated within MoAE.
On dual-fate updates, MoAE uses one shared expert and Top-2 of eight
routed experts in each even-numbered action layer.

## Installation

Use Python 3.10 or newer and the CUDA 12.8 PyTorch wheels on the training
machines. Install the package from this directory:

```sh
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

OpenDM is included with its license in `third_party/`. Checkpoint and
corpus locations are supplied through command-line arguments. The source
package contains no model weights or episode images.

## Data Preparation

The recorded-data format is defined in `docs/DATA_FORMAT.md`. The 3,400
collected positive/negative records are partitioned by source family before
window formation. Original demonstrations use the same family assignment
where identities overlap. Native-success negatives supply decision
supervision only.

```sh
python -m robofate.prepare \
  --manifest data/recorded/manifest.json \
  --checkpoint weights/dm05 \
  --output data/prepared
```

`--cache-vision --device cuda:0` caches frozen SigLIP outputs on an allocated
device. Both cached-feature and raw-image inputs recompute the trainable
visual projection, backbone prefixes, Perceiver, and risk condition.

## Distributed Training

Set the following variables for the allocated job on each worker node:
`CUDA_VISIBLE_DEVICES`, `NNODES`, `GPUS_PER_NODE`, `NODE_RANK`,
`MASTER_ADDR`, and `MASTER_PORT`. `NNODES * GPUS_PER_NODE` must equal 16.
The launcher uses only the listed local devices and does not allocate or
connect to other machines.

```sh
bash scripts/train.sh \
  --config configs/robofate_dm05.json \
  --manifest data/prepared/manifest.json \
  --pretrained weights/dm05 \
  --output runs/seed191126 --seed 191126
```

Repeat with seeds 191127 and 191128 using the same prepared manifest and
separate output directories. To resume, pass `--resume` with a saved
checkpoint and choose a new output directory. Model weights, optimizer
state, per-rank random states, sampling position, and the earlier best
checkpoint are preserved.

## Selection And Evaluation

Each checkpoint is scored on development families. For each task, the score
is class-balanced decision BCE plus source-balanced normalized teacher
reconstruction MSE. The final selection averages tasks equally and resolves
ties in favor of the earlier update. Reconstruction uses noise seeds
31001-31003 and ten Euler steps. The selected checkpoint is then evaluated
once on the test partition, with `CUDA_VISIBLE_DEVICES` set to the
allocated evaluation device:

```sh
python -m robofate.evaluation \
  --manifest data/prepared/manifest.json \
  --pretrained weights/dm05 --run runs/seed191126
```

`robofate.policy.TakeoverPolicy` provides the policy-query interface. The
host supplies current observations and the native action-backbone input;
the policy exposes at most 25 commands per query. The first controller
check is query 8, after 175 applied actions. After takeover it stores the
risk once, clears pending BAE commands, and stops the Perceiver and
controller for the rest of the episode. See `docs/INFERENCE.md`.

Native RoboDojo trial records are aggregated with:

```sh
python -m robofate.metrics --trials evaluation/trials.jsonl --output evaluation/summary.json
```

The evaluator checks all 2,550 admitted trials, including unsuccessful
terminations, and uses capability-weighted rather than pooled success rate.

## Unit Tests And Code Map

```sh
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

Tests use synthetic CPU tensors and records. They cover corpus quotas,
family isolation, query alignment, batch construction, sparse routing,
gradients, masking, numerical precision, checkpoint selection, takeover,
and native trial accounting. Full-sized parameter counts are verified on
the metadata-only `meta` device; integration tests use reduced-width native
DM0.5 modules with random weights.

- `robofate/model.py`: Perceiver, controller, MoAE, and native action objectives.
- `robofate/prepare.py`, `robofate/data.py`: preprocessing, splits, and sampling.
- `robofate/train.py`, `robofate/evaluation.py`: distributed optimization and selection.
- `robofate/policy.py`: one-time takeover and query-level state.
- `robofate/collection.py`: offline annotation and restoration validation.
- `robofate/metrics.py`: native trial aggregation.
- `docs/PAPER_MAP.md`: paper-to-code correspondence.

## Release Verification

Verify the source inventory with:

```sh
python scripts/verify_release.py
```
