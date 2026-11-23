# Data Format

## Recorded Corpus

The input JSON object has `format: "robofate-recorded-v1"` and a `records`
array. Every record contains:

| Field | Meaning |
| --- | --- |
| `source_id` | Unique episode/source identifier |
| `family_id` | Shared identity for all branches, retries, and augmentations of one source/scene family |
| `task` | One of the seventeen identifiers in `configs/robofate_dm05.json` |
| `kind` | `dual_fate`, `native_success`, or `original` |
| `episode` | Episode JSON path relative to the recorded manifest directory |
| `boundary` | Dual-fate only: number of commands applied before teacher takeover |

Each task supplies 100 collected `dual_fate` and 100 `native_success`
records. `original` records are the separate demonstration replay corpus;
they are not counted toward 3,400. Family IDs are global across all three
source types. Source IDs must be unique. The partition is assigned before
windows are created, using seed 191126, with task/class stratification when
independent-family counts permit it. A single family never generates its
own development or test family by splitting windows.

## Episode JSON

| Field | Meaning |
| --- | --- |
| `instruction` | Native task instruction |
| `action_interface` | `dm05_absolute_joint_command` |
| `native_success` | Boolean terminal native-success result |
| `actions` | Ordered executed commands with zero-based `index` and 14-value `applied_joint_command` |
| `observations` | Ordered records containing `applied_actions`, raw 14-value `state`, and `images` |
| `images` | `head`, `left_wrist`, and `right_wrist` paths relative to the episode JSON |
| `teacher_start` | Dual-fate only: count of commands before the teacher's first action |
| `first_adverse_action` | Dual-fate only: first adverse-action boundary in the original failed branch |
| `action_budget` | Dual-fate only: original episode command limit |

Command order is left six joints, left gripper, right six joints, right
gripper. Joint values are in radians; gripper API commands use [0,1].
The targets are applied API commands before interpolation, not measured
future joint states. Pose-primitive records are converted through the
native IK/control record before entering this interface.

Observation index `n` means `n` commands have been applied. Command index
`n` is the next command after observation `n`. A successful dual branch
contains the recorded common prefix followed by the complete teacher
suffix. Its failed-branch record and annotation remain associated with
its source identity. Annotations and full-state comparisons can be
validated with `robofate.collection`.

## Temporal Alignment

The policy grid is `n = 25 * (i - 1)`. A positive boundary must be a multiple
of 25 and at least 175, with observations at `n-175, n-150, ..., n`.
The teacher action suffix starts at command `n`. Unsupported or missing
histories remain rejected entries in the preparation manifest rather than
being rounded, repeated, or relabeled.

The action backbone uses the current three views and twenty past head
slots at 25-command intervals, strictly before the current observation.
Unavailable past slots are represented by native masked history tokens.
The Perceiver receives current-only contexts; its temporal history is the
eight-query summary buffer.

## Action Windows And Sampling

Windows cover each admitted teacher suffix in contiguous 50-command chunks.
The final chunk has a valid-time mask; only the first fourteen of thirty-two
coordinates are supervised. Recorded holding commands retain their masks.
Action normalization is `2 * (a-q01)/(q99-q01+1e-6) - 1` without clipping.
Robot-state text uses native clipping and 256-bin discretization.

Action sampling first chooses a task uniformly and then a source uniformly.
Within a dual-fate source, windows beginning in its first 100 action slots
and remaining windows receive equal probability mass when both exist.
Original demonstrations are sampled separately. Controller decisions use
a separately sampled class-homogeneous batch, balanced by task and source;
negative sources with many eligible checks do not gain source probability.

## Prepared Corpus

Preparation writes `manifest.json` with `format: "robofate-prepared-v1"`.
It retains every collected record, its family partition and admission result,
with a `prepared` source-index path for admitted records. A source index
contains action windows, intervention-history references, and eligible
decision histories. Context tensors are stored separately and loaded on
demand with `weights_only=True`. Source-manifest and normalization hashes
are recorded with the preparation output. Frozen-feature caches also record
the SigLIP weight hash.

The trainer requires admitted training coverage for every task in all three
corpora, and admitted positive/negative development coverage for every task.
Test families are read only by the selected-checkpoint evaluation command.
