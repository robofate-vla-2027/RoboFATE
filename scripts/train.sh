#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set the devices allocated to this worker node}"
: "${NNODES:?Set the number of allocated worker nodes}"
: "${GPUS_PER_NODE:?Set the number of allocated GPUs on this node}"
: "${NODE_RANK:?Set the worker node rank}"
: "${MASTER_ADDR:?Set the rendezvous address within the allocated job}"
: "${MASTER_PORT:?Set the rendezvous port for this job}"
if (( NNODES * GPUS_PER_NODE != 16 )); then
  printf '%s\n' 'The paper configuration requires 16 total workers.' >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#devices[@]} != GPUS_PER_NODE )); then
  printf '%s\n' 'Visible device count must equal GPUS_PER_NODE.' >&2
  exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
exec torchrun --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE" \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --max-restarts 0 -m robofate.train "$@"
