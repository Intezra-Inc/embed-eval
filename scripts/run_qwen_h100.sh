#!/usr/bin/env bash
# Qwen3-Embedding-8B and -0.6B on the H100 box, same harness/venv as the Blackwell four-model run.
#   qwen3-8b : H100 GPU 3 now (idle), batch 8 + expandable segments (8B models OOM'd at batch 32 on long docs)
#   qwen3-0.6b: first of H100 GPUs 0-2 to become free (they run the Harrier quant code suite)
set -u
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID EE_ROOT=~/embed-eval PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
run() { local g=$1 key=$2 bs=$3
  log "START $key gpu$g"
  CUDA_VISIBLE_DEVICES=$g timeout 30h $P scripts/run_eval.py --model $key --batch-size $bs > logs/run_$key.log 2>&1
  log "END $key exit=$?"; }
run 3 qwen3-8b 8 > logs/gpu3_qwen3-8b.log 2>&1 &
(
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1<3 && $2<1000 {print $1; exit}')
    [ -n "$g" ] && break; sleep 60
  done
  sleep 30  # let the previous job's teardown finish
  run $g qwen3-0.6b 32
) > logs/gpuX_qwen3-0.6b.log 2>&1 &
wait
log "QWEN RUNS DONE" >> logs/gpu3_qwen3-8b.log
