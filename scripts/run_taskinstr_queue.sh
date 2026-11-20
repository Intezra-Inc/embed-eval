#!/usr/bin/env bash
# Side experiment: generic vs vendor task-specific instructions for the instruction-tuned models.
# Same harness; only the per-task query instruction changes (Qwen3-Embedding's published task_prompts.json,
# applied verbatim to both Qwen and Harrier). Results in results/mteb_cache_taskinstr (main cache untouched).
# Jobs start on H100 GPUs as they free up, after the main qwen3-0.6b run has claimed its GPU.
set -u
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID EE_ROOT=~/embed-eval PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=venv/bin/python
TASKS="CodeSearchNetCCRetrieval CodeTransOceanContest CodeTransOceanDL CosQA AppsRetrieval SyntheticText2SQL"
JOBS=("harrier-27b 16" "qwen3-8b 8" "harrier-0.6b 32" "qwen3-0.6b 32")
log() { echo "[$(date -u +%FT%TZ)] $*"; }
until grep -q "START qwen3-0.6b" logs/gpuX_qwen3-0.6b.log 2>/dev/null; do sleep 60; done
sleep 120
for j in "${JOBS[@]}"; do
  set -- $j; key=$1; bs=$2
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000 {print $1; exit}')
    [ -n "$g" ] && break; sleep 60
  done
  log "START taskinstr $key gpu$g"
  ( CUDA_VISIBLE_DEVICES=$g timeout 24h $P scripts/run_eval.py --model $key --batch-size $bs --tasks $TASKS \
      --instructions configs/qwen3_task_prompts.json --label ${key}_taskinstr > logs/run_${key}_taskinstr.log 2>&1
    log "END taskinstr $key exit=$?" ) &
  sleep 180  # let the job allocate GPU memory before picking the next free GPU
done
wait
log "TASKINSTR DONE"
