#!/usr/bin/env bash
# Cost benchmark: every model on the SAME card (Blackwell GPU 0), one at a time, nothing else on that GPU.
# Batch 32 for all; on CUDA OOM retry at 16 then 8 (the batch actually used is recorded in cost/<model>.json).
set -u
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in harrier-0.6b qwen3-0.6b zembed-1 qwen3-8b nemotron-8b harrier-27b; do
  for bs in 32 16 8; do
    echo "[$(date -u +%FT%TZ)] START cost $m batch=$bs"
    timeout 2h venv/bin/python scripts/cost_bench.py run --model $m --batch-size $bs > logs/cost_$m.log 2>&1
    rc=$?
    if [ -f cost/$m.json ] && grep -q "\"batch_size\": $bs" cost/$m.json; then echo "[$(date -u +%FT%TZ)] END cost $m batch=$bs ok"; break; fi
    if grep -qE "OutOfMemory|out of memory" logs/cost_$m.log; then echo "  OOM at batch $bs, retrying smaller"; continue; fi
    echo "[$(date -u +%FT%TZ)] END cost $m batch=$bs FAILED rc=$rc"; break
  done
done
echo "[$(date -u +%FT%TZ)] COST DONE"
