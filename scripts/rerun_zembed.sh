#!/usr/bin/env bash
# zembed-1 attempt 2: attempt 1 hit its 16h timeout during SWEbenchCodeRetrieval (task 18/18).
# Completed tasks are reused from the mteb cache; only the remaining task runs. 48h safety timeout.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START zembed-1 (attempt 2, remaining tasks) gpu2"
CUDA_VISIBLE_DEVICES=2 timeout 48h venv/bin/python scripts/run_eval.py --model zembed-1 > logs/run_zembed-1.log 2>&1
echo "[$(date -u +%FT%TZ)] END zembed-1 exit=$?"
