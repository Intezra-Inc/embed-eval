#!/usr/bin/env bash
# Nemotron attempt 2: OOM at batch 32 on CodeFeedbackMT (long docs); batch 8 + expandable segments. Completed tasks are reused from the mteb cache.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START nemotron-8b (attempt 2, batch 8) gpu1"
CUDA_VISIBLE_DEVICES=1 timeout 16h venv/bin/python scripts/run_eval.py --model nemotron-8b --batch-size 8 > logs/run_nemotron-8b.log 2>&1
echo "[$(date -u +%FT%TZ)] END nemotron-8b exit=$?"
