#!/usr/bin/env bash
# Harrier-27B helper on GPU 0 (freed by nemotron helper 2): runs part B's last queued task (SyntheticText2SQL) now,
# while part B (GPU 3) is still on CodeFeedbackMT. Shared mteb cache, so part B skips it if finished here first.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START harrier-27b helper (SyntheticText2SQL) gpu0"
CUDA_VISIBLE_DEVICES=0 timeout 24h venv/bin/python scripts/run_eval.py --model harrier-27b --label harrier-27b_C \
  --tasks SyntheticText2SQL > logs/run_harrier-27b_C.log 2>&1
echo "[$(date -u +%FT%TZ)] END harrier-27b helper exit=$?"
