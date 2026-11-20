#!/usr/bin/env bash
# Qwen3-Embedding-0.6B main run on H100 GPU 3 (freed by qwen3-8b). Replaces run_qwen_h100.sh's waiter,
# which only considered GPUs 0-2. Same settings: batch 32, expandable segments, 30h timeout.
set -u
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID EE_ROOT=~/embed-eval PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START qwen3-0.6b gpu3" >> logs/gpuX_qwen3-0.6b.log
CUDA_VISIBLE_DEVICES=3 timeout 30h venv/bin/python scripts/run_eval.py --model qwen3-0.6b --batch-size 32 > logs/run_qwen3-0.6b.log 2>&1
echo "[$(date -u +%FT%TZ)] END qwen3-0.6b exit=$?" >> logs/gpuX_qwen3-0.6b.log
echo "[$(date -u +%FT%TZ)] QWEN RUNS DONE" >> logs/gpu3_qwen3-8b.log
