#!/usr/bin/env bash
# Second Nemotron helper on GPU 0 (freed by harrier-27b part A): the middle of nemotron's remaining list, forward order.
# Meets helper 1 (GPU 2, reverse from SWEbench); shared mteb cache, so tasks finished by one run are skipped by the others.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START nemotron-8b helper2 (middle, forward) gpu0"
CUDA_VISIBLE_DEVICES=0 timeout 48h venv/bin/python scripts/run_eval.py --model nemotron-8b --batch-size 8 --label nemotron-8b_helper2 \
  --tasks StackOverflowQA SyntheticText2SQL DS1000Retrieval MBPPRetrieval WikiSQLRetrieval FreshStackRetrieval > logs/run_nemotron-8b_helper2.log 2>&1
echo "[$(date -u +%FT%TZ)] END nemotron-8b helper2 exit=$?"
