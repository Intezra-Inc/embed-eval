#!/usr/bin/env bash
# Nemotron helper on GPU 2 (freed by zembed-1): same harness/settings as the main nemotron run on GPU 1,
# taking tasks from the END of its list. Shared mteb cache: the main run skips tasks this one finishes.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START nemotron-8b helper (reverse tail) gpu2"
CUDA_VISIBLE_DEVICES=2 timeout 48h venv/bin/python scripts/run_eval.py --model nemotron-8b --batch-size 8 --label nemotron-8b_helper \
  --tasks SWEbenchCodeRetrieval FreshStackRetrieval WikiSQLRetrieval MBPPRetrieval DS1000Retrieval SyntheticText2SQL > logs/run_nemotron-8b_helper.log 2>&1
echo "[$(date -u +%FT%TZ)] END nemotron-8b helper exit=$?"
