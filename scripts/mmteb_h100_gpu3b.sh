#!/usr/bin/env bash
# H100 GPU3: Harrier-27B TRECCOVID (moved off throttled Blackwell GPU3). Batch 8 on the 80 GB card.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START mmteb_h100_harrier_trec gpu3" >> logs/mmteb_h100_queue.log
CUDA_VISIBLE_DEVICES=3 venv/bin/python scripts/run_eval.py --model harrier-27b --label mmteb_h100_harrier_trec --batch-size 8 \
  --tasks TRECCOVID > logs/run_mmteb_h100_harrier_trec.log 2>&1
echo "[$(date -u +%FT%TZ)] END mmteb_h100_harrier_trec exit=$?" >> logs/mmteb_h100_queue.log
