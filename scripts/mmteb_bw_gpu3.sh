#!/usr/bin/env bash
# Blackwell GPU3 (cool, freed by harrier part B): Harrier-27B's non-MIRACL tasks from part A's list. Shared cache, so part A skips them.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START mmteb_harrier-27b_C gpu3" >> logs/mmteb_queue.log
CUDA_VISIBLE_DEVICES=3 venv/bin/python scripts/run_eval.py --model harrier-27b --label mmteb_harrier-27b_C --batch-size 16 \
  --tasks StatcanDialogueDatasetRetrieval TRECCOVID ArguAna AILAStatutes SpartQA WinoGrande > logs/run_mmteb_harrier-27b_C.log 2>&1
echo "[$(date -u +%FT%TZ)] END mmteb_harrier-27b_C exit=$?" >> logs/mmteb_queue.log
