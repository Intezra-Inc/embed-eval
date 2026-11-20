#!/usr/bin/env bash
# H100 GPU3 (freed by Qwen's non-MIRACL run): Harrier-27B small English tasks, forward order (hand-off rest2 runs them in reverse;
# shared cache, only-missing). Batch 8 on the 80 GB PCIe card (batch size does not affect scores).
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START mmteb_h100_harrier_rest3 gpu3" >> logs/mmteb_h100_queue.log
CUDA_VISIBLE_DEVICES=3 venv/bin/python scripts/run_eval.py --model harrier-27b --label mmteb_h100_harrier_rest3 --batch-size 8 \
  --tasks ArguAna AILAStatutes SpartQA WinoGrande > logs/run_mmteb_h100_harrier_rest3.log 2>&1
echo "[$(date -u +%FT%TZ)] END mmteb_h100_harrier_rest3 exit=$?" >> logs/mmteb_h100_queue.log
