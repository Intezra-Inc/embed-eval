#!/usr/bin/env bash
# H100 GPU1 (freed by Harrier mB): Qwen3-8B MIRACL languages from the end of qwen_m's queue; qwen_m is stopped after 'ko'.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START mmteb_h100_qwen_m2 gpu1" >> logs/mmteb_h100_queue.log
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/run_subsets.py --model qwen3-8b --task MIRACLRetrievalHardNegatives \
  --subsets yo sw zh fr de > logs/run_mmteb_h100_qwen_m2.log 2>&1
echo "[$(date -u +%FT%TZ)] END mmteb_h100_qwen_m2 exit=$?" >> logs/mmteb_h100_queue.log
