#!/usr/bin/env bash
# Qwen3-8B MIRACL 'de' on free H100 GPU2 in parallel; qwen_m2 (GPU1) is stopped after 'fr'.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START mmteb_h100_qwen_de gpu2" >> logs/mmteb_h100_queue.log
CUDA_VISIBLE_DEVICES=2 venv/bin/python scripts/run_subsets.py --model qwen3-8b --task MIRACLRetrievalHardNegatives --subsets de > logs/run_mmteb_h100_qwen_de.log 2>&1 &
until grep -q "SUBSET DONE qwen3-8b MIRACLRetrievalHardNegatives fr " logs/run_mmteb_h100_qwen_m2.log; do sleep 30; done
for p in $(pgrep -f "^venv/bin/python scripts/run_subsets.py --model qwen3-8b"); do readlink /proc/$p/fd/1 | grep -q "run_mmteb_h100_qwen_m2.log" && kill $p; done
echo "[$(date -u +%FT%TZ)] HANDOFF mmteb_h100_qwen_m2 stopped after fr" >> logs/mmteb_h100_queue.log
wait
echo "[$(date -u +%FT%TZ)] END mmteb_h100_qwen_de" >> logs/mmteb_h100_queue.log
