#!/usr/bin/env bash
# Re-run of the harrier-27b side-experiment job: attempt 1 failed in 7 s because models/harrier-oss-v1-27b
# was missing on this host (now a symlink to the checksummed /data/models/harrier-oss-v1-27b @ 0c0fc62).
# Waits until run_taskinstr_queue.sh has launched its last job (no GPU race), then takes the next free GPU.
set -u
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID EE_ROOT=~/embed-eval PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASKS="CodeSearchNetCCRetrieval CodeTransOceanContest CodeTransOceanDL CosQA AppsRetrieval SyntheticText2SQL"
until grep -q "START taskinstr qwen3-0.6b" logs/taskinstr_queue.log 2>/dev/null; do sleep 60; done
sleep 180
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000 {print $1; exit}')
  [ -n "$g" ] && break; sleep 60
done
mv -f logs/run_harrier-27b_taskinstr.log logs/run_harrier-27b_taskinstr.attempt1-missing-model.log 2>/dev/null
echo "[$(date -u +%FT%TZ)] START taskinstr harrier-27b (attempt 2) gpu$g" >> logs/taskinstr_queue.log
CUDA_VISIBLE_DEVICES=$g timeout 36h venv/bin/python scripts/run_eval.py --model harrier-27b --batch-size 16 --tasks $TASKS \
  --instructions configs/qwen3_task_prompts.json --label harrier-27b_taskinstr > logs/run_harrier-27b_taskinstr.log 2>&1
echo "[$(date -u +%FT%TZ)] END taskinstr harrier-27b (attempt 2) exit=$?" >> logs/taskinstr_queue.log
