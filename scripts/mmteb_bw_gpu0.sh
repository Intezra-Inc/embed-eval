#!/usr/bin/env bash
# Blackwell GPU0 (cool, freed by Nemotron): Harrier-27B MIRACL languages taken from the END of the two H100 queues
# (mA ends ... de ko yo, mB ends ... fa zh); the H100 runs are stopped before they reach these.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%FT%TZ)] START mmteb_harrier-27b_D gpu0" >> logs/mmteb_queue.log
CUDA_VISIBLE_DEVICES=0 venv/bin/python scripts/run_subsets.py --model harrier-27b --task MIRACLRetrievalHardNegatives \
  --subsets zh yo fa ko de > logs/run_mmteb_harrier-27b_D.log 2>&1
echo "[$(date -u +%FT%TZ)] END mmteb_harrier-27b_D exit=$?" >> logs/mmteb_queue.log
