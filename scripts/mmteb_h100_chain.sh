#!/usr/bin/env bash
# Hand-offs for Harrier-27B on the H100 box:
#  - stop H100 MIRACL run mA after 'fr' (de ko yo are on Blackwell GPU0) and mB after 'es' (fa zh are on Blackwell GPU0)
#  - then reuse those GPUs for Harrier's remaining non-MIRACL tasks (moved off the throttled Blackwell GPU3)
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
handoff() { label=$1; last=$2; gpu=$3; newlabel=$4; shift 4
  until grep -q "SUBSET DONE harrier-27b MIRACLRetrievalHardNegatives $last " logs/run_$label.log; do sleep 60; done
  for p in $(pgrep -f "^venv/bin/python scripts/run_subsets.py --model harrier-27b"); do
    readlink /proc/$p/fd/1 | grep -q "run_$label.log" && kill $p; done
  sleep 30
  echo "[$(date -u +%FT%TZ)] HANDOFF $label stopped after $last; START $newlabel gpu$gpu" >> logs/mmteb_h100_queue.log
  CUDA_VISIBLE_DEVICES=$gpu venv/bin/python scripts/run_eval.py --model harrier-27b --label $newlabel --batch-size 16 --tasks "$@" > logs/run_$newlabel.log 2>&1
  echo "[$(date -u +%FT%TZ)] END $newlabel exit=$?" >> logs/mmteb_h100_queue.log
}
handoff mmteb_h100_harrier_mB es 1 mmteb_h100_harrier_rest1 TRECCOVID &
handoff mmteb_h100_harrier_mA fr 0 mmteb_h100_harrier_rest2 WinoGrande SpartQA AILAStatutes ArguAna &
wait
