#!/usr/bin/env bash
# Replaces mmteb_h100_chain.sh (its non-MIRACL tasks were picked up by H100 GPU3 instead): only stops the H100 MIRACL runs
# before languages that Blackwell GPU0 covers (mA after 'fr': de ko yo on BW; mB after 'es': fa zh on BW).
cd ~/embed-eval
stop_after() { label=$1; last=$2
  until grep -q "SUBSET DONE harrier-27b MIRACLRetrievalHardNegatives $last " logs/run_$label.log; do sleep 60; done
  for p in $(pgrep -f "^venv/bin/python scripts/run_subsets.py --model harrier-27b"); do
    readlink /proc/$p/fd/1 | grep -q "run_$label.log" && kill $p; done
  echo "[$(date -u +%FT%TZ)] HANDOFF $label stopped after $last" >> logs/mmteb_h100_queue.log
}
stop_after mmteb_h100_harrier_mB es &
stop_after mmteb_h100_harrier_mA fr &
wait
