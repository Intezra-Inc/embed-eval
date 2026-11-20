#!/usr/bin/env bash
# Stop Harrier part C (throttled GPU3) once StatcanDialogueDatasetRetrieval has all 4 split/subset scores; its remaining
# tasks (TRECCOVID, ArguAna, AILAStatutes, SpartQA, WinoGrande) run on the H100 box instead.
cd ~/embed-eval
f=$(ls results/mteb_cache/results/microsoft__harrier-oss-v1-27b/*/StatcanDialogueDatasetRetrieval.json)
until [ "$(python3 -c "import json,sys;print(sum(len(v) for v in json.load(open(sys.argv[1]))['scores'].values()))" $f)" -ge 4 ]; do sleep 60; done
sleep 30
pkill -f "^venv/bin/python scripts/run_eval.py --model harrier-27b --label mmteb_harrier-27b_C"
echo "[$(date -u +%FT%TZ)] STOPPED mmteb_harrier-27b_C after Statcan (rest moved to H100)" >> logs/mmteb_queue.log
