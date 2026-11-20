#!/usr/bin/env bash
# Event watcher for the MMTEB run on both boxes; one line per new event (state file dedups across re-arms).
STATE=~/embed-eval/logs/.mmteb_watch_seen; touch $STATE
emit() { grep -qxF -- "$1" $STATE || { echo "$1" >> $STATE; echo "$1"; }; }
BW="${BW_HOST:?set BW_HOST=user@host of the GPU box}"
while true; do
  while IFS= read -r l; do [ -n "$l" ] && emit "BW: $l"; done < <(ssh -o BatchMode=yes -o ConnectTimeout=10 $BW 'cd ~/embed-eval; grep -hE "END|STOPPED" logs/mmteb_queue.log; grep -h "SUBSET DONE" logs/run_mmteb_harrier-27b_D.log 2>/dev/null | cut -d" " -f2-6;
    for m in Qwen__Qwen3-Embedding-8B microsoft__harrier-oss-v1-27b; do f=$(ls results/mteb_cache/results/$m/*/MIRACLRetrievalHardNegatives.json 2>/dev/null); [ -n "$f" ] && python3 -c "import json,sys;print(\"MIRACL subsets $m:\", \",\".join(s[\"hf_subset\"] for s in json.load(open(sys.argv[1]))[\"scores\"][\"dev\"]))" $f; done;
    grep -lE "Traceback|OutOfMemory" logs/run_mmteb_*.log 2>/dev/null | sed "s/^/ERROR in /"' 2>/dev/null)
  while IFS= read -r l; do [ -n "$l" ] && emit "H100: $l"; done < <(cd ~/embed-eval; grep -hE "END|HANDOFF" logs/mmteb_h100_queue.log; grep -h "SUBSET DONE" logs/run_mmteb_h100_*.log | cut -d" " -f2-6; grep -lE "Traceback|OutOfMemory" logs/run_mmteb_h100_*.log 2>/dev/null | sed "s/^/ERROR in /")
  sleep 120
done
