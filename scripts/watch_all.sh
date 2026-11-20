#!/usr/bin/env bash
# Event watcher for every run on both boxes. Emits one line per NEW event; already-reported events are kept
# in a state file so nothing is lost between monitor re-arms.
# Events: run completed (summary file written), run process gone without a summary (DIED),
#         tracebacks / OOM in run logs, GPU holding memory at 0% utilisation for ~10 checks (STALL?).
STATE=~/embed-eval/logs/.watch_seen
touch "$STATE"
declare -A idle
emit() { grep -qxF -- "$1" "$STATE" || { echo "$1" >> "$STATE"; echo "$1"; }; }
BW="${BW_HOST:?set BW_HOST=user@host of the GPU box}"
# label | box | summary file | process match (anchored, never matches this script)
RUNS=(
  "harrier-27b_A|BW|embed-eval/results/summaries/harrier-27b_A.json|python scripts/run_eval.py --label harrier-27b_A"
  "harrier-27b_B|BW|embed-eval/results/summaries/harrier-27b_B.json|python scripts/run_eval.py --label harrier-27b_B"
  "harrier-27b_C|BW|embed-eval/results/summaries/harrier-27b_C.json|python scripts/run_eval.py --model harrier-27b --label harrier-27b_C"
  # nemotron-8b main stopped deliberately 01:41 UTC after finishing 17/18 (its last task, SWEbench, was already running in nemotron-8b_helper)
  "nemotron-8b_helper|BW|embed-eval/results/summaries/nemotron-8b_helper.json|python scripts/run_eval.py --model nemotron-8b --batch-size 8 --label nemotron-8b_helper --tasks"
  "nemotron-8b_helper2|BW|embed-eval/results/summaries/nemotron-8b_helper2.json|python scripts/run_eval.py --model nemotron-8b --batch-size 8 --label nemotron-8b_helper2 --tasks"
  "zembed-1|BW|embed-eval/results/summaries/zembed-1.json|python scripts/run_eval.py --model zembed-1"
  "code-vllm-bf16|H100|/data/models/harrier-quant/eval/code-vllm-bf16/summary.json|python scripts/run_mteb.py --suite code --engine vllm --model /data/models/harrier-oss-v1-27b"
  "code-vllm-g64|H100|/data/models/harrier-quant/eval/code-vllm-w4a16-gptq-late12-attn-g64/summary.json|python scripts/run_mteb.py --suite code --engine vllm --model w4a16-gptq-late12-attn-g64"
  "code-vllm-w4a16|H100|/data/models/harrier-quant/eval/code-vllm-w4a16-gptq/summary.json|python scripts/run_mteb.py --suite code --engine vllm --model w4a16-gptq --label"
  "qwen3-0.6b|H100|~/embed-eval/results/summaries/qwen3-0.6b.json|python scripts/run_eval.py --model qwen3-0.6b"
)
while true; do
  bwfiles=$(ssh -o BatchMode=yes -o ConnectTimeout=10 $BW 'ls embed-eval/results/summaries/ 2>/dev/null; echo PROCS; ps -eo args; echo SMI; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits; echo ERR; grep -aHE "Traceback|OutOfMemory|CUDA out of memory" embed-eval/logs/run_*.log 2>/dev/null | grep -v -e attempt1 -e "falling back to CPU" | head -5' 2>/dev/null)
  bw_sum=$(sed -n '1,/^PROCS$/p' <<< "$bwfiles"); bw_ps=$(sed -n '/^PROCS$/,/^SMI$/p' <<< "$bwfiles")
  bw_smi=$(sed -n '/^SMI$/,/^ERR$/p' <<< "$bwfiles" | grep -E '^[0-9]'); bw_err=$(sed -n '/^ERR$/,$p' <<< "$bwfiles" | tail -n +2)
  h_ps=$(ps -eo args)
  for r in "${RUNS[@]}"; do
    IFS='|' read -r label box sum match <<< "$r"
    if [ "$box" = BW ]; then
      if grep -qx "$(basename $sum)" <<< "$bw_sum"; then emit "DONE $label (summary written)"; continue; fi
      grep -q -- "$match" <<< "$bw_ps" || emit "DIED? $label: process gone, no summary"
    else
      if [ -f "$sum" ]; then emit "DONE $label (summary written)"; continue; fi
      grep -q -- "$match" <<< "$h_ps" || emit "DIED? $label: process gone, no summary"
    fi
  done
  # side experiment (started by the queue as GPUs free up)
  for m in harrier-27b qwen3-8b harrier-0.6b qwen3-0.6b; do
    s=~/embed-eval/results/summaries/${m}_taskinstr.json
    [ -f "$s" ] && emit "DONE taskinstr $m (summary written)"
    grep -q "START taskinstr $m" ~/embed-eval/logs/taskinstr_queue.log 2>/dev/null && emit "STARTED taskinstr $m"
  done
  while IFS= read -r l; do [ -n "$l" ] && emit "BW ERROR: ${l:0:180}"; done <<< "$bw_err"
  while IFS= read -r l; do [ -n "$l" ] && emit "COST: $l"; done < <(ssh -o BatchMode=yes -o ConnectTimeout=10 $BW 'grep -aE "all evaluation runs finished|END cost|OOM|COST DONE" embed-eval/logs/cost_run.log 2>/dev/null' 2>/dev/null)
  while IFS= read -r l; do [ -n "$l" ] && emit "H100 ERROR: ${l:0:180}"; done < <(grep -aHE 'Traceback|OutOfMemory|out of memory' /data/models/harrier-quant/logs/mteb_code-*.log ~/embed-eval/logs/run_qwen*.log ~/embed-eval/logs/run_*_taskinstr.log 2>/dev/null | grep -v 'falling back to CPU' | head -5)
  smi_check() { local box=$1; while IFS=', ' read -r i mem util; do [ -z "$i" ] && continue; k="$box$i"
      if [ "$mem" -gt 2000 ] && [ "$util" -eq 0 ]; then idle[$k]=$(( ${idle[$k]:-0} + 1 )); [ "${idle[$k]}" -eq 10 ] && echo "STALL? $box GPU $i idle at 0% with ${mem} MiB for ~10 min ($(date -u +%H:%M))"
      else idle[$k]=0; fi; done; }
  smi_check BW <<< "$bw_smi"
  smi_check H100 < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  sleep 60
done
