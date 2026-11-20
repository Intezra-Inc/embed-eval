#!/usr/bin/env bash
# Stop qwen_m (H100 GPU2) after MIRACL 'ko'; its remaining languages (de fr zh sw yo) run on GPU1 (qwen_m2).
cd ~/embed-eval
until grep -q "SUBSET DONE qwen3-8b MIRACLRetrievalHardNegatives ko " logs/run_mmteb_h100_qwen_m.log; do sleep 60; done
for p in $(pgrep -f "^venv/bin/python scripts/run_subsets.py --model qwen3-8b"); do readlink /proc/$p/fd/1 | grep -q "run_mmteb_h100_qwen_m.log" && kill $p; done
echo "[$(date -u +%FT%TZ)] HANDOFF mmteb_h100_qwen_m stopped after ko" >> logs/mmteb_h100_queue.log
