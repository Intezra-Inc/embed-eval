#!/usr/bin/env bash
# Start the cost benchmark only once every evaluation process on this box has exited (clean throughput numbers).
cd ~/embed-eval
while pgrep -f '^venv/bin/python scripts/run_eval.py' > /dev/null; do sleep 120; done
sleep 60
echo "[$(date -u +%FT%TZ)] all evaluation runs finished; starting cost benchmark"
scripts/run_cost.sh
