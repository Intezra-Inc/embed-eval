#!/usr/bin/env bash
# Full benchmark: MTEB(Code, v1) + RTEB(Code, beta) public tasks, all four models, BF16, sentence-transformers.
# GPU 0: harrier-27b (part A)   GPU 1: nemotron-8b   GPU 2: zembed-1
# GPU 3: harrier-0.6b, then harrier-27b (part B). All four GPUs are identical RTX PRO 6000 Blackwell Max-Q cards.
set -u
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID
P=venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
run() { local g=$1 label=$2 to=$3; shift 3
  log "START $label gpu$g"
  CUDA_VISIBLE_DEVICES=$g timeout "$to" $P scripts/run_eval.py --label "$label" "$@" > logs/run_$label.log 2>&1
  log "END $label exit=$?"; }
A="COIRCodeSearchNetRetrieval AppsRetrieval CodeEditSearchRetrieval CodeSearchNetRetrieval CodeTransOceanContest CodeTransOceanDL CosQA StackOverflowQA HumanEvalRetrieval DS1000Retrieval MBPPRetrieval WikiSQLRetrieval FreshStackRetrieval SWEbenchCodeRetrieval"
B="CodeSearchNetCCRetrieval CodeFeedbackST CodeFeedbackMT SyntheticText2SQL"
run 0 harrier-27b_A 20h --model harrier-27b --tasks $A > logs/gpu0.log 2>&1 &
run 1 nemotron-8b 16h --model nemotron-8b > logs/gpu1.log 2>&1 &
run 2 zembed-1 16h --model zembed-1 > logs/gpu2.log 2>&1 &
( run 3 harrier-0.6b 6h --model harrier-0.6b; run 3 harrier-27b_B 20h --model harrier-27b --tasks $B ) > logs/gpu3.log 2>&1 &
wait
log "ALL RUNS DONE" >> logs/gpu0.log
