#!/usr/bin/env bash
# H100 helpers for the MMTEB retrieval run (Blackwell GPUs 1-2 thermally throttled to ~600 MHz). ar+bn stay on Blackwell.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
q() { gpu=$1; label=$2; shift 2
  ( echo "[$(date -u +%FT%TZ)] START $label gpu$gpu"; CUDA_VISIBLE_DEVICES=$gpu venv/bin/python "$@" > logs/run_$label.log 2>&1
    echo "[$(date -u +%FT%TZ)] END $label exit=$?" ) >> logs/mmteb_h100_queue.log 2>&1 & }
M=MIRACLRetrievalHardNegatives
q 0 mmteb_h100_harrier_mA scripts/run_subsets.py --model harrier-27b --task $M --subsets ru fi ja sw hi fr de ko yo
q 1 mmteb_h100_harrier_mB scripts/run_subsets.py --model harrier-27b --task $M --subsets te en th id es fa zh
q 2 mmteb_h100_qwen_m scripts/run_subsets.py --model qwen3-8b --task $M --subsets te hi fa th ru fi id ja es en ko de fr zh sw yo
q 3 mmteb_h100_qwen_rest scripts/run_eval.py --model qwen3-8b --label mmteb_h100_qwen3-8b_rest --tasks BelebeleRetrieval WikipediaRetrievalMultilingual \
  MLQARetrieval StatcanDialogueDatasetRetrieval CovidRetrieval TwitterHjerneRetrieval TRECCOVID SCIDOCS ArguAna HagridRetrieval AILAStatutes \
  LegalBenchCorporateLobbying LEMBPasskeyRetrieval SpartQA TempReasonL1 WinoGrande
wait
