#!/usr/bin/env bash
# MMTEB(Multilingual, v2) retrieval subset (18 tasks), same harness/settings as the code benchmark (BF16, max len 8192, seed 42).
# GPU0 Nemotron-8B, GPU1 Qwen3-8B, GPUs 2+3 Harrier-27B (task list split in two). Shared mteb cache.
cd ~/embed-eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ALL="MIRACLRetrievalHardNegatives BelebeleRetrieval WikipediaRetrievalMultilingual MLQARetrieval StatcanDialogueDatasetRetrieval CovidRetrieval TwitterHjerneRetrieval TRECCOVID SCIDOCS ArguAna HagridRetrieval StackOverflowQA AILAStatutes LegalBenchCorporateLobbying LEMBPasskeyRetrieval SpartQA TempReasonL1 WinoGrande"
H_A="MIRACLRetrievalHardNegatives StatcanDialogueDatasetRetrieval TRECCOVID ArguAna StackOverflowQA AILAStatutes SpartQA WinoGrande"
H_B="BelebeleRetrieval WikipediaRetrievalMultilingual MLQARetrieval CovidRetrieval TwitterHjerneRetrieval SCIDOCS HagridRetrieval LegalBenchCorporateLobbying LEMBPasskeyRetrieval TempReasonL1"
run() { gpu=$1; model=$2; label=$3; bs=$4; shift 4
  ( echo "[$(date -u +%FT%TZ)] START $label gpu$gpu"
    CUDA_VISIBLE_DEVICES=$gpu venv/bin/python scripts/run_eval.py --model $model --label $label --batch-size $bs --tasks "$@" > logs/run_$label.log 2>&1
    echo "[$(date -u +%FT%TZ)] END $label exit=$?" ) >> logs/mmteb_queue.log 2>&1 &
}
run 0 nemotron-8b mmteb_nemotron-8b 8 $ALL
run 1 qwen3-8b mmteb_qwen3-8b 8 $ALL
run 2 harrier-27b mmteb_harrier-27b_A 16 $H_A
run 3 harrier-27b mmteb_harrier-27b_B 16 $H_B
wait
