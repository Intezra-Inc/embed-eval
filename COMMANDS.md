# Exact commands (code-retrieval benchmark)

Host: RTX PRO 6000 Blackwell box, `~/embed-eval` (Qwen accuracy runs and the side experiment ran on the H100 box with the identical venv).
Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and pin one GPU per run with `CUDA_VISIBLE_DEVICES`.

```bash
# 1. Environment (versions frozen in env/freeze.txt)
uv venv --python 3.12 venv && VIRTUAL_ENV=$PWD/venv uv pip install "mteb==2.24.10" sentence-transformers transformers torch accelerate "huggingface_hub[hf_xet]"
#    reproduce exactly: VIRTUAL_ENV=$PWD/venv uv pip install -r env/freeze.txt

# 2. Models at pinned revisions (checksums: env/sha256_*.txt)
venv/bin/hf download nvidia/Nemotron-3-Embed-8B-BF16 --revision d1f2f25730bbd775b99b29185134bc86653bf2d1 --local-dir models/Nemotron-3-Embed-8B-BF16
venv/bin/hf download zeroentropy/zembed-1-embedding  --revision cf13c81f3274394053d166740294f7eea4586f7a --local-dir models/zembed-1-embedding
venv/bin/hf download microsoft/harrier-oss-v1-0.6b   --revision f9b9dc8d367d443f2479d27aa5d8d2850c0774ee --local-dir models/harrier-oss-v1-0.6b
venv/bin/hf download Qwen/Qwen3-Embedding-8B         --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af --local-dir models/Qwen3-Embedding-8B
venv/bin/hf download Qwen/Qwen3-Embedding-0.6B       --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --local-dir models/Qwen3-Embedding-0.6B
ln -s ~/harrier-oss-v1-27b models/harrier-oss-v1-27b   # microsoft/harrier-oss-v1-27b @ 0c0fc62f6d8af9e8604cb818c412301b103a0093, checksum-verified

# 3. Card checks (harness loading path vs model-card example scores)
for m in nemotron-8b zembed-1 qwen3-8b qwen3-0.6b; do CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/validate_cards.py $m; done

# 4. Benchmark: MTEB(Code, v1) + RTEB(Code, beta) public tasks (18), one model per GPU
CUDA_VISIBLE_DEVICES=0 venv/bin/python scripts/run_eval.py --model harrier-27b      # split into --tasks subsets A/B/C in practice (scripts/run_all.sh, harrier27b_helper.sh)
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venv/bin/python scripts/run_eval.py --model nemotron-8b --batch-size 8
CUDA_VISIBLE_DEVICES=2 venv/bin/python scripts/run_eval.py --model zembed-1
CUDA_VISIBLE_DEVICES=3 venv/bin/python scripts/run_eval.py --model harrier-0.6b
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venv/bin/python scripts/run_eval.py --model qwen3-8b --batch-size 8   # H100 box
CUDA_VISIBLE_DEVICES=3 venv/bin/python scripts/run_eval.py --model qwen3-0.6b                                                              # H100 box
#    (encode batch size changes throughput only; results are cached per task, so interrupted runs resume)

# 5. Side experiment: vendor task-specific instructions (instruction-tuned models; separate cache)
TASKS="CodeSearchNetCCRetrieval CodeTransOceanContest CodeTransOceanDL CosQA AppsRetrieval SyntheticText2SQL"
for m in harrier-27b qwen3-8b harrier-0.6b qwen3-0.6b; do
  venv/bin/python scripts/run_eval.py --model $m --tasks $TASKS --instructions configs/qwen3_task_prompts.json --label ${m}_taskinstr
done

# 6. Cost: same GPU for all models, batch 32 (16/8 only on OOM)
venv/bin/python scripts/cost_bench.py build
CUDA_VISIBLE_DEVICES=0 scripts/run_cost.sh

# 7. Report
venv/bin/python scripts/make_report.py && venv/bin/python scripts/build_report.py
```

Seed 42 throughout (mteb tasks, torch, cost-document sampling). Raw mteb result JSONs: `results/mteb_cache/results/` (main) and
`results/mteb_cache_taskinstr/results/` (side experiment). Summaries with per-run settings: `results/summaries/`.

## Multilingual retrieval (MMTEB v2 retrieval subset, 18 tasks) — report: `results_mmteb.md`

```bash
cd ~/embed-eval
# Main launch (Blackwell): GPU0 Nemotron, GPU1 Qwen3-8B, GPUs 2+3 Harrier-27B halves
scripts/mmteb_launch.sh
# After thermal throttling on Blackwell GPUs 1-2: remaining work moved to the H100 box (identical venv), per MIRACL language
scripts/mmteb_h100.sh                  # H100: Harrier MIRACL halves (GPU0/1), Qwen MIRACL (GPU2), Qwen other 16 tasks (GPU3)
scripts/mmteb_bw_gpu3.sh               # Blackwell GPU3: Harrier Statcan (later TRECCOVID etc. moved to H100)
scripts/mmteb_bw_gpu0.sh               # Blackwell GPU0: Harrier MIRACL zh yo fa ko de
scripts/mmteb_h100_gpu3.sh; scripts/mmteb_h100_gpu3b.sh   # H100 GPU3: Harrier ArguAna AILAStatutes SpartQA WinoGrande; TRECCOVID
scripts/mmteb_h100_gpu1.sh; scripts/mmteb_h100_qde.sh     # H100: Qwen MIRACL tail languages split across GPUs 1/2
# (hand-off/stop helpers: mmteb_h100_chain2.sh, mmteb_h100_stopq.sh, mmteb_bw_stopC.sh; watcher: mmteb_watch.sh)
# single-subset form used by all MIRACL helpers:
CUDA_VISIBLE_DEVICES=0 venv/bin/python scripts/run_subsets.py --model qwen3-8b --task MIRACLRetrievalHardNegatives --subsets de fr
# Merge (H100 caches copied to results/h100_mteb_cache{,_split}), completeness check, tables
venv/bin/python scripts/mmteb_report.py        # -> results/mmteb/{scores.json,tables.md,merged/}
venv/bin/python scripts/mmteb_extra.py         # -> results/mmteb/extra.json (subset means, Belebele breakdown)
venv/bin/python scripts/mmteb_vs_public.py     # run where the public results repo is cached (~/.cache/mteb) -> results/mmteb/vs_public.json
# Diagnostics: scripts/diag_speed.py (encode speed), diag_miracl.py (stack dumps), diag_tokens.py (tokens per language)
```
Logs: `logs/run_mmteb_*.log`, `logs/mmteb_queue.log` (Blackwell), `logs/h100/` (H100 runs).
