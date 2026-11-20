
---

## Side experiment: generic vs task-specific instructions (instruction-tuned models only)

For these code tasks, mteb 2.24 defines no task prompt, so Harrier and Qwen receive the generic `"Retrieve text based on user query."`.
Here each model instead got Qwen3-Embedding's **published per-task instructions** (`configs/qwen3_task_prompts.json`, SHA-256 `c9493103…`,
from QwenLM/Qwen3-Embedding), used verbatim for both families. Everything else was identical: same harness, sentence-transformers on both
sides of every comparison. Results are in `results/mteb_cache_taskinstr/` and `results/taskinstr_comparison.json`.

SIDE_EXPERIMENT_TABLE

- **CodeSearchNetCC (code-to-code) is where instructions matter:** +0.06 to +0.37, and all four models land between 0.91 and 0.96 with the
  instruction. Under the generic instruction, scores on this task mostly measure how badly each model handles the wrong framing.
- **Harrier-27B also gains +0.068 on Apps.** Elsewhere the effects are within ±0.02.
- **Instructions don't transfer cleanly between model families.** Qwen's prompts help Qwen most.
- **With instructions, our harness reproduces Qwen3-0.6B's published mteb scores** (CodeSearchNetCC 0.914 vs 0.917 published).
- **Practical advice:** if you deploy Harrier or Qwen for code search, set per-task instructions. Nemotron and zembed use fixed
  query/document prefixes and have no per-task instructions.

## Notes: failures, workarounds, and data issues

1. **Tie-dominated tasks.** In CodeTransOceanDL (100%), SyntheticText2SQL (94%), DS1000 (93%) and CosQA (85%), the relevant document has an
   identical-text **non-relevant** twin in the corpus (`results/tie_affected_tasks.json`; CodeFeedbackST 16%). Identical texts get identical
   embeddings, so whether the relevant copy ranks first is decided by the similarity search's tie-breaking, not by the model.
   Example: Harrier-27B's SyntheticText2SQL nDCG@10 was 0.761 in this harness and 0.608 through vLLM, with **identical recall@10**.
   These tasks stay in the "Mean, all 18" figures as specified, but they are excluded from the ranking, from the flip analysis and from the
   tie-robust means. They also likely explain gaps against published leaderboard numbers, e.g. Qwen3-0.6B SyntheticText2SQL 0.60 here vs 0.77
   published.
2. **zembed-1 on sentence-transformers 6.x.** Its custom `ZembedTransformer.tokenize()` (which appends `<|im_end|>\n`) is never called by
   ST 6.x, which uses `preprocess()`. Without a fix, pooling reads the wrong token: HumanEval scored 0.64, and the card example came out at
   0.85/0.59/0.83 instead of 0.75/0.57/0.68.
   - Fix: `scripts/zembed_compat.py` applies the same suffix through `preprocess()`. With it, the card example matches to 0.0007.
   - The invalid first HumanEval result is kept as `HumanEvalRetrieval.invalid-noshim.json`.
   - Over-length inputs lose the suffix, exactly as with the model's own code (it truncates after appending).
3. **Harrier-27B on transformers 5.19.** transformers maps `gemma3_text` to the multimodal processor. `scripts/st_compat.py` falls back to the
   tokenizer. Tokenization is unchanged.
4. **Nemotron out of memory** at encode batch 32 on CodeFeedbackMT (long multi-turn documents). It was rerun at batch 8 with
   `expandable_segments`; batch size doesn't change scores. Two helper processes later split its remaining tasks across freed GPUs, and its
   duplicate main process was stopped once its last task was already running elsewhere. All 18 tasks are complete.
5. **Time limits.** zembed-1 hit a 16-hour limit during SWE-bench, its last task. It was restarted, and its 17 cached tasks were reused. All
   other limits were removed before they could fire.
6. **Harrier-27B split across GPUs** (parts A, B and C) to finish sooner. Same model, settings and cache. Results are per task and don't
   depend on the GPU.
7. **Qwen models on H100s.** They were added mid-run and evaluated on H100s with the identical venv. Their cost numbers were measured on the
   same Blackwell card as everyone else's.
8. **Harrier-27B cost run** ran out of memory at batch 32 (50 GiB of weights plus activations for 32 × 8k-token documents). It used batch 16;
   all other models used 32.
9. **SWE-bench Code** scores are near the floor for every model (0.06–0.12). It is by far the slowest task (2–24 h per model) and has little
   influence on the ranking.

## Reproduce

`scripts/run_eval.py` (harness), `scripts/run_all.sh` (main launch), `scripts/cost_bench.py` + `scripts/run_cost.sh` (cost),
`scripts/validate_cards.py` (card checks), `scripts/make_report.py` (tables), `scripts/build_report.py` (this file), with `EE_ROOT` pointing at
this folder. Launch and recovery scripts actually used: `scripts/*.sh`. Package lists: `env/`. Model checksums: `env/sha256_*.txt`.
Logs: `logs/`.
