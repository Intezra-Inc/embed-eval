# Code-retrieval benchmark: six embedding models, BF16

Run 2026-10-12 to 2026-10-14. Evaluation only (no quantization). Models: microsoft/harrier-oss-v1-27b, nvidia/Nemotron-3-Embed-8B-BF16,
zeroentropy/zembed-1, microsoft/harrier-oss-v1-0.6b, plus (added on request) Qwen/Qwen3-Embedding-8B and Qwen/Qwen3-Embedding-0.6B.

**Multilingual follow-up (2026-10-16):** Nemotron-8B vs Qwen3-8B vs Harrier-27B on the 18 MMTEB(Multilingual, v2) retrieval tasks is in **`results_mmteb.md`**.

## Ranking and recommendation

Mean nDCG@10 over the **14 tie-robust tasks** of MTEB(Code, v1) + RTEB(Code, beta) public (the 4 tie-dominated tasks are excluded; see notes):

| Rank | Model | Mean, 14 tie-robust | Mean, all 18 | Zero-shot mean (5 tie-robust tasks no model declares as training data) | Docs/sec (same GPU) | Vector size |
|---|---|---|---|---|---|---|
| 1 | **Nemotron-3-Embed-8B** | **0.813** | **0.758** | **0.677** | 5.3 | 4096 |
| 2 | Harrier-27B | 0.791 | 0.733 | 0.632 | 1.7 | 5376 |
| 3 | Qwen3-Embedding-8B | 0.790 | 0.728 | 0.639 | 5.3 | 4096 |
| 4 | Qwen3-Embedding-0.6B | 0.764 | 0.702 | 0.578 | 35.0 | 1024 |
| 5 | Harrier-0.6B | 0.748 | 0.695 | 0.570 | 36.4 | 1024 |
| 6 | zembed-1 (4B) | 0.702 | 0.652 | 0.636 | 8.4 | 2560 |

**Recommendation: deploy Nemotron-3-Embed-8B for code retrieval.**
- It has the best quality on MTEB(Code), on RTEB code and on the zero-shot subset. It is the top model on 10 of the 14 tie-robust tasks.
- It costs the same as Qwen3-8B to run. Its BF16 weights (14.8 GiB) should fit a 24 GB card without quantization, though this was not tested on a 24 GB card. Context and batch would need limiting.
- Its lead is not only from training overlap. Nemotron declares Apps, CodeTransOcean, CosQA, StackOverflowQA, SyntheticText2SQL and WikiSQL as
  training data, but it still leads on the zero-shot subset (0.677 vs 0.639 for the next model).
- Licence: Nemotron is under **OpenMDW-1.1**, not an OSI licence. Check it against your use case. If it doesn't fit, **Qwen3-Embedding-8B**
  (Apache-2.0) is the best alternative at the same cost. On CodeSearchNetCC-style code-to-code search, give it task-specific instructions
  (side experiment below).

**Quantized Harrier-27B on a 24 GB card: worth it for multilingual retrieval, not for code alone.**
- Intezra's published 4-bit quantizations put Harrier-27B on a single 24 GB card with almost no retrieval loss: mean nDCG@10 −0.0015 vs BF16
  on these 18 code tasks for the recommended variant. Code, results and build logs are in
  [Intezra-Inc/harrier-quant](https://github.com/Intezra-Inc/harrier-quant); weights are at
  [IntezraInc/harrier-oss-v1-27b-W4A16-g64-FP8mix](https://huggingface.co/IntezraInc/harrier-oss-v1-27b-W4A16-g64-FP8mix) (20.3 GB),
  [IntezraInc/harrier-oss-v1-27b-W4A16-FP8edges](https://huggingface.co/IntezraInc/harrier-oss-v1-27b-W4A16-FP8edges) (18.5 GB, longer
  inputs) and, for llama.cpp, [IntezraInc/harrier-oss-v1-27b-GGUF](https://huggingface.co/IntezraInc/harrier-oss-v1-27b-GGUF).
- For code retrieval alone the BF16 ranking above carries over to the quantized model: it ties Qwen3-8B (0.791 vs 0.790) and trails
  Nemotron-8B (0.813), while running 3× slower (1.7 vs 5.3 docs/s) and producing 31% larger vectors.
- Where it pays off is **multilingual retrieval**: it has the best overall MMTEB(Multilingual, v2) retrieval mean of the three large models,
  holds up on cross-lingual search and low-resource scripts where Nemotron-8B breaks down, and is MIT-licensed and US-origin (Microsoft). See
  `results_mmteb.md`, "Origin, licence and a 24 GB card". It is also the top model here on **conversational code Q&A** (CodeFeedbackMT).

**Small models:** Qwen3-Embedding-0.6B (0.764) edges Harrier-0.6B (0.748). Both run at about 35 docs/s with 1024-dim vectors (4.1 GB per 1M
float32), about 7× faster than the 8B models. Harrier-0.6B is notably better on WikiSQL and CodeFeedback; Qwen3-0.6B is better on CodeSearchNetCC.

**zembed-1** is last on code here, despite its card claims. It was validated against its card example (difference 0.0007), and it uses its
documented prompts. Its weakest tasks are code-to-code and code-to-comment: CodeSearchNetCC 0.525, COIR CodeSearchNet 0.453.

**Tasks where the top model differs from the overall winner** (tie-robust tasks only; the full order per task is in the ranking-flips table):
- CodeFeedbackMT: Harrier-27B wins.
- CodeSearchNetCC: Qwen3-0.6B wins, and this task is highly instruction-sensitive.
- CodeSearchNet and COIR CodeSearchNet: Qwen3-8B wins.

Nemotron is 5th of 6 on COIR CodeSearchNet (0.720) and 4th on CodeSearchNetCC, so it is relatively weak on CodeSearchNet-family code↔comment
tasks. If those dominate your workload, test Qwen3-8B.

**Instruction sensitivity:** see the side experiment. With task-specific instructions, the instruction-tuned models (Harrier, Qwen) gain
+0.02 to +0.12 on the tie-robust subset, almost all of it on CodeSearchNetCC and, for Harrier-27B, Apps. The main tables use mteb's official
configuration, which gives them only a generic instruction on these code tasks.

## Setup

- **Harness:** mteb 2.24.10 + sentence-transformers 6.1.0 + transformers 5.19.0 + torch 2.14.1 (cu130), identical venv on both hosts
  (`env/freeze.txt`, `env/h100/freeze.txt`).
- **Models:** each loaded through its **official mteb ModelMeta loader**, with that model's registered prompts, instructions and pooling.
- **Settings:** BF16, max sequence length 8,192 for every model, seed 42.
- **Raw results:** every raw mteb result JSON is in `results/mteb_cache/results/`.
- **Pinned revisions** (weights are byte-identical to mteb's registered revisions where those differ):

| Model | Revision | Licence | Pooling / prompts used |
|---|---|---|---|
| harrier-oss-v1-27b | `0c0fc62` | MIT | last token; `Instruct: {task}\nQuery:` on queries (mteb generic task instruction on code tasks), none on documents |
| Nemotron-3-Embed-8B-BF16 | `d1f2f25` | OpenMDW-1.1 | mean pooling; `query: ` / `passage: `. **mteb registers max length 4096; the card documents 32,768; we used 8,192** |
| zembed-1 (repo now `zeroentropy/zembed-1-embedding`) | `cf13c81` | **Apache-2.0** | last token; chat-style `query` / `document` system prompts + `<\|im_end\|>` suffix. mteb's registry pins the pre-relicence revision `10378878` (CC-BY-NC-4.0); we used the Apache-2.0 revision (identical weights) |
| harrier-oss-v1-0.6b | `f9b9dc8` | MIT | as Harrier-27B (tokenizer appends `<\|endoftext\|>`) |
| Qwen3-Embedding-8B | `1d8ad4c` | Apache-2.0 | last token; `Instruct: {task}\nQuery:` (mteb generic task instruction on code tasks) |
| Qwen3-Embedding-0.6B | `97b0c61` | Apache-2.0 | as Qwen3-8B |

- **Card checks:** every model was checked against its model card's example scores through the harness's own loading path: Nemotron within
  0.0043, zembed 0.0007, Qwen3-8B 0.0024, Qwen3-0.6B 0.0014. Harrier was checked earlier by agreement with sentence-transformers and vLLM.
- **Hardware:** RTX PRO 6000 Blackwell Max-Q (96 GB) for the four original models and **all cost measurements, which ran on one card**.
  Qwen3-8B and Qwen3-0.6B accuracy runs were on H100s with the identical venv. BF16 embeddings agree across these GPUs at cosine 0.9999.
- **RTEB:** 7 public code datasets were run. **Skipped as private held-out sets:** Code1Retrieval, JapaneseCode1Retrieval.
  AppsRetrieval is in both benchmarks and was run once.

---

# Detailed results

### a) MTEB(Code, v1): nDCG@10

| Task | Harrier-27B | Nemotron-3-Embed-8B | zembed-1 (4B) | Harrier-0.6B | Qwen3-Embedding-8B | Qwen3-Embedding-0.6B | Top model |
|---|---|---|---|---|---|---|---|
| AppsRetrieval | 0.7228† | **0.9594**† | 0.5881? | 0.5825† | 0.9085 | 0.7599 | Nemotron-3-Embed-8B |
| CodeEditSearchRetrieval | 0.7590 | **0.8297** | 0.7458? | 0.6189 | 0.7554 | 0.6339 | Nemotron-3-Embed-8B |
| CodeFeedbackMT | **0.9519**† | 0.8700 | 0.7158? | 0.9347† | 0.9310 | 0.9042 | Harrier-27B |
| CodeFeedbackST | 0.8929† | **0.8990** | 0.8087? | 0.8775† | 0.8969 | 0.8650 | Nemotron-3-Embed-8B |
| CodeSearchNetCCRetrieval | 0.7904† | 0.7348 | 0.5254? | 0.7386† | 0.5936 | **0.8509** | Qwen3-Embedding-0.6B |
| CodeSearchNetRetrieval | 0.8921† | 0.8948 | 0.8357? | 0.8858† | **0.9180** | 0.8995 | Qwen3-Embedding-8B |
| CodeTransOceanContest | 0.9345 | **0.9513**† | 0.9041? | 0.8918 | 0.9392 | 0.8645 | Nemotron-3-Embed-8B |
| CodeTransOceanDL \* | 0.3468 | 0.3392† | 0.3321? | **0.3746** | 0.3381 | 0.3293 | Harrier-0.6B (tie-dominated) |
| CosQA \* | 0.3465 | 0.4056† | 0.2865? | 0.3940 | **0.4115** | 0.4075 | Qwen3-Embedding-8B (tie-dominated) |
| COIRCodeSearchNetRetrieval | 0.8621† | 0.7196 | 0.4531? | 0.8215† | **0.8809** | 0.8420 | Qwen3-Embedding-8B |
| StackOverflowQA | 0.9691 | **0.9768**† | 0.9237? | 0.9373 | 0.9469 | 0.8924 | Nemotron-3-Embed-8B |
| SyntheticText2SQL \* | **0.7607** | 0.7460† | 0.6663? | 0.6621 | 0.6235 | 0.5989 | Harrier-27B (tie-dominated) |
| **Mean** | 0.7691 | **0.7772** | 0.6488 | 0.7266 | 0.7620 | 0.7373 | |
| **Mean, 9 tie-robust tasks** | 0.8639 | **0.8706** | 0.7223 | 0.8099 | 0.8634 | 0.8347 | |

† = listed in the model's declared training data (mteb ModelMeta); ? = model declares no training data (overlap unknown).

\* Tie-dominated task (see notes): per-task nDCG@10 and rankings there reflect tie-breaking, not model quality.

### b) RTEB(Code, beta), public datasets: nDCG@10

| Task | Harrier-27B | Nemotron-3-Embed-8B | zembed-1 (4B) | Harrier-0.6B | Qwen3-Embedding-8B | Qwen3-Embedding-0.6B | Top model |
|---|---|---|---|---|---|---|---|
| AppsRetrieval | 0.7228† | **0.9594**† | 0.5881? | 0.5825† | 0.9085 | 0.7599 | Nemotron-3-Embed-8B |
| DS1000Retrieval \* | 0.6704 | **0.7633** | 0.6211? | 0.6103 | 0.6787 | 0.6062 | Nemotron-3-Embed-8B (tie-dominated) |
| HumanEvalRetrieval | 0.9724 | **1.0000** | 0.9719? | 0.8800 | 0.9922 | 0.9424 | Nemotron-3-Embed-8B |
| MBPPRetrieval | 0.9226 | **0.9564** | 0.8906? | 0.8936 | 0.9196 | 0.8799 | Nemotron-3-Embed-8B |
| WikiSQLRetrieval | 0.8994 | **0.9950**† | 0.8897? | 0.9563 | 0.8473 | 0.9259 | Nemotron-3-Embed-8B |
| FreshStackRetrieval | 0.4431 | **0.4846** | 0.4832? | 0.3859 | 0.4304 | 0.3770 | Nemotron-3-Embed-8B |
| SWEbenchCodeRetrieval | 0.0643 | **0.1166** | 0.0890? | 0.0724 | 0.0963 | 0.0587 | Nemotron-3-Embed-8B |
| **Mean** | 0.6707 | **0.7536** | 0.6477 | 0.6258 | 0.6961 | 0.6500 | |
| **Mean, 6 tie-robust tasks** | 0.6708 | **0.7520** | 0.6521 | 0.6284 | 0.6990 | 0.6573 | |

Skipped (private RTEB held-out sets, cannot be run locally): Code1Retrieval, JapaneseCode1Retrieval.
AppsRetrieval appears in both MTEB(Code, v1) and RTEB(Code); it was run once and appears in both tables.

### Zero-shot subset: tasks no model declares as training data

| Task | Harrier-27B | Nemotron-3-Embed-8B | zembed-1 (4B) | Harrier-0.6B | Qwen3-Embedding-8B | Qwen3-Embedding-0.6B | Top model |
|---|---|---|---|---|---|---|---|
| CodeEditSearchRetrieval | 0.7590 | **0.8297** | 0.7458? | 0.6189 | 0.7554 | 0.6339 | Nemotron-3-Embed-8B |
| DS1000Retrieval \* | 0.6704 | **0.7633** | 0.6211? | 0.6103 | 0.6787 | 0.6062 | Nemotron-3-Embed-8B (tie-dominated) |
| HumanEvalRetrieval | 0.9724 | **1.0000** | 0.9719? | 0.8800 | 0.9922 | 0.9424 | Nemotron-3-Embed-8B |
| MBPPRetrieval | 0.9226 | **0.9564** | 0.8906? | 0.8936 | 0.9196 | 0.8799 | Nemotron-3-Embed-8B |
| FreshStackRetrieval | 0.4431 | **0.4846** | 0.4832? | 0.3859 | 0.4304 | 0.3770 | Nemotron-3-Embed-8B |
| SWEbenchCodeRetrieval | 0.0643 | **0.1166** | 0.0890? | 0.0724 | 0.0963 | 0.0587 | Nemotron-3-Embed-8B |
| **Mean** | 0.6387 | **0.6918** | 0.6336 | 0.5769 | 0.6454 | 0.5830 | |
| **Mean, 5 tie-robust tasks** | 0.6323 | **0.6774** | 0.6361 | 0.5702 | 0.6388 | 0.5784 | |

### Ranking flips vs overall (overall = mean over the 14 tie-robust tasks; tie-dominated tasks excluded from flips)

Overall order: Nemotron-3-Embed-8B (0.8134) > Harrier-27B (0.7912) > Qwen3-Embedding-8B (0.7897) > Qwen3-Embedding-0.6B (0.7640) > Harrier-0.6B (0.7483) > zembed-1 (4B) (0.7018)

Mean over all 18 (incl. tie-dominated): Nemotron-3-Embed-8B 0.7579, Harrier-27B 0.7334, Qwen3-Embedding-8B 0.7282, Qwen3-Embedding-0.6B 0.7021, Harrier-0.6B 0.6954, zembed-1 (4B) 0.6517

| Task | Order on this task | Top-1 differs from overall |
|---|---|---|
| AppsRetrieval | Nemotron-3-Embed-8B 0.959 > Qwen3-Embedding-8B 0.908 > Qwen3-Embedding-0.6B 0.760 > Harrier-27B 0.723 > zembed-1 (4B) 0.588 > Harrier-0.6B 0.582 | no |
| CodeEditSearchRetrieval | Nemotron-3-Embed-8B 0.830 > Harrier-27B 0.759 > Qwen3-Embedding-8B 0.755 > zembed-1 (4B) 0.746 > Qwen3-Embedding-0.6B 0.634 > Harrier-0.6B 0.619 | no |
| CodeFeedbackMT | Harrier-27B 0.952 > Harrier-0.6B 0.935 > Qwen3-Embedding-8B 0.931 > Qwen3-Embedding-0.6B 0.904 > Nemotron-3-Embed-8B 0.870 > zembed-1 (4B) 0.716 | **yes** |
| CodeFeedbackST | Nemotron-3-Embed-8B 0.899 > Qwen3-Embedding-8B 0.897 > Harrier-27B 0.893 > Harrier-0.6B 0.878 > Qwen3-Embedding-0.6B 0.865 > zembed-1 (4B) 0.809 | no |
| CodeSearchNetCCRetrieval | Qwen3-Embedding-0.6B 0.851 > Harrier-27B 0.790 > Harrier-0.6B 0.739 > Nemotron-3-Embed-8B 0.735 > Qwen3-Embedding-8B 0.594 > zembed-1 (4B) 0.525 | **yes** |
| CodeSearchNetRetrieval | Qwen3-Embedding-8B 0.918 > Qwen3-Embedding-0.6B 0.899 > Nemotron-3-Embed-8B 0.895 > Harrier-27B 0.892 > Harrier-0.6B 0.886 > zembed-1 (4B) 0.836 | **yes** |
| CodeTransOceanContest | Nemotron-3-Embed-8B 0.951 > Qwen3-Embedding-8B 0.939 > Harrier-27B 0.935 > zembed-1 (4B) 0.904 > Harrier-0.6B 0.892 > Qwen3-Embedding-0.6B 0.865 | no |
| COIRCodeSearchNetRetrieval | Qwen3-Embedding-8B 0.881 > Harrier-27B 0.862 > Qwen3-Embedding-0.6B 0.842 > Harrier-0.6B 0.822 > Nemotron-3-Embed-8B 0.720 > zembed-1 (4B) 0.453 | **yes** |
| StackOverflowQA | Nemotron-3-Embed-8B 0.977 > Harrier-27B 0.969 > Qwen3-Embedding-8B 0.947 > Harrier-0.6B 0.937 > zembed-1 (4B) 0.924 > Qwen3-Embedding-0.6B 0.892 | no |
| HumanEvalRetrieval | Nemotron-3-Embed-8B 1.000 > Qwen3-Embedding-8B 0.992 > Harrier-27B 0.972 > zembed-1 (4B) 0.972 > Qwen3-Embedding-0.6B 0.942 > Harrier-0.6B 0.880 | no |
| MBPPRetrieval | Nemotron-3-Embed-8B 0.956 > Harrier-27B 0.923 > Qwen3-Embedding-8B 0.920 > Harrier-0.6B 0.894 > zembed-1 (4B) 0.891 > Qwen3-Embedding-0.6B 0.880 | no |
| WikiSQLRetrieval | Nemotron-3-Embed-8B 0.995 > Harrier-0.6B 0.956 > Qwen3-Embedding-0.6B 0.926 > Harrier-27B 0.899 > zembed-1 (4B) 0.890 > Qwen3-Embedding-8B 0.847 | no |
| FreshStackRetrieval | Nemotron-3-Embed-8B 0.485 > zembed-1 (4B) 0.483 > Harrier-27B 0.443 > Qwen3-Embedding-8B 0.430 > Harrier-0.6B 0.386 > Qwen3-Embedding-0.6B 0.377 | no |
| SWEbenchCodeRetrieval | Nemotron-3-Embed-8B 0.117 > Qwen3-Embedding-8B 0.096 > zembed-1 (4B) 0.089 > Harrier-0.6B 0.072 > Harrier-27B 0.064 > Qwen3-Embedding-0.6B 0.059 | no |

### c) Cost (RTX PRO 6000 Blackwell Max-Q, BF16, 5,000 code documents, max 8,192 tokens, same GPU for every model)

Batch 32 for every model except where it ran out of memory (then 16, then 8). Peak memory is dominated by the activations of the longest batch (p95 input = 8,192 tokens), not the weights; smaller batches or shorter inputs need far less.

| Model | Params | Weights (BF16) | Embedding dim | Batch | Peak GPU memory (torch reserved / nvidia-smi) | Docs/sec | Tokens/sec | Storage per 1M vectors, float32 | int8 |
|---|---|---|---|---|---|---|---|---|---|
| Harrier-27B | 27.01B | 50.3 GiB | 5376 | 16 | 77.9 / 78.6 GiB | 1.7 | 1,704 | 21.5 GB | 5.38 GB |
| Nemotron-3-Embed-8B | 7.95B | 14.8 GiB | 4096 | 32 | 84.9 / 85.6 GiB | 5.3 | 5,126 | 16.4 GB | 4.10 GB |
| zembed-1 (4B) | 4.02B | 7.5 GiB | 2560 | 32 | 72.8 / 73.5 GiB | 8.4 | 7,904 | 10.2 GB | 2.56 GB |
| Harrier-0.6B | 0.60B | 1.1 GiB | 1024 | 32 | 17.6 / 18.2 GiB | 36.4 | 34,321 | 4.1 GB | 1.02 GB |
| Qwen3-Embedding-8B | 7.57B | 14.1 GiB | 4096 | 32 | 85.2 / 85.9 GiB | 5.3 | 5,006 | 16.4 GB | 4.10 GB |
| Qwen3-Embedding-0.6B | 0.60B | 1.1 GiB | 1024 | 32 | 43.6 / 44.3 GiB | 35.0 | 33,045 | 4.1 GB | 1.02 GB |


---

## Side experiment: generic vs task-specific instructions (instruction-tuned models only)

For these code tasks, mteb 2.24 defines no task prompt, so Harrier and Qwen receive the generic `"Retrieve text based on user query."`.
Here each model instead got Qwen3-Embedding's **published per-task instructions** (`configs/qwen3_task_prompts.json`, SHA-256 `c9493103…`,
from QwenLM/Qwen3-Embedding), used verbatim for both families. Everything else was identical: same harness, sentence-transformers on both
sides of every comparison. Results are in `results/mteb_cache_taskinstr/` and `results/taskinstr_comparison.json`.

| Task | Harrier-27B: generic → task instr (Δ) | Qwen3-8B: generic → task instr (Δ) | Harrier-0.6B: generic → task instr (Δ) | Qwen3-0.6B: generic → task instr (Δ) |
|---|---|---|---|---|
| CodeSearchNetCCRetrieval | 0.790 → 0.916 (+0.125) | 0.594 → 0.961 (+0.368) | 0.739 → 0.918 (+0.180) | 0.851 → 0.914 (+0.063) |
| CodeTransOceanContest | 0.935 → 0.941 (+0.007) | 0.939 → 0.929 (-0.011) | 0.892 → 0.883 (-0.009) | 0.865 → 0.864 (-0.000) |
| CodeTransOceanDL \* | 0.347 → 0.354 (+0.007) | 0.338 → 0.335 (-0.003) | 0.375 → 0.364 (-0.011) | 0.329 → 0.320 (-0.010) |
| CosQA \* | 0.346 → 0.361 (+0.014) | 0.411 → 0.411 (-0.000) | 0.394 → 0.404 (+0.010) | 0.407 → 0.398 (-0.009) |
| AppsRetrieval | 0.723 → 0.791 (+0.068) | 0.908 → 0.909 (+0.000) | 0.582 → 0.585 (+0.003) | 0.760 → 0.751 (-0.009) |
| SyntheticText2SQL \* | 0.761 → 0.766 (+0.005) | 0.623 → 0.643 (+0.019) | 0.662 → 0.622 (-0.040) | 0.599 → 0.616 (+0.017) |
| **Mean, 6 tasks** | **0.650 → 0.688 (+0.038)** | **0.636 → 0.698 (+0.062)** | **0.607 → 0.629 (+0.022)** | **0.635 → 0.644 (+0.009)** |
| **Mean, 3 tie-robust tasks** | **0.816 → 0.883 (+0.067)** | **0.814 → 0.933 (+0.119)** | **0.738 → 0.795 (+0.058)** | **0.825 → 0.843 (+0.018)** |

\* tie-dominated task; its deltas reflect tie-breaking noise and are not meaningful.

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
