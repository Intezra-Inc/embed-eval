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

