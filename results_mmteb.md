# Multilingual retrieval: Nemotron-3-Embed-8B vs Qwen3-Embedding-8B vs Harrier-27B

Run 2026-10-15 to 2026-10-16. The task set is the 18 retrieval tasks of **MTEB(Multilingual, v2)**. The harness and settings are the same as the
code benchmark (`results.md`): mteb 2.24.10, each model's official mteb loader and prompts, BF16, max length 8,192, seed 42.
Scores use each task's official main score, computed with mteb's own `TaskResult.get_score`. All 18 tasks are complete for all three
models, covering every language subset and eval split.

## Summary

| | Nemotron-8B | Qwen3-8B | Harrier-27B |
|---|---|---|---|
| Mean, all 18 tasks | 0.760 | 0.707 | **0.781** |
| Mean, 7 multilingual / non-English tasks | 0.786 | 0.774 | **0.805** |
| Mean, 6 multilingual excl. Statcan (recall@10 task) | 0.834 | 0.830 | **0.837** |
| MIRACL hard negatives (18 languages) | **0.731** (wins 17/18) | 0.697 | 0.694 |
| Belebele, 122 monolingual language pairs | 0.831 | **0.874** | 0.854 |
| Belebele, 254 cross-lingual pairs (query and passage in different languages) | 0.820 | **0.896** | 0.871 |
| Mean, 14 standard retrieval tasks (excl. Passkey and the 3 RAR-b reasoning tasks) | 0.769 | **0.797** | 0.783 |
| MIRACL corpus size in each model's tokens (18 languages, ~2.65M passages) | 374M | 575M | **336M** |

**No single winner for multilingual retrieval.** On the six multilingual tasks the three models are within 0.007 of each other.
Which model is best depends on the kind of multilingual workload:

- **Search in major languages with natural queries (RAG over Wikipedia-style content): Nemotron-8B.**
  - It wins MIRACL on 17 of 18 languages, by +0.034 mean nDCG@10 over the next model. Its largest margins over the next-best model are Korean and Bengali (+0.058) and Chinese (+0.051). Against Qwen alone, its biggest lead is Swahili (+0.10).
  - It also wins MLQA (7 languages) and Wikipedia-multilingual (16 languages).
  - This matches the public RTEB picture from earlier: European and Japanese legal/enterprise retrieval.
- **Cross-lingual retrieval, or long-tail languages and scripts: Qwen3-8B.**
  - On Belebele cross-lingual pairs it leads Nemotron by 0.076 and Harrier by 0.025.
  - Nemotron breaks down on several scripts. Its scores fall below 0.6 on 24 of 122 Belebele languages, against 11 for Qwen and 9 for Harrier. Its worst gaps vs Qwen are Odia (−0.71), Lao (−0.54), Khmer (−0.52), Sinhala (−0.34), Amharic (−0.27) and Tibetan (−0.23).
  - Qwen also wins Chinese CovidRetrieval.
- **Harrier-27B has the best overall average.** Its lead comes from Danish (TwitterHjerne), English/French dialogue (Statcan) and the reasoning tasks, not from multilingual search.
  - It is never best on MIRACL except Yoruba.
  - It is second on Belebele.
  - It has the most compact tokenization: 10% fewer tokens than Nemotron on MIRACL.
  - It is still a 27B model, about 3× slower per token than the 8B models in the code benchmark's cost measurements.

**The "all 18" mean is driven by a few non-multilingual tasks. Read it with care:**
- **RAR-b reasoning tasks.** SpartQA, TempReasonL1 and WinoGrande are reasoning tasks posed as retrieval. Under mteb's configured instructions, Qwen3-8B
  scores 0.156 / 0.013 / 0.550 on them, against 0.66–0.88 / 0.44–0.53 / 0.89–0.93 for the others. The public leaderboard has the same Qwen
  numbers (0.157 / 0.013 / 0.542), so this is how the model behaves under that configuration, not an error here.
  - Excluding these three tasks and Passkey, **Qwen3-8B has the best mean (0.797)**. That mean includes the four English tasks AILAStatutes, ArguAna, SCIDOCS and TRECCOVID, where Qwen leads.
- **LEMBPasskey truncation.** LEMBPasskeyRetrieval has documents up to 32k tokens, and every model was truncated at 8,192, so all three score about 0.84.
  The public Harrier result (0.975) evidently used a longer context.

**Tokenizer cost on non-Latin scripts.** Qwen's tokenizer needs 2.4–3.3× more tokens than Nemotron's for Bengali, Hindi and Telugu, and 1.8×
for Persian. Across the MIRACL corpora that is 54% more tokens than Nemotron and 71% more than Harrier. Encoding cost scales with tokens
at the same model size, so for an Indic or Persian corpus Qwen3-8B costs about 1.5–3× more to index than Nemotron-8B. Per-language
token counts are in `results/mmteb/miracl_tokens.txt` (`scripts/diag_tokens.py`).

### Recommendation

| Workload | Pick |
|---|---|
| Code retrieval | **Nemotron-8B** (`results.md`) |
| Multilingual RAG/search, major languages, same-language queries | **Nemotron-8B** |
| Cross-lingual search (query language ≠ document language), or low-resource scripts (Odia, Lao, Khmer, Sinhala, Amharic, Tibetan, …) | **Qwen3-8B** (Apache-2.0), with the tokenizer cost above |
| Mixed / unknown multilingual traffic and one model for everything, cost secondary | Harrier-27B (MIT). Best average and never far from the best on any multilingual task, at 3× the compute |

Nemotron's licence (OpenMDW-1.1) still needs checking against your use case.

### Origin, licence and a 24 GB card

| Model | Origin | Licence | On one 24 GB card |
|---|---|---|---|
| Nemotron-3-Embed-8B | NVIDIA, US | OpenMDW-1.1 (not an OSI licence) | BF16 weights 14.8 GiB; expected to fit with limited context and batch (not tested on a 24 GB card) |
| Qwen3-Embedding-8B | Alibaba, China | Apache-2.0 | about the same size as Nemotron in BF16 |
| Harrier-27B | Microsoft, US | MIT | only quantized: Intezra's 4-bit builds, 20.3 GB with 8k-token inputs or 18.5 GB with 16k-token inputs ([harrier-quant](https://github.com/Intezra-Inc/harrier-quant)) |

**If the model must be US-origin and run on a single 24 GB card, the choice is Nemotron-8B or quantized Harrier-27B:**
- **Quantized Harrier-27B** for cross-lingual search, low-resource scripts or mixed multilingual traffic. On Belebele cross-lingual pairs it scores
  0.871 against Nemotron's 0.820, and it falls below 0.6 on 9 of 122 Belebele languages against Nemotron's 24. It has the best overall mean
  (0.781 vs 0.760) and a permissive licence (MIT).
- **Nemotron-8B** for same-language search in major languages, where it wins MIRACL (0.731 vs 0.694) and MLQA, at about 3× the throughput.

The quantization costs Harrier little on these tasks: the recommended 20.3 GB build is within 0.004 nDCG@10 of BF16 on BelebeleRetrieval and
MLQARetrieval in the quantization evaluation (`results.md` in harrier-quant). The full 18-task multilingual suite above was run on BF16 weights
only, and the 24 GB fit was measured with vLLM's memory capped on a larger card, not on a physical 24 GB card.

## Agreement with the public leaderboard

Our scores reproduce the public `embeddings-benchmark/results` numbers closely. For Qwen3-8B and Harrier-27B, which both have complete public
runs, 30 of 36 task scores agree within ±0.01 (`results/mmteb/vs_public.json`). The exceptions:
- **LEMBPasskey, Harrier:** −0.137, from 8,192 truncation (see above).
- **Statcan:** Harrier +0.064, Qwen +0.030. It has long dialogue queries, so this is most likely a max-length difference.
- **Harrier, TempReasonL1:** +0.017.
- **Qwen, AILAStatutes −0.031 and MLQA −0.020:** the public Qwen run used an older revision (`4e42393`, vs `1d8ad4c` here).
- **Nemotron, MIRACL:** +0.014 vs NVIDIA's public result, which was run at max length 4,096. Nemotron has no other complete public MMTEB results.

## How it was run (deviations from a single clean run)

- **GPU throttling.** Blackwell GPUs 1–2, later also 0 and 3, overheated to 91–92 °C and dropped to 560–900 MHz at 115 W, versus 1.6 GHz at
  300 W on cool cards. After the first 15 hours, the remaining Qwen3-8B and Harrier-27B work was moved to the H100 box (identical venv, as in
  the code benchmark), split by MIRACL language with `scripts/run_subsets.py`. That script uses the identical load path and settings and evaluates one
  language subset at a time. Nemotron-8B ran entirely on Blackwell GPU 0.
- **Merging.** Per-subset results from all caches on both boxes are merged by `scripts/mmteb_report.py`, which also checks completeness
  against each task's eval splits × subsets. Merged JSONs are in `results/mmteb/merged/`, and the provenance of every task/subset is in
  `results/mmteb/scores.json`. If a subset appears in more than one cache, the first source in a fixed order is used and the choice is recorded. No task score mixes splits from different runs except across language subsets.
- **No throughput figures from this run.** Accuracy only: GPUs were mixed and throttled.
- **Cooling.** The Blackwell box needs a look at chassis airflow or fan curves. Under sustained 4-GPU load, every card throttled.

# Detailed results

| Task | Main score | Nemotron-8B | Qwen3-8B | Harrier-27B | Best |
|---|---|---|---|---|---|
| **Multilingual / non-English** | |  |  |  | |
| TwitterHjerneRetrieval | ndcg_at_10 | 0.828 | 0.789 | 0.909 | Harrier-27B |
| BelebeleRetrieval | ndcg_at_10 | 0.824 | 0.889 | 0.866 | Qwen3-8B |
| MLQARetrieval | ndcg_at_10 | 0.847 | 0.816 | 0.796 | Nemotron-8B |
| StatcanDialogueDatasetRetrieval | recall_at_10 | 0.499 | 0.435 | 0.615 | Harrier-27B |
| WikipediaRetrievalMultilingual | ndcg_at_10 | 0.937 | 0.915 | 0.924 | Nemotron-8B |
| CovidRetrieval | ndcg_at_10 | 0.835 | 0.875 | 0.832 | Qwen3-8B |
| MIRACLRetrievalHardNegatives | ndcg_at_10 | 0.731 | 0.697 | 0.694 | Nemotron-8B |
| **English-only** | |  |  |  | |
| StackOverflowQA | ndcg_at_10 | 0.977 | 0.947 | 0.969 | Nemotron-8B |
| AILAStatutes | ndcg_at_10 | 0.583 | 0.820 | 0.550 | Qwen3-8B |
| ArguAna | ndcg_at_10 | 0.632 | 0.765 | 0.682 | Qwen3-8B |
| HagridRetrieval | ndcg_at_10 | 0.989 | 0.988 | 0.989 | Harrier-27B |
| LegalBenchCorporateLobbying | ndcg_at_10 | 0.960 | 0.952 | 0.957 | Nemotron-8B |
| LEMBPasskeyRetrieval | ndcg_at_1 | 0.850 | 0.843 | 0.838 | Nemotron-8B |
| SCIDOCS | ndcg_at_10 | 0.255 | 0.327 | 0.291 | Qwen3-8B |
| SpartQA | ndcg_at_10 | 0.657 | 0.156 | 0.880 | Harrier-27B |
| TempReasonL1 | ndcg_at_10 | 0.526 | 0.013 | 0.436 | Nemotron-8B |
| TRECCOVID | ndcg_at_10 | 0.866 | 0.947 | 0.891 | Qwen3-8B |
| WinoGrande | ndcg_at_10 | 0.893 | 0.550 | 0.931 | Harrier-27B |
| **Mean, all 18** | | **0.760** | **0.707** | **0.781** | |
| **Mean, 7 multilingual / non-English** | | **0.786** | **0.774** | **0.805** | |
| **Mean, 11 English-only** | | **0.744** | **0.665** | **0.765** | |

MIRACL hard negatives, nDCG@10 per language:

| Lang | Nemotron-8B | Qwen3-8B | Harrier-27B | Best |
|---|---|---|---|---|
| ar | 0.814 | 0.791 | 0.796 | Nemotron-8B |
| bn | 0.840 | 0.779 | 0.782 | Nemotron-8B |
| de | 0.628 | 0.609 | 0.592 | Nemotron-8B |
| en | 0.616 | 0.593 | 0.578 | Nemotron-8B |
| es | 0.599 | 0.594 | 0.567 | Nemotron-8B |
| fa | 0.650 | 0.610 | 0.602 | Nemotron-8B |
| fi | 0.814 | 0.785 | 0.786 | Nemotron-8B |
| fr | 0.598 | 0.597 | 0.586 | Nemotron-8B |
| hi | 0.688 | 0.639 | 0.652 | Nemotron-8B |
| id | 0.585 | 0.558 | 0.552 | Nemotron-8B |
| ja | 0.782 | 0.745 | 0.725 | Nemotron-8B |
| ko | 0.737 | 0.679 | 0.671 | Nemotron-8B |
| ru | 0.764 | 0.735 | 0.709 | Nemotron-8B |
| sw | 0.802 | 0.701 | 0.762 | Nemotron-8B |
| te | 0.884 | 0.841 | 0.828 | Nemotron-8B |
| th | 0.847 | 0.812 | 0.811 | Nemotron-8B |
| yo | 0.814 | 0.825 | 0.882 | Harrier-27B |
| zh | 0.696 | 0.645 | 0.605 | Nemotron-8B |

Public-leaderboard comparison (ours / public): `results/mmteb/vs_public.json`. Raw merged results: `results/mmteb/merged/`.
