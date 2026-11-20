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

