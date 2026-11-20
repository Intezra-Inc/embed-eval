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
