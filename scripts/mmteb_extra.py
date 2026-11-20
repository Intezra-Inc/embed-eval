"""Extra breakdowns for the MMTEB retrieval report (reads results/mmteb/scores.json + merged Belebele results)."""
import json, glob
import numpy as np
S = json.load(open("results/mmteb/scores.json"))["scores"]
L = list(S)
groups = {
 "all 18": list(S[L[0]]),
 "17 (excl. LEMBPasskey, truncated at 8192)": [t for t in S[L[0]] if t != "LEMBPasskeyRetrieval"],
 "14 standard retrieval (excl. Passkey + RAR-b reasoning SpartQA/TempReasonL1/WinoGrande)": [t for t in S[L[0]] if t not in ("LEMBPasskeyRetrieval", "SpartQA", "TempReasonL1", "WinoGrande")],
 "6 multilingual excl. Statcan (recall@10, en/fr dialogue)": ["TwitterHjerneRetrieval", "BelebeleRetrieval", "MLQARetrieval", "WikipediaRetrievalMultilingual", "CovidRetrieval", "MIRACLRetrievalHardNegatives"],
}
out = {}
for g, ts in groups.items():
    out[g] = {l: round(float(np.mean([S[l][t] for t in ts])), 3) for l in L}
    print(f"{g:90s}", "  ".join(f"{l} {v:.3f}" for l, v in out[g].items()))
# Belebele per language pair: where does Nemotron fall behind?
B = {}
for l, d in [("Nemotron-8B", "nvidia__Nemotron-3-Embed-8B-BF16"), ("Qwen3-8B", "Qwen__Qwen3-Embedding-8B"), ("Harrier-27B", "microsoft__harrier-oss-v1-27b")]:
    r = json.load(open(glob.glob(f"results/mmteb/merged/{d}/BelebeleRetrieval.json")[0]))
    B[l] = {x["hf_subset"]: x["main_score"] for x in r["scores"]["test"]}
subs = sorted(B["Qwen3-8B"])
mono = [s for s in subs if s.split("-")[0] == s.split("-")[1]]
cross = [s for s in subs if s not in mono]
for name, ss in (("monolingual pairs", mono), ("cross-lingual pairs", cross)):
    print(f"Belebele {name} ({len(ss)}):", "  ".join(f"{l} {np.mean([B[l][s] for s in ss]):.3f}" for l in B))
gap = sorted(((B["Nemotron-8B"][s] - B["Qwen3-8B"][s]), s) for s in mono)
print("Belebele monolingual, Nemotron minus Qwen, worst 12:", [(s.split("-")[0], round(d, 2)) for d, s in gap[:12]])
print("Belebele monolingual, Nemotron minus Qwen, best 6:", [(s.split("-")[0], round(d, 2)) for d, s in gap[-6:]])
print("Belebele monolingual languages where Nemotron < 0.6:", sum(B["Nemotron-8B"][s] < 0.6 for s in mono), "Qwen:", sum(B["Qwen3-8B"][s] < 0.6 for s in mono), "Harrier:", sum(B["Harrier-27B"][s] < 0.6 for s in mono))
json.dump(dict(groups=out), open("results/mmteb/extra.json", "w"), indent=1)
