#!/usr/bin/env python
"""Build the results tables for results.md from raw mteb result JSONs and cost/*.json.

Reads results/mteb_cache/results/<org__name>/<revision>/<Task>.json directly (main score = nDCG@10, averaged
over a task's subsets/splits, as mteb does). Writes results/tables.md and results/report.json.
The ranking/recommendation prose in results.md is written by hand on top of these tables.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EE_ROOT", Path.home() / "embed-eval"))
# Tie-dominated tasks: most relevant docs have an identical-text NON-relevant twin in the corpus (results/tie_affected_tasks.json),
# so nDCG@10 there depends on the search implementation's tie-breaking, not the model.
TIE_TASKS = {"CodeTransOceanDL", "SyntheticText2SQL", "DS1000Retrieval", "CosQA"}
STAR = " \\*"  # markdown-escaped asterisk (kept outside f-strings for Python < 3.12)
MODELS = [  # key, registry name, cache dir name, revision, display
    ("harrier-27b", "microsoft/harrier-oss-v1-27b", "microsoft__harrier-oss-v1-27b", "0c0fc62f6d8af9e8604cb818c412301b103a0093", "Harrier-27B"),
    ("nemotron-8b", "nvidia/Nemotron-3-Embed-8B-BF16", "nvidia__Nemotron-3-Embed-8B-BF16", "d1f2f25730bbd775b99b29185134bc86653bf2d1", "Nemotron-3-Embed-8B"),
    ("zembed-1", "zeroentropy/zembed-1", "zeroentropy__zembed-1", "cf13c81f3274394053d166740294f7eea4586f7a", "zembed-1 (4B)"),
    ("harrier-0.6b", "microsoft/harrier-oss-v1-0.6b", "microsoft__harrier-oss-v1-0.6b", "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee", "Harrier-0.6B"),
    ("qwen3-8b", "Qwen/Qwen3-Embedding-8B", "Qwen__Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "Qwen3-Embedding-8B"),
    ("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", "Qwen__Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "Qwen3-Embedding-0.6B"),
]


def task_lists():
    import mteb
    code = [t.metadata.name for t in mteb.get_benchmark("MTEB(Code, v1)").tasks]
    rteb = mteb.get_benchmark("RTEB(Code, beta)").tasks
    return code, [t.metadata.name for t in rteb if t.metadata.is_public], [t.metadata.name for t in rteb if not t.metadata.is_public]


def training_overlap():
    import mteb
    out = {}
    for key, name, *_ in MODELS:
        td = mteb.get_model_meta(name).training_datasets
        out[key] = None if td is None else set(td)
    return out


def score(cache_dir, rev, task):
    p = ROOT / "results" / "mteb_cache" / "results" / cache_dir / rev / f"{task}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    s = [x["ndcg_at_10"] for split in d["scores"].values() for x in split]
    return float(np.mean(s)) if s else None


def ranks(vals):
    order = sorted([k for k in vals if vals[k] is not None], key=lambda k: -vals[k])
    return order


def table(tasks, data, overlap, disp, title):
    L = [f"### {title}", "", "| Task | " + " | ".join(disp[k] for k, *_ in MODELS) + " | Top model |",
         "|---|" + "---|" * (len(MODELS) + 1)]
    for t in tasks:
        vals = {k: data[k].get(t) for k, *_ in MODELS}
        best = max((v for v in vals.values() if v is not None), default=None)
        cells = []
        for k, *_ in MODELS:
            v = vals[k]
            mark = "" if overlap[k] is None else ("†" if t in overlap[k] else "")
            if overlap[k] is None:
                mark = "?"
            cells.append("–" if v is None else (f"**{v:.4f}**" if v == best else f"{v:.4f}") + mark)
        top = ranks(vals)
        L.append(f"| {t}{STAR if t in TIE_TASKS else ''} | " + " | ".join(cells) + f" | {disp[top[0]] if top else '–'}{' (tie-dominated)' if t in TIE_TASKS else ''} |")
    means = {k: (np.mean([data[k][t] for t in tasks]) if all(data[k].get(t) is not None for t in tasks) else None) for k, *_ in MODELS}
    bm = max((v for v in means.values() if v is not None), default=None)
    L.append("| **Mean** | " + " | ".join("–" if means[k] is None else (f"**{means[k]:.4f}**" if means[k] == bm else f"{means[k]:.4f}") for k, *_ in MODELS) + " | |")
    rob = [t for t in tasks if t not in TIE_TASKS]
    if len(rob) < len(tasks):
        mr = {k: (np.mean([data[k][t] for t in rob]) if all(data[k].get(t) is not None for t in rob) else None) for k, *_ in MODELS}
        br = max((v for v in mr.values() if v is not None), default=None)
        L.append(f"| **Mean, {len(rob)} tie-robust tasks** | " + " | ".join("–" if mr[k] is None else (f"**{mr[k]:.4f}**" if mr[k] == br else f"{mr[k]:.4f}") for k, *_ in MODELS) + " | |")
    return L, means


def main():
    code, rteb_pub, rteb_priv = task_lists()
    overlap = training_overlap()
    disp = {k: d for k, _, _, _, d in MODELS}
    alltasks = list(dict.fromkeys(code + rteb_pub))
    data = {k: {t: score(c, r, t) for t in alltasks} for k, _, c, r, _ in MODELS}
    zero_shot = [t for t in alltasks if all(overlap[k] is None or t not in overlap[k] for k, *_ in MODELS)]

    L = []
    l1, m_code = table(code, data, overlap, disp, "a) MTEB(Code, v1): nDCG@10")
    l2, m_rteb = table(rteb_pub, data, overlap, disp, "b) RTEB(Code, beta), public datasets: nDCG@10")
    l3, m_zs = table(zero_shot, data, overlap, disp, "Zero-shot subset: tasks no model declares as training data")
    L += l1 + ["", "† = listed in the model's declared training data (mteb ModelMeta); ? = model declares no training data (overlap unknown).", ""]
    L += ["\\* Tie-dominated task (see notes): per-task nDCG@10 and rankings there reflect tie-breaking, not model quality.", ""]
    L += l2 + ["", f"Skipped (private RTEB held-out sets, cannot be run locally): {', '.join(rteb_priv)}.",
               "AppsRetrieval appears in both MTEB(Code, v1) and RTEB(Code); it was run once and appears in both tables.", ""]
    L += l3 + [""]

    # ranking flips
    m_all = {k: (np.mean([data[k][t] for t in alltasks]) if all(data[k].get(t) is not None for t in alltasks) else None) for k, *_ in MODELS}
    robust_all = [t for t in alltasks if t not in TIE_TASKS]
    m_rob = {k: (np.mean([data[k][t] for t in robust_all]) if all(data[k].get(t) is not None for t in robust_all) else None) for k, *_ in MODELS}
    overall = ranks(m_rob)
    flips = []
    for t in robust_all:
        r = ranks({k: data[k].get(t) for k, *_ in MODELS})
        if r and overall and r != overall:
            flips.append((t, r, r[0] != overall[0]))
    L += [f"### Ranking flips vs overall (overall = mean over the {len(robust_all)} tie-robust tasks; tie-dominated tasks excluded from flips)", "",
          ("Overall order: " + " > ".join(f"{disp[k]} ({m_rob[k]:.4f})" for k in overall)) if overall else "Overall order: (pending)", "",
          ("Mean over all 18 (incl. tie-dominated): " + ", ".join(f"{disp[k]} {m_all[k]:.4f}" for k in ranks(m_all))) if ranks(m_all) else "", "",
          "| Task | Order on this task | Top-1 differs from overall |", "|---|---|---|"]
    for t, r, top in flips:
        L.append(f"| {t} | " + " > ".join(f"{disp[k]} {data[k][t]:.3f}" for k in r) + f" | {'**yes**' if top else 'no'} |")
    L += [""]

    # cost
    L += ["### c) Cost (RTX PRO 6000 Blackwell Max-Q, BF16, 5,000 code documents, max 8,192 tokens, same GPU for every model)", "",
          "Batch 32 for every model except where it ran out of memory (then 16, then 8). Peak memory is dominated by the activations of "
          "the longest batch (p95 input = 8,192 tokens), not the weights; smaller batches or shorter inputs need far less.", "",
          "| Model | Params | Weights (BF16) | Embedding dim | Batch | Peak GPU memory (torch reserved / nvidia-smi) | Docs/sec | Tokens/sec | Storage per 1M vectors, float32 | int8 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    costs = {}
    for k, *_ in MODELS:
        p = ROOT / "cost" / f"{k}.json"
        if p.exists():
            c = json.loads(p.read_text()); costs[k] = c
            L.append(f"| {disp[k]} | {c['n_params'] / 1e9:.2f}B | {c['n_params'] * 2 / 2**30:.1f} GiB | {c['embed_dim']} | {c['batch_size']} | "
                     f"{c['peak_torch_reserved_gib']:.1f} / {c['peak_nvidia_smi_gib']:.1f} GiB | {c['docs_per_s']:.1f} | {c['tokens_per_s']:,} | "
                     f"{c['storage_per_1M_float32_gb']:.1f} GB | {c['storage_per_1M_int8_gb']:.2f} GB |")
        else:
            L.append(f"| {disp[k]} | – | – | – | – | – | – | – | – | – |")
    L += [""]
    (ROOT / "results" / "tables.md").write_text("\n".join(L) + "\n")
    (ROOT / "results" / "report.json").write_text(json.dumps(dict(scores=data, mean_code=m_code, mean_rteb_public=m_rteb, mean_tie_robust=m_rob, tie_tasks=sorted(TIE_TASKS),
                                                                  mean_zero_shot=m_zs, mean_all=m_all, zero_shot_tasks=zero_shot,
                                                                  rteb_private_skipped=rteb_priv, costs=costs,
                                                                  overlap={k: (None if v is None else sorted(set(v) & set(alltasks))) for k, v in overlap.items()}),
                                                             indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
