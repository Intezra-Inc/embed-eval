#!/usr/bin/env python
"""MMTEB(Multilingual, v2) retrieval subset report: merge per-subset results from every cache (Blackwell main cache, H100 cache,
per-language split caches on both boxes), check completeness, score with mteb's TaskResult.get_score (official main score,
mean over eval splits x subsets), write merged JSONs + results/mmteb/tables.md + results/mmteb/scores.json.
Usage (Blackwell box, after copying H100 caches to results/h100_*): python mmteb_report.py
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
import mteb
from mteb.results.task_result import TaskResult

ROOT = Path(os.environ.get("EE_ROOT", Path.home() / "embed-eval"))
MODELS = {"Nemotron-8B": "nvidia__Nemotron-3-Embed-8B-BF16", "Qwen3-8B": "Qwen__Qwen3-Embedding-8B",
          "Harrier-27B": "microsoft__harrier-oss-v1-27b"}
SOURCES = ["results/mteb_cache", "results/mteb_cache_split/*", "results/h100_mteb_cache", "results/h100_mteb_cache_split/*"]
TASKS = [t for t in mteb.get_benchmark("MTEB(Multilingual, v2)").tasks if t.metadata.type == "Retrieval"]
ENGLISH_ONLY = {t.metadata.name for t in TASKS if set(l.split("-")[0] for l in
                (t.metadata.eval_langs if isinstance(t.metadata.eval_langs, list) else sum(t.metadata.eval_langs.values(), []))) == {"eng"}}
OUT = ROOT / "results" / "mmteb"


def merged(model_dir, task):
    name = task.metadata.name
    files = [f for s in SOURCES for f in sorted(glob.glob(str(ROOT / s / "results" / model_dir / "*" / f"{name}.json")))]
    res, srcs = None, []
    for f in files:
        tr = TaskResult.from_disk(Path(f))
        have = {(sp, x["hf_subset"]) for sp, l in (res.scores.items() if res else []) for x in l}
        new = {sp: [x for x in l if (sp, x["hf_subset"]) not in have] for sp, l in tr.scores.items()}
        if not any(new.values()):
            continue
        srcs.append(str(Path(f).relative_to(ROOT)))
        if res is None:
            res = tr
        else:
            for sp, l in new.items():
                res.scores.setdefault(sp, []).extend(l)
    return res, srcs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scores, prov, missing, langs = {}, {}, {}, {}
    for label, mdir in MODELS.items():
        for task in TASKS:
            name = task.metadata.name
            tr, srcs = merged(mdir, task)
            prov[f"{label}/{name}"] = srcs
            if tr is None:
                missing[f"{label}/{name}"] = "no result"; continue
            want = {(sp, s) for sp in task.metadata.eval_splits for s in task.hf_subsets}
            got = {(sp, x["hf_subset"]) for sp, l in tr.scores.items() for x in l}
            if want - got:
                missing[f"{label}/{name}"] = sorted(want - got)
            d = OUT / "merged" / mdir / f"{name}.json"; d.parent.mkdir(parents=True, exist_ok=True)
            tr.to_disk(d)
            scores.setdefault(label, {})[name] = float(tr.get_score(splits=[s for s in task.metadata.eval_splits if s in tr.scores]))
            if name == "MIRACLRetrievalHardNegatives":
                langs[label] = {x["hf_subset"]: x["main_score"] for x in tr.scores["dev"]}
    json.dump(dict(scores=scores, miracl_per_language=langs, missing=missing, provenance=prov), open(OUT / "scores.json", "w"), indent=1)
    if missing:
        print("INCOMPLETE:", json.dumps(missing, indent=1)[:3000]); 
    labels = list(MODELS)
    names = [t.metadata.name for t in TASKS]
    multi = [n for n in names if n not in ENGLISH_ONLY]
    eng = [n for n in names if n in ENGLISH_ONLY]
    ms = {t.metadata.name: t.metadata.main_score for t in TASKS}
    L = ["| Task | Main score | " + " | ".join(labels) + " | Best |", "|---|---|" + "---|" * (len(labels) + 1)]
    for group, ns in (("Multilingual / non-English", multi), ("English-only", eng)):
        L.append(f"| **{group}** | | " + " | ".join("" for _ in labels) + " | |")
        for n in ns:
            v = [scores.get(l, {}).get(n) for l in labels]
            best = labels[int(np.nanargmax([x if x is not None else np.nan for x in v]))] if any(x is not None for x in v) else ""
            L.append(f"| {n} | {ms[n]} | " + " | ".join(f"{x:.3f}" if x is not None else "—" for x in v) + f" | {best} |")
    def mean(l, ns):
        v = [scores.get(l, {}).get(n) for n in ns]
        return f"{np.mean(v):.3f}" if all(x is not None for x in v) else "incomplete"
    for title, ns in ((f"Mean, all {len(names)}", names), (f"Mean, {len(multi)} multilingual / non-English", multi), (f"Mean, {len(eng)} English-only", eng)):
        L.append(f"| **{title}** | | " + " | ".join(f"**{mean(l, ns)}**" for l in labels) + " | |")
    if langs:
        ll = sorted(next(iter(langs.values())))
        L += ["", "MIRACL hard negatives, nDCG@10 per language:", "", "| Lang | " + " | ".join(labels) + " | Best |", "|---|" + "---|" * (len(labels) + 1)]
        for lg in ll:
            v = [langs.get(l, {}).get(lg) for l in labels]
            best = labels[int(np.nanargmax([x if x is not None else np.nan for x in v]))] if any(x is not None for x in v) else ""
            L.append(f"| {lg} | " + " | ".join(f"{x:.3f}" if x is not None else "—" for x in v) + f" | {best} |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
