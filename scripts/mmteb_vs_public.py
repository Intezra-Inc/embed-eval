"""Cross-check: our MMTEB retrieval scores vs the public embeddings-benchmark/results repo (same tasks, same scoring)."""
import json, logging, warnings
import mteb
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
ours = json.load(open("results/mmteb/scores.json"))["scores"]
tasks = [t for t in mteb.get_benchmark("MTEB(Multilingual, v2)").tasks if t.metadata.type == "Retrieval"]
M = {"Nemotron-8B": "nvidia/Nemotron-3-Embed-8B-BF16", "Qwen3-8B": "Qwen/Qwen3-Embedding-8B", "Harrier-27B": "microsoft/harrier-oss-v1-27b"}
res = mteb.ResultCache().load_results(models=list(M.values()), tasks=tasks, download_latest=True)
pub, revs = {}, {}
for mr in res.model_results:
    revs[mr.model_name] = mr.model_revision
    for tr in mr.task_results:
        t = mteb.get_task(tr.task_name)
        try:
            want = {(sp, s) for sp in t.metadata.eval_splits for s in t.hf_subsets}
            got = {(sp, x["hf_subset"]) for sp, l in tr.scores.items() for x in l}
            if want - got:
                continue  # partial public result: not comparable
            pub[(mr.model_name, tr.task_name)] = tr.get_score(splits=[s for s in t.metadata.eval_splits if s in tr.scores])
        except Exception:
            pass
out = {"public_revisions": revs, "rows": {}}
print("task".ljust(32) + "".join(k.rjust(26) for k in M))
for t in tasks:
    n = t.metadata.name; row = n.ljust(32)
    for k, v in M.items():
        p, o = pub.get((v, n)), ours[k][n]
        out["rows"].setdefault(n, {})[k] = dict(ours=o, public=p)
        row += (f"{o:.3f} / {p:.3f} ({o - p:+.3f})" if p is not None else f"{o:.3f} / —").rjust(26)
    print(row)
json.dump(out, open("results/mmteb/vs_public.json", "w"), indent=1)
print("public revisions:", revs)
