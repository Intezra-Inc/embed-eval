"""Diagnostic: encode speed on a MIRACL-HN subset corpus sample, per model (same load path as run_eval)."""
import os, sys, time, json, random
sys.path.insert(0, os.path.dirname(__file__))
import st_compat  # noqa
import torch, mteb
from run_eval import load_encoder
key, lang, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = mteb.get_task("MIRACLRetrievalHardNegatives", hf_subsets=[lang]); t.load_data()
ds = t.dataset[lang]["dev"]
corpus = [(r.get("title") or "") + " " + r["text"] for r in ds["corpus"]]
print("corpus size", len(corpus), "queries", len(ds["queries"]), flush=True)
random.seed(0); docs = random.sample(corpus, n)
enc, info = load_encoder(key); st = enc.model
lens = [len(x) for x in st.tokenizer(docs)["input_ids"]]
print("tokens mean", sum(lens)/len(lens), "max", max(lens), flush=True)
st.encode(docs[:16], batch_size=8); torch.cuda.synchronize()
t0 = time.time(); st.encode(docs, batch_size=8); torch.cuda.synchronize()
dt = time.time() - t0
print(json.dumps(dict(model=key, lang=lang, n=n, docs_per_s=round(n/dt, 1), est_corpus_hours=round(len(corpus)/(n/dt)/3600, 2))), flush=True)
os._exit(0)
