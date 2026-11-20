"""Diagnostic: MIRACL-HN corpus size and mean tokens/doc per language per model tokenizer (500-doc sample)."""
import random, json
import mteb
from transformers import AutoTokenizer
toks = {k: AutoTokenizer.from_pretrained(p) for k, p in
        [("nemotron", "models/Nemotron-3-Embed-8B-BF16"), ("qwen", "models/Qwen3-Embedding-8B"), ("harrier", "models/harrier-oss-v1-27b")]}
t = mteb.get_task("MIRACLRetrievalHardNegatives"); t.load_data()
out = {}
for lang, splits in t.dataset.items():
    c = splits["dev"]["corpus"]
    texts = [((r.get("title") or "") + " " + r["text"]) for r in c]
    random.seed(0); s = random.sample(texts, min(500, len(texts)))
    out[lang] = dict(n=len(texts), **{k: sum(len(x) for x in tk(s)["input_ids"]) / len(s) for k, tk in toks.items()})
    print(lang, json.dumps(out[lang]), flush=True)
tot = {k: sum(v["n"] * v[k] for v in out.values()) / 1e6 for k in toks}
print("TOTAL Mtokens", json.dumps(tot))
