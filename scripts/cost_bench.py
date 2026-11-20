#!/usr/bin/env python
"""Encoding cost per model: throughput, peak GPU memory, embedding size, index storage.

  build : sample 5,000 documents (seed 42) from three code corpora used in the benchmark
          (COIRCodeSearchNetRetrieval 50%, StackOverflowQA 25%, SWEbenchCodeRetrieval 25%: short
          functions, Q&A posts, long source files) -> cost/docs.jsonl
  run   : load the model through run_eval.load_encoder (same path/settings as the benchmark: BF16,
          max_seq_length 8192, documented document prompt), warm up, then encode all 5,000 documents
          at batch size 32 and record docs/s, tokens/s, peak memory -> cost/<model>.json

Peak memory = torch.cuda.max_memory_reserved (PyTorch allocator) and the maximum process memory seen by
an nvidia-smi sampler every 0.2 s (includes CUDA context and non-PyTorch allocations).
Usage: python cost_bench.py build ; CUDA_VISIBLE_DEVICES=0 python cost_bench.py run --model nemotron-8b
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import st_compat  # noqa: F401,E402

ROOT = Path(os.environ.get("EE_ROOT", Path.home() / "embed-eval"))
OUT = ROOT / "cost"
SEED = 42
N = 5000
BATCH = 32  # default; --batch-size overrides (used only as an OOM fallback, recorded in the output)
SOURCES = [("COIRCodeSearchNetRetrieval", 0.50), ("StackOverflowQA", 0.25), ("SWEbenchCodeRetrieval", 0.25)]


def corpus_texts(task_name):
    import mteb
    t = mteb.get_task(task_name)
    t.load_data()
    texts = []
    if t.dataset is not None:
        for splits in t.dataset.values():
            for d in splits.values():
                for r in d["corpus"]:
                    title = (r.get("title") or "").strip()
                    texts.append(f"{title} {r['text']}".strip() if title else r["text"])
    else:  # legacy layout, nested {subset: {split: {doc_id: {title, text}}}} (depth varies by task)
        def walk(node):
            if isinstance(node, dict) and "text" in node and not isinstance(node["text"], dict):
                title = (node.get("title") or "").strip()
                texts.append(f"{title} {node['text']}".strip() if title else node["text"])
            elif isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, str):
                texts.append(node)
        walk(t.corpus)
    return texts


def build(_):
    rng = random.Random(SEED)
    docs = []
    for name, frac in SOURCES:
        pool = sorted(set(corpus_texts(name)))
        k = int(N * frac)
        docs += [dict(source=name, text=x) for x in rng.sample(pool, k)]
    rng.shuffle(docs)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print("wrote", len(docs), "docs")


class SmiSampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.stop = str(pid), 0, False
        self.dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    def run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(["nvidia-smi", "-i", self.dev, "--query-compute-apps=pid,used_memory",
                                               "--format=csv,noheader,nounits"], text=True)
                for line in out.strip().splitlines():
                    p, m = [x.strip() for x in line.split(",")]
                    if p == self.pid:
                        self.peak = max(self.peak, int(m))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)


def run(args):
    import numpy as np
    import torch
    from run_eval import load_encoder
    docs = [json.loads(l)["text"] for l in open(OUT / "docs.jsonl")]
    enc, info = load_encoder(args.model)
    st = enc.model
    tok_lens = [min(len(x), st.max_seq_length) for x in st.tokenizer(docs, add_special_tokens=True)["input_ids"]]
    smi = SmiSampler(os.getpid()); smi.start()
    st.encode_document(docs[:args.batch_size * 2], batch_size=args.batch_size)  # warm-up (kernels, allocator)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    emb = st.encode_document(docs, batch_size=args.batch_size, convert_to_numpy=True)
    torch.cuda.synchronize()
    dt = time.time() - t
    time.sleep(0.5); smi.stop = True
    dim = int(emb.shape[1])
    res = dict(model=args.model, gpu=torch.cuda.get_device_name(0), n_docs=len(docs), batch_size=args.batch_size,
               max_seq_length=st.max_seq_length, dtype=info["dtype"], n_params=info["n_params"], embed_dim=dim,
               seconds=round(dt, 2), docs_per_s=round(len(docs) / dt, 1), tokens_per_s=round(sum(tok_lens) / dt),
               mean_tokens=round(float(np.mean(tok_lens)), 1), p95_tokens=int(np.percentile(tok_lens, 95)),
               peak_torch_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
               peak_torch_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               peak_nvidia_smi_gib=round(smi.peak / 1024, 2),
               storage_per_1M_float32_gb=round(dim * 4 * 1e6 / 1e9, 2), storage_per_1M_int8_gb=round(dim * 1e6 / 1e9, 2))
    (OUT / f"{args.model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run"); r.add_argument("--model", required=True)
    r.add_argument("--batch-size", type=int, default=BATCH)
    args = ap.parse_args()
    {"build": build, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
