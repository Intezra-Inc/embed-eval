#!/usr/bin/env python
"""Code-retrieval benchmark harness: MTEB(Code, v1) + RTEB(Code, beta) public tasks, BF16, sentence-transformers.

Every model is loaded through its official mteb ModelMeta loader (documented prompts/instructions and
pooling as registered in mteb), with only these overrides, applied identically:
  - weights from the local download at a pinned revision (recorded in the result metadata)
  - max sequence length 8192 for every model (Nemotron: mteb registers 4096; its card documents 32768)
  - BF16 weights; fixed seeds (42)
Raw mteb result JSONs: results/mteb_cache/results/<org__name>/<revision>/<Task>.json

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python run_eval.py --model harrier-27b [--tasks HumanEvalRetrieval ...]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import st_compat  # noqa: F401,E402  transformers maps gemma3_text to the multimodal processor; fall back to tokenizer

import mteb  # noqa: E402
import zembed_compat  # noqa: E402  ST 6.x never calls zembed's custom tokenize(); see zembed_compat.py

ROOT = Path(os.environ.get("EE_ROOT", Path.home() / "embed-eval"))
MAX_LEN = 8192
SEED = 42
MODELS = {  # key: (mteb registry name, local dir, pinned HF revision, encode batch size)
    "harrier-27b": ("microsoft/harrier-oss-v1-27b", "models/harrier-oss-v1-27b", "0c0fc62f6d8af9e8604cb818c412301b103a0093", 16),
    "nemotron-8b": ("nvidia/Nemotron-3-Embed-8B-BF16", "models/Nemotron-3-Embed-8B-BF16", "d1f2f25730bbd775b99b29185134bc86653bf2d1", 32),
    "zembed-1": ("zeroentropy/zembed-1", "models/zembed-1-embedding", "cf13c81f3274394053d166740294f7eea4586f7a", 32),
    "harrier-0.6b": ("microsoft/harrier-oss-v1-0.6b", "models/harrier-oss-v1-0.6b", "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee", 32),
    # added on request after the main four (run on the H100 box with an identical venv; weights identical to mteb's registered revisions)
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "models/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", 32),
    "qwen3-8b": ("Qwen/Qwen3-Embedding-8B", "models/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", 8),
}


def code_tasks():
    code = [t.metadata.name for t in mteb.get_benchmark("MTEB(Code, v1)").tasks]
    rteb_all = mteb.get_benchmark("RTEB(Code, beta)").tasks
    rteb_public = [t.metadata.name for t in rteb_all if t.metadata.is_public]
    rteb_private = [t.metadata.name for t in rteb_all if not t.metadata.is_public]
    return code, rteb_public, rteb_private


def load_encoder(key):
    name, local, rev, _ = MODELS[key]
    meta = mteb.get_model_meta(name)
    kw = dict(meta.loader_kwargs or {})
    path = str(ROOT / local)
    if meta.loader.__name__ == "InstructSentenceTransformerModel":
        enc = meta.loader(path, None, device="cuda", max_seq_length=MAX_LEN, model_kwargs={"dtype": torch.bfloat16}, **kw)
    else:
        if "processor_kwargs" in kw:  # Nemotron: mteb registers model_max_length=4096
            kw["processor_kwargs"] = {**kw["processor_kwargs"], "model_max_length": MAX_LEN}
        mk = {k: v for k, v in kw.pop("model_kwargs", {}).items() if k != "torch_dtype"}
        enc = meta.loader(path, None, device="cuda", model_kwargs={**mk, "dtype": torch.bfloat16}, **kw)
    st = enc.model
    zembed_patched = zembed_compat.patch(st)  # no-op for other models
    st.max_seq_length = MAX_LEN
    if next(st.parameters()).dtype != torch.bfloat16:
        st.to(torch.bfloat16)
    enc.mteb_model_meta = meta.model_copy(update={"revision": rev})
    info = dict(model=key, registry_name=name, local_path=path, revision=rev, loader=meta.loader.__name__,
                max_seq_length=st.max_seq_length, dtype=str(next(st.parameters()).dtype), zembed_compat_shim=zembed_patched,
                st_prompts=getattr(st, "prompts", None), modules=[type(m).__name__ for m in st],
                pooling=str(st[1]) if len(st) > 1 else None, embed_dim=st.get_embedding_dimension(),
                n_params=sum(p.numel() for p in st.parameters()),
                instruction_template="Instruct: {task}\\nQuery: (queries only; mteb Harrier loader)" if "instruction_template" in kw else None,
                loader_kwargs_overridden={k: (v if not isinstance(v, dict) or len(v) < 5 else f"dict[{len(v)}]") for k, v in kw.items() if k != "instruction_template"})
    return enc, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--tasks", nargs="*", default=None, help="default: MTEB(Code, v1) + RTEB(Code, beta) public")
    ap.add_argument("--label", default=None, help="summary file label (default: model key)")
    ap.add_argument("--batch-size", type=int, default=None, help="encode batch size (throughput only; default per model)")
    ap.add_argument("--instructions", default=None,
                    help="side experiment: JSON {task: instruction} applied as task-specific query instructions to "
                         "instruction-tuned models (Harrier/Qwen wrappers); results go to a separate cache")
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    code, rteb_public, rteb_private = code_tasks()
    names = args.tasks or list(dict.fromkeys(code + rteb_public))
    t0 = time.time()
    enc, info = load_encoder(args.model)
    print(json.dumps(info, default=str, indent=1), flush=True)
    out = ROOT / "results" / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label or args.model}_model_info.json").write_text(json.dumps(info, default=str, indent=1))

    cache_dir = ROOT / "results" / "mteb_cache"
    if args.instructions:
        if not hasattr(enc, "prompts_dict"):
            raise SystemExit(f"{args.model} has no per-task instruction support (fixed query/document prefixes)")
        instr = json.loads(Path(args.instructions).read_text())
        enc.prompts_dict = {**(enc.prompts_dict or {}), **{n: instr[n] for n in names if n in instr}}
        info["task_instructions"] = {n: instr.get(n) for n in names}
        (out / f"{args.label or args.model}_model_info.json").write_text(json.dumps(info, default=str, indent=1))
        cache_dir = ROOT / "results" / "mteb_cache_taskinstr"
        print("task-specific instructions:", json.dumps(info["task_instructions"], indent=1), flush=True)
    tasks = [mteb.get_task(n) for n in names]
    res = mteb.evaluate(enc, tasks, cache=mteb.ResultCache(cache_path=cache_dir),
                        encode_kwargs={"batch_size": args.batch_size or MODELS[args.model][3]}, overwrite_strategy="only-missing",
                        co2_tracker=False)
    rows = {}
    for tr in res.task_results:
        s = [x for split in tr.scores.values() for x in split]
        rows[tr.task_name] = dict(ndcg_at_10=float(np.mean([x["ndcg_at_10"] for x in s])),
                                  recall_at_10=float(np.mean([x["recall_at_10"] for x in s])), n_subsets=len(s),
                                  eval_time_s=tr.evaluation_time)
    summary = dict(model=args.model, revision=info["revision"], batch_size=args.batch_size or MODELS[args.model][3], minutes=round((time.time() - t0) / 60, 1),
                   tasks=rows, mteb_code=code, rteb_public=rteb_public, rteb_private_skipped=rteb_private)
    (out / f"{args.label or args.model}.json").write_text(json.dumps(summary, indent=1))
    for n, r in rows.items():
        print(f"{n:30s} nDCG@10={r['ndcg_at_10']:.4f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
