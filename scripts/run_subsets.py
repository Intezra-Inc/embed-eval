#!/usr/bin/env python
"""Helper for the MMTEB retrieval run: evaluate selected language subsets of one task with run_eval's exact load path and
settings (BF16, max len 8192, seed 42, per-model batch size). Used to move MIRACL-HN languages off thermally throttled GPUs;
writes raw mteb results to a separate cache (merged per subset by the report script).
Usage: CUDA_VISIBLE_DEVICES=0 python run_subsets.py --model qwen3-8b --task MIRACLRetrievalHardNegatives --subsets de en ...
"""
import argparse, os, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import st_compat  # noqa: F401,E402
import numpy as np, torch, mteb  # noqa: E402
import zembed_compat  # noqa: F401,E402
from run_eval import load_encoder, MODELS, ROOT, SEED  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=MODELS)
ap.add_argument("--task", required=True)
ap.add_argument("--subsets", nargs="+", required=True)
ap.add_argument("--cache", default="results/mteb_cache_split")
args = ap.parse_args()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
enc, info = load_encoder(args.model)
for s in args.subsets:  # one subset at a time so each finished language is saved immediately
    t0 = time.time()
    mteb.evaluate(enc, [mteb.get_task(args.task, hf_subsets=[s])], cache=mteb.ResultCache(cache_path=ROOT / args.cache / s),
                  encode_kwargs={"batch_size": MODELS[args.model][3]}, overwrite_strategy="only-missing", co2_tracker=False)
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] SUBSET DONE {args.model} {args.task} {s} {(time.time()-t0)/60:.1f} min", flush=True)
os._exit(0)
