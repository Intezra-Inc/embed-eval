"""Diagnostic: reproduce run_eval's MIRACL-HN evaluation for one subset with stack dumps every 5 min (temp cache)."""
import faulthandler, os, sys, time
faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
sys.path.insert(0, os.path.dirname(__file__))
import st_compat  # noqa
import mteb
from run_eval import load_encoder, MODELS
key, lang = sys.argv[1], sys.argv[2]
enc, info = load_encoder(key)
t = mteb.get_task("MIRACLRetrievalHardNegatives", hf_subsets=[lang])
t0 = time.time()
mteb.evaluate(enc, [t], cache=mteb.ResultCache(cache_path=f"/tmp/diag_cache_{key}"), encode_kwargs={"batch_size": MODELS[key][3]},
              overwrite_strategy="always", co2_tracker=False)
print("DONE", lang, round((time.time() - t0) / 3600, 2), "h", flush=True)
os._exit(0)
