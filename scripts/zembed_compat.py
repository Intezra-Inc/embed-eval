"""Compatibility shim: zembed-1's custom module on sentence-transformers 6.x.

zembed-1 ships `modeling_zembed.ZembedTransformer`, which overrides `Transformer.tokenize()` to append
"<|im_end|>\\n" to every input before tokenizing (the end-of-turn token its last-token pooling relies on).
sentence-transformers 6.x routes encoding through `preprocess()` and never calls `tokenize()`, so the
suffix is silently dropped and pooling lands on the wrong token (HumanEvalRetrieval nDCG@10 0.64).

This applies the same behaviour through `preprocess()`: append the suffix, then tokenize with the parent
implementation (right truncation at max_seq_length, as the original code does, so over-length inputs lose
the suffix exactly as they would with the model's own code). Validated against the model card's example.
"""
from sentence_transformers.models import Transformer

SUFFIX = "<|im_end|>\n"


def patch(st_model):
    mod = st_model[0]
    if type(mod).__name__ != "ZembedTransformer":
        return False
    cls = type(mod)
    if getattr(cls, "_zembed_compat", False):
        return True

    def preprocess(self, inputs, *args, **kwargs):
        inputs = [x + SUFFIX if isinstance(x, str) else x for x in inputs]
        return Transformer.preprocess(self, inputs, *args, **kwargs)

    cls.preprocess = preprocess
    cls._zembed_compat = True
    return True
