#!/usr/bin/env python
"""Reproduce the zembed-1 model card example (expected similarities 0.7525, 0.5670, 0.6835) with and without the shim."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from sentence_transformers import SentenceTransformer
import zembed_compat

m = SentenceTransformer(os.path.expanduser("~/embed-eval/models/zembed-1-embedding"), trust_remote_code=True,
                        device="cuda", model_kwargs={"dtype": torch.bfloat16})
q = "What is backpropagation?"
docs = ["Backpropagation is a fundamental algorithm for training neural networks by computing gradients.",
        "Gradient descent is used to optimize model parameters during the training process.",
        "Neural network training relies on efficient computation of derivatives through backpropagation."]
def sims():
    return m.similarity(m.encode_query(q), m.encode_document(docs)).float().numpy().round(4).tolist()
print("card expects   [[0.7525, 0.567, 0.6835]]")
print("without shim  ", sims())
zembed_compat.patch(m)
print("with shim     ", sims())
ids = m.preprocess([m.prompts["document"] + docs[0]])["input_ids"][0]
print("last tokens with shim:", repr(m.tokenizer.decode(ids[-4:])))
