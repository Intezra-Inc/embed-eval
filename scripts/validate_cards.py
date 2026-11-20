#!/usr/bin/env python
"""Check the harness's model loading against each model card's published example scores.

Uses run_eval.load_encoder (the exact path the benchmark uses), then the card's query/document example.
Usage: CUDA_VISIBLE_DEVICES=0 python validate_cards.py nemotron-8b|zembed-1
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_eval import load_encoder  # noqa: E402

_Q = dict(queries=["What is the capital of China?", "Explain gravity"],
          docs=["The capital of China is Beijing.",
                "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."])
CARDS = {
    "qwen3-0.6b": dict(**_Q, expected=[[0.7646, 0.1414], [0.1355, 0.6000]]),
    "qwen3-8b": dict(**_Q, expected=[[0.7493, 0.0751], [0.0880, 0.6318]]),
    "nemotron-8b": dict(
        queries=["Write a Python function that counts the frequency of each element in a list of lists.",
                 "Write a function that orders a dictionary with tuple keys by the product of each key's tuple values.",
                 "What symptoms and common triggers help distinguish eczema from other inflammatory skin conditions?",
                 "How can someone reduce exposure to pollen during allergy season?"],
        docs=["def frequency_lists(list1):\n    flattened = [item for sublist in list1 for item in sublist]\n    counts = {}\n    for item in flattened:\n        if item in counts:\n            counts[item] += 1\n        else:\n            counts[item] = 1\n    return counts",
              "def sort_dict_item(test_dict):\n    return {key: test_dict[key] for key in sorted(test_dict.keys(), key=lambda ele: ele[0] * ele[1])}",
              "Eczema commonly causes itchy, dry, inflamed patches of skin. The affected areas may look red, scaly, cracked, or darker than the surrounding skin depending on skin tone. Symptoms can flare after exposure to irritants, allergens, stress, or changes in weather.",
              "People with pollen allergy can reduce exposure by staying indoors on dry, windy days, avoiding early-morning outdoor activity, and going outside after rain when pollen levels are lower. They should check pollen forecasts, close windows and doors when counts are high, and consider starting allergy medication before symptoms begin if high pollen is expected. After being outside, showering, changing clothes, avoiding outdoor laundry drying, and wearing a face mask for yard work can help limit pollen contact."],
        expected=[[0.7852, 0.0496, 0.0115, -0.0299], [0.0457, 0.6562, -0.0645, 0.0452],
                  [-0.0016, -0.0530, 0.6602, 0.1123], [-0.0170, -0.0036, 0.1147, 0.8008]]),
    "zembed-1": dict(
        queries=["What is backpropagation?"],
        docs=["Backpropagation is a fundamental algorithm for training neural networks by computing gradients.",
              "Gradient descent is used to optimize model parameters during the training process.",
              "Neural network training relies on efficient computation of derivatives through backpropagation."],
        expected=[[0.7525, 0.5670, 0.6835]]),
}


def main():
    key = sys.argv[1]
    c = CARDS[key]
    enc, info = load_encoder(key)
    st = enc.model
    s = st.similarity(st.encode_query(c["queries"]), st.encode_document(c["docs"])).float().numpy()
    diff = np.abs(s - np.array(c["expected"])).max()
    print(json.dumps({"model": key, "zembed_compat_shim": info.get("zembed_compat_shim"),
                      "scores": s.round(4).tolist(), "card": c["expected"], "max_abs_diff": round(float(diff), 4),
                      "ok": bool(diff < 0.02)}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
