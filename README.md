# embed-eval

Embedding-model evaluations run with mteb 2.24.10 (BF16, max length 8,192, seed 42).

- [results.md](results.md): code retrieval, 6 models (MTEB(Code, v1) + RTEB(Code) public), with cost measurements
- [results_mmteb.md](results_mmteb.md): multilingual retrieval, Nemotron-3-Embed-8B vs Qwen3-Embedding-8B vs Harrier-27B (MTEB(Multilingual, v2) retrieval tasks)
- Quantized Harrier-27B for 24 GB cards: [Intezra-Inc/harrier-quant](https://github.com/Intezra-Inc/harrier-quant) (summarized in both result files)
- [COMMANDS.md](COMMANDS.md): how to reproduce every run
- `results/`: raw mteb result JSONs and merged scores; `logs/`: run logs; `env/`: pip freeze and model weight checksums

Not included: venv (rebuild from `env/freeze.txt`), model weights (pinned revisions in results.md), `cost/docs.jsonl` (`scripts/cost_bench.py build`).

## Licence

- **Code** (`scripts/`, `configs/`): Apache License 2.0, see [`LICENSE`](LICENSE).
- **Results and write-ups** (`results.md`, `results_mmteb.md`, the header and notes files, `results/`, `cost/`, `logs/`, `env/`):
  Creative Commons Attribution 4.0 International, see [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0). You may reuse and adapt them,
  including commercially, with credit to Intezra, Inc.
- The evaluated models and the mteb datasets are not included and stay under their own licences (listed in `results.md`).

Copyright 2026 Intezra, Inc.
