#!/usr/bin/env python
"""Compose results.md = results_header.md + generated tables (results/tables.md) + notes.md with the side-experiment table filled in."""
import json, os
from pathlib import Path
ROOT = Path(os.environ.get("EE_ROOT", Path.home() / "embed-eval"))
T = ["CodeSearchNetCCRetrieval", "CodeTransOceanContest", "CodeTransOceanDL", "CosQA", "AppsRetrieval", "SyntheticText2SQL"]
TIE = {"CodeTransOceanDL", "SyntheticText2SQL", "CosQA"}
STAR = " \\*"
c = json.loads((ROOT / "results" / "taskinstr_comparison.json").read_text())
models = list(c)
L = ["| Task | " + " | ".join(f"{m}: generic → task instr (Δ)" for m in models) + " |", "|---|" + "---|" * len(models)]
for t in T:
    L.append(f"| {t}{STAR if t in TIE else ''} | " + " | ".join(
        f"{c[m][t]['generic']:.3f} → {c[m][t]['task_instr']:.3f} ({c[m][t]['task_instr'] - c[m][t]['generic']:+.3f})" for m in models) + " |")
for label, ts in [("Mean, 6 tasks", T), ("Mean, 3 tie-robust tasks", [t for t in T if t not in TIE])]:
    row = []
    for m in models:
        g = sum(c[m][t]["generic"] for t in ts) / len(ts); i = sum(c[m][t]["task_instr"] for t in ts) / len(ts)
        row.append(f"**{g:.3f} → {i:.3f} ({i - g:+.3f})**")
    L.append(f"| **{label}** | " + " | ".join(row) + " |")
L.append("")
L.append("\\* tie-dominated task; its deltas reflect tie-breaking noise and are not meaningful.")
notes = (ROOT / "notes.md").read_text().replace("SIDE_EXPERIMENT_TABLE", "\n".join(L))
out = (ROOT / "results_header.md").read_text() + "# Detailed results\n\n" + (ROOT / "results" / "tables.md").read_text() + notes
(ROOT / "results.md").write_text(out)
print("wrote results.md", len(out.splitlines()), "lines")
