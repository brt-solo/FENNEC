"""E0 artefact checks (PROJECT_CONTEXT.md section 6).

1. Sortability: varsortability and R^2-sortability of the main generator and of gCastle's generator
   (raw and standardized). Computed directly here (cheap).
2. Penalty audit: skeleton F1 of FENNEC with 1x, 2x, 4x, 8x the BIC penalty, separately on the
   exploratory seeds 0-4 and the held-out seeds 5-9. Is 4x still the best choice on 5-9?
3. Outside generator: FENNEC vs baselines on gCastle data (no FENNEC choice was made on it).
   Also the sort-by-variance and sort-by-R^2 baselines on both generators.

Steps 2-3 read benchmark results; produce them first (see run_e0.sh):
    python -m benchmarks.run --methods fennec_pen2 fennec_pen8 var_sortnregress
    python -m benchmarks.run --generator gcastle --methods <gcastle_methods from configs/e0.json>
Then:
    python -m experiments.e0_artifacts
Writes results/tables/E0a-d.csv and TABLES_e0.md.
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.generate import MECHANISMS, simulate
from benchmarks.generate_gcastle import GC_MECHANISMS, simulate_gcastle
from benchmarks.sortability import r2_sortability, varsortability
from benchmarks.summarize import LABEL
from experiments.common import MD, TABLES, boot_ci, table

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "results"


def sortability(cfg):
    g = cfg["sortability"]
    rows = []
    for m, d, n, s in product(MECHANISMS, g["d"], g["n"], g["seeds"]):
        df, adj, _ = simulate(d, n, m, s)
        X = df.to_numpy()
        rows.append({"generator": "FENNEC benchmark", "graph": "ER", "mechanism": m,
                     "varsortability (raw)": np.nan, "varsortability": varsortability(X, adj),
                     "R2-sortability": r2_sortability(X, adj)})
    for gr, m, d, n, s in product(["ER", "SF"], GC_MECHANISMS, g["d"], g["n"], g["seeds"]):
        df, adj, info = simulate_gcastle(d, n, m, s, graph=gr)
        X = df.to_numpy()
        rows.append({"generator": "gCastle", "graph": gr, "mechanism": m,
                     "varsortability (raw)": info["varsortability_raw"], "varsortability": varsortability(X, adj),
                     "R2-sortability": r2_sortability(X, adj)})
    return pd.DataFrame(rows).groupby(["generator", "graph", "mechanism"]).mean()


def load(name):
    p = BENCH / name
    if not p.exists():
        return None
    d = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines()])
    return d[d.error.isna() & (d.seed < 10)]  # the grid is seeds 0-9; some files hold extra seeds


def penalty_audit(cfg):
    d = load("results.jsonl")
    pm = cfg["penalty_methods"]
    d = d[d.method.isin(pm)].copy()
    d["mult"] = d.method.map(pm)
    d["split"] = np.where(d.seed < 5, "exploratory seeds 0-4", "held-out seeds 5-9")
    t = d.groupby(["split", "mult"]).agg(skel_f1=("skel_f1", "mean"), shd=("shd", "mean"),
                                         datasets=("skel_f1", "size")).unstack("split")
    by_cell = d.groupby(["split", "mechanism", "n", "mult"]).skel_f1.mean().unstack("mult")
    best = by_cell.idxmax(axis=1).unstack("split")
    # paired difference 4x minus each alternative on held-out seeds
    held = d[d.seed >= 5].pivot_table(index=["mechanism", "d", "n", "seed"], columns="mult", values="skel_f1")
    diffs = []
    for m in sorted(pm.values()):
        if m == 4 or m not in held:
            continue
        mu, lo, hi = boot_ci((held[4] - held[m]).dropna())
        diffs.append({"comparison": f"4x minus {m}x (held-out)", "skel F1 diff": mu, "lo": lo, "hi": hi})
    return t, best, pd.DataFrame(diffs).set_index("comparison") if diffs else None


def gcastle_table():
    d = load("results_gcastle.jsonl")
    if d is None:
        return None, None
    d = d.copy()
    d["orient_recall"] = d.dir_tp / d.true_edges.clip(lower=1)
    f1 = d.groupby(["method", "graph", "mechanism", "n"]).skel_f1.mean().unstack(["graph", "mechanism", "n"])
    f1 = f1.loc[f1.mean(axis=1).sort_values(ascending=False).index]
    f1.index = [LABEL.get(i, i) for i in f1.index]
    summ = d.groupby("method").agg(skel_f1=("skel_f1", "mean"), orient_recall=("orient_recall", "mean"),
                                   shd=("shd", "mean"), seconds=("seconds", "mean"))
    summ = summ.sort_values("skel_f1", ascending=False)
    summ.index = [LABEL.get(i, i) for i in summ.index]
    return f1, summ


def sort_baselines():
    rows = []
    for name, f in [("FENNEC benchmark", "results.jsonl"), ("gCastle", "results_gcastle.jsonl")]:
        d = load(f)
        if d is None:
            continue
        for m in ["r2_sortnregress", "var_sortnregress", "fennec"]:
            s = d[d.method == m]
            if len(s):
                rows.append({"generator": name, "method": LABEL.get(m, m), "skel F1": s.skel_f1.mean(),
                             "oriented recall": (s.dir_tp / s.true_edges.clip(lower=1)).mean(), "datasets": len(s)})
    return pd.DataFrame(rows).set_index(["generator", "method"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e0.json")
    cfg = json.loads(ap.parse_args().config.read_text())
    s = sortability(cfg)
    print(s.round(3).to_string())
    table(s, "E0a", "Sortability (0.5 = variance carries no order information; R2-sortability ~0 = none)",
          "Raw = gCastle data before standardization. Both generators are standardized before any method sees them.")
    t, best, diffs = penalty_audit(cfg)
    print(t.round(3).to_string()); print(best.to_string())
    table(t, "E0b", "Penalty audit: FENNEC skeleton F1 / SHD by BIC multiplier, exploratory vs held-out seeds")
    table(best, "E0b2", "Best multiplier per (mechanism, n) cell, by seed split")
    if diffs is not None:
        print(diffs.round(3).to_string())
        table(diffs, "E0b3", "Held-out seeds: paired skeleton-F1 difference, 4x minus alternative (95% CI)")
    f1, summ = gcastle_table()
    if f1 is not None:
        print(summ.round(3).to_string())
        table(summ, "E0c", "Outside generator (gCastle, standardized): mean over all datasets")
        table(f1, "E0c2", "Outside generator: skeleton F1 by graph type, mechanism and n")
    sb = sort_baselines()
    print(sb.round(3).to_string())
    table(sb, "E0d", "Sort-and-regress artefact baselines vs FENNEC")
    (TABLES / "TABLES_e0.md").write_text("# E0 artefact checks\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
