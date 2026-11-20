"""Analysis of E2c (e2c_freq_cv.py): top-k by FENNEC frequency with k chosen by inner CV, vs fixed thresholds.

    python -m experiments.analyze_e2c
Writes results/tables/E2c_a.csv (and TABLES_e2c.md).
"""
import json

import pandas as pd

from experiments.common import MD, TABLES, boot_ci, table
from experiments.e2c_freq_cv import RAW

KEY = ["p", "n", "signal_r2", "seed"]


def main():
    recs = [json.loads(l) for p in sorted(RAW.glob("e2c_shard*.jsonl")) for l in p.read_text().splitlines()]
    d = pd.DataFrame([{**{k: r[k] for k in KEY}, **x} for r in recs for x in r["rows"]])
    print(f"{len(recs)} systems")
    rows = []
    for (pr, s), g in d.groupby(["predictor", "set"]):
        base = d[(d.predictor == pr) & (d.set == "all")].set_index(KEY)
        g = g.set_index(KEY)
        row = {"model": pr, "features": s, "n selected": g.n_selected.mean(), "true causes hit": g.true_hit.mean()}
        for reg in ("iid", "proxy_break"):
            diff = (g[f"r2_{reg}"] - base[f"r2_{reg}"]).dropna()
            mu, lo, hi = boot_ci(diff)
            row[f"R2 {reg}"] = g[f"r2_{reg}"].mean()
            row[f"gain vs all ({reg})"] = f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            if reg == "iid":
                row["share of systems better than all"] = (diff > 0).mean()
        rows.append(row)
    t = pd.DataFrame(rows).set_index(["model", "features"])
    print(t.round(3).to_string())
    table(t, "E2c_a", "E2c: FENNEC frequency selection, inner-CV top-k vs fixed thresholds (paired vs all features)",
          "The frequencies use all training rows, so the inner CV over k is optimistic; it picks too many features.")
    (TABLES / "TABLES_e2c.md").write_text("# E2c tables\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
