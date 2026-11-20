"""Analysis of E2 (e2_cohort.py): CHILD-like data, many small effects.

    python -m experiments.analyze_e2 [--pilot]
Baseline = elastic net on all features (the strongest simple model, and the best model on CHILD).
Writes results/tables/E2_a-d.csv and results/figures/fig_e2_cohort.png.
"""
import argparse
import json

import numpy as np
import pandas as pd

from experiments.common import AQUA, BLUE, GRID, INK2, MD, MUTED, ORANGE, RED, TABLES, boot_ci, plt, save_fig, table
from experiments.e2_cohort import RAW

KEY = ["p", "n", "signal_r2", "seed"]
REG = ["iid", "proxy_break", "cause_shift"]
RLAB = {"iid": "same distribution", "proxy_break": "proxies break", "cause_shift": "causes shift +1 SD"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    paths = [RAW / "e2_pilot.jsonl"] if a.pilot else sorted(p for p in RAW.glob("e2*.jsonl") if "pilot" not in p.name)
    recs = [json.loads(l) for p in paths for l in p.read_text().splitlines()]
    d = pd.DataFrame([{**{k: r[k] for k in KEY}, **x} for r in recs for x in r["rows"]])
    d["route"] = d.set + " -> " + d.predictor
    P = "E2pilot_" if a.pilot else "E2_"
    print(f"{len(recs)} systems")

    base = d[(d.set == "all") & (d.predictor == "enet")].set_index(KEY)
    rows = []
    for (s, pr), g in d.groupby(["set", "predictor"]):
        g = g.set_index(KEY)
        row = {"features": s, "model": pr, "n selected": g.n_selected.mean(), "true causes hit (of 15)": g.true_hit.mean(),
               "proxies included": g.proxies.mean()}
        for reg in REG:
            row[f"R2 {RLAB[reg]}"] = g[f"r2_{reg}"].mean()
            mu, lo, hi = boot_ci((g[f"r2_{reg}"] - base[f"r2_{reg}"]).dropna())
            row[f"gain {RLAB[reg]}"] = f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
    t = pd.DataFrame(rows).set_index(["features", "model"])
    print(t.round(3).to_string())
    table(t, P + "a", "E2: test R^2 by feature set and model; gain vs elastic net on all features (95% CI over systems)")

    for reg in ("iid", "proxy_break"):
        sub = d[d.predictor == "enet"].pivot_table(index=["p", "n"], columns="set", values=f"r2_{reg}")
        sub = sub.sub(sub["all"], axis=0).drop(columns="all")
        print(f"\nenet, gain vs all, {reg}\n", sub.round(3).to_string())
        table(sub, P + ("b" if reg == "iid" else "c"), f"E2: elastic-net gain over all features by p and n ({RLAB[reg]})")

    sel = d[d.predictor == "enet"].groupby(["set", "p"])[["n_selected", "true_hit", "proxies"]].mean().unstack("p")
    print(sel.round(1).to_string())
    table(sel, P + "d", "E2: selection size, true causes hit (of 15) and proxies included, by p")

    show = [("all", "enet", INK2, "-"), ("all", "xgb", INK2, ":"), ("oracle", "enet", RED, "-"), ("lasso", "enet", ORANGE, "-"),
            ("fennec_lin0.5", "enet", BLUE, "-"), ("fennec_lin1", "enet", AQUA, "-"), ("fennec_lin4", "enet", MUTED, "-")]
    ps = sorted(d.p.unique())
    fig, axes = plt.subplots(len(REG), len(ps), figsize=(4.6 * len(ps), 3.4 * len(REG)), sharey="row", squeeze=False)
    for i, reg in enumerate(REG):
        for j, p in enumerate(ps):
            ax = axes[i, j]
            for s, pr, c, ls in show:
                q = d[(d.set == s) & (d.predictor == pr) & (d.p == p)].groupby("n")[f"r2_{reg}"].mean()
                ax.plot(q.index, q.values, color=c, ls=ls, marker="o", ms=3, label=f"{s} -> {pr}")
            ax.set_xscale("log"); ax.set_xticks(sorted(d.n.unique()), [str(v) for v in sorted(d.n.unique())])
            ax.grid(color=GRID, lw=0.5)
            if i == 0:
                ax.set_title(f"p = {p} features")
            if j == 0:
                ax.set_ylabel(f"test R², {RLAB[reg]}")
            if i == len(REG) - 1:
                ax.set_xlabel("training rows n")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(f"E2: CHILD-like data, 15 weak causes among p features ({len(recs)} systems, both signal levels pooled)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "fig_e2_cohort_pilot.png" if a.pilot else "fig_e2_cohort.png")
    (TABLES / f"TABLES_{P.lower().rstrip('_')}.md").write_text("# E2 tables\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
