"""Analysis of E2b (e2b_cohort_noisy.py): noisy CHILD-like regime where all features hurt.

    python -m experiments.analyze_e2b [--pilot]
Baselines: elastic net on all features, and regularized XGBoost on all features.
Writes results/tables/E2b_a-d.csv and results/figures/fig_e2b_selection.png.
"""
import argparse
import json

import numpy as np
import pandas as pd

from experiments.common import AQUA, BLUE, GRID, INK2, MD, MUTED, ORANGE, RED, TABLES, boot_ci, plt, save_fig, table
from experiments.e2b_cohort_noisy import RAW

KEY = ["p", "n", "signal_r2", "seed"]
REG = ["iid", "proxy_break", "cause_shift"]
RLAB = {"iid": "same distribution", "proxy_break": "proxies break", "cause_shift": "causes shift"}
ORDER = ["all", "oracle", "lasso", "fennec", "fennec_tebic", "fennec_tseq", "fennec_xgbreg",
         "fennec_freq0.3", "fennec_freq0.5", "fennec_freq0.7", "hybrid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    paths = [RAW / "e2b_pilot.jsonl"] if a.pilot else sorted(p for p in RAW.glob("e2b*.jsonl") if "pilot" not in p.name)
    recs = [json.loads(l) for p in paths for l in p.read_text().splitlines()]
    d = pd.DataFrame([{**{k: r[k] for k in KEY}, **x} for r in recs for x in r["rows"]])
    P = "E2b_pilot_" if a.pilot else "E2b_"
    print(f"{len(recs)} systems")

    bases = {m: d[(d.set == "all") & (d.predictor == m)].set_index(KEY) for m in ("enet", "xgb_reg")}
    rows = []
    for (s, pr), g in d.groupby(["set", "predictor"]):
        g = g.set_index(KEY)
        row = {"features": s, "model": pr, "n selected": g.n_selected.mean(), "true causes hit": g.true_hit.mean(),
               "proxies": g.proxies.mean()}
        for reg in REG:
            row[f"R2 {RLAB[reg]}"] = g[f"r2_{reg}"].mean()
        for bm, b in bases.items():
            for reg in ("iid", "proxy_break"):
                mu, lo, hi = boot_ci((g[f"r2_{reg}"] - b[f"r2_{reg}"]).dropna())
                row[f"vs all->{bm}, {RLAB[reg]}"] = f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
    t = pd.DataFrame(rows)
    t["o"] = t.features.map({s: i for i, s in enumerate(ORDER)})
    t = t.sort_values(["o", "model"]).drop(columns="o").set_index(["features", "model"])
    print(t.round(3).to_string())
    table(t, P + "a", "E2b: test R^2 by feature set and model; paired gains vs all features (95% CI over systems)")

    for pr in ("enet", "xgb_reg"):
        sub = d[d.predictor == pr].pivot_table(index=["p", "n"], columns="set", values="r2_iid")
        sub = sub.sub(sub["all"], axis=0).drop(columns="all").reindex(columns=[c for c in ORDER if c in sub and c != "all"])
        print(f"\n{pr}: gain vs all features, same distribution\n", sub.round(3).to_string())
        table(sub, P + ("b" if pr == "enet" else "c"), f"E2b: {pr} gain over all features by p and n (same distribution)")

    fq = pd.DataFrame([{k: r[k] for k in KEY + ["freq_true", "freq_proxy", "freq_other", "top30_true"]} for r in recs])
    f = fq.groupby("p")[["freq_true", "freq_proxy", "freq_other", "top30_true"]].mean()
    print("\nselection frequency (mean per feature class), top-30 true causes\n", f.round(3).to_string())
    table(f, P + "d", "E2b: mean population selection frequency of true causes / proxies / other features")

    show = [("all", "enet", INK2, "-"), ("all", "xgb_reg", INK2, ":"), ("oracle", "enet", RED, "-"),
            ("lasso", "enet", ORANGE, "-"), ("fennec", "enet", MUTED, "-"), ("fennec_freq0.3", "enet", BLUE, "-"),
            ("fennec_tebic", "enet", AQUA, "-"), ("hybrid", "enet", "#7b4fd6", "-"), ("fennec_freq0.3", "xgb_reg", BLUE, ":")]
    ps = sorted(d.p.unique())
    fig, axes = plt.subplots(2, len(ps), figsize=(4.6 * len(ps), 7), sharey="row", squeeze=False)
    for i, reg in enumerate(["iid", "proxy_break"]):
        for j, p in enumerate(ps):
            ax = axes[i, j]
            for s, pr, c, ls in show:
                q = d[(d.set == s) & (d.predictor == pr) & (d.p == p)].groupby("n")[f"r2_{reg}"].mean()
                ax.plot(q.index, q.values, color=c, ls=ls, marker="o", ms=3, label=f"{s} -> {pr}")
            ax.set_xticks(sorted(d.n.unique()))
            ax.grid(color=GRID, lw=0.5)
            if i == 0:
                ax.set_title(f"p = {p} features")
            if j == 0:
                ax.set_ylabel(f"test R², {RLAB[reg]}")
            if i == 1:
                ax.set_xlabel("training rows n")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(f"E2b: noisy CHILD-like data, all features hurt ({len(recs)} systems, signal levels pooled)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "fig_e2b_selection_pilot.png" if a.pilot else "fig_e2b_selection.png")
    (TABLES / f"TABLES_{P.lower()}md".replace("_md", ".md")).write_text("# E2b tables\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
