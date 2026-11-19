"""Analysis of E1 (e1_quality_vs_gain.py): prediction gain over M0 (XGBoost, all features) vs graph quality.

    python -m experiments.analyze_e1 [--pilot] [--seeds 5-9|all]
Writes results/tables/E1a-c.csv and results/figures/fig_e1_gain_vs_quality.png.
Default seeds 5-9: FENNEC's penalty was chosen on seeds 0-4.
"""
import argparse
import json

import numpy as np
import pandas as pd

from experiments.common import BLUE, GRID, INK2, MUTED, ORANGE, RED, AQUA, TABLES, boot_ci, plt, save_fig, table

from experiments.e1_quality_vs_gain import RAW

KEY = ["mechanism", "d", "n", "seed"]
SHOW = ["M1_parents", "M1_blanket", "M2_constrained", "M3_chain", "M3_chain_stacked"]


def gtype(g):
    if g.startswith("corrupt"):
        return "corrupted oracle"
    if g.startswith("random"):
        return "random"
    return {"oracle": "oracle", "empty": "empty"}.get(g, "learned")


def load(pilot, seeds):
    path = RAW / ("e1_pilot.jsonl" if pilot else "e1.jsonl")
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    d = pd.DataFrame([{**{k: r[k] for k in KEY}, **x} for r in recs for x in r["rows"]])
    if "error" in d:
        print("compiler errors:", int(d.error.notna().sum()))
        d = d[d.error.isna()]
    d = d[d.seed.isin(seeds)]
    m0 = d[(d.compiler == "M0_all") & (d.graph == "oracle")].set_index(KEY).r2.rename("r2_m0")
    d = d.join(m0, on=KEY)
    d["gain"] = d.r2 - d.r2_m0
    d["type"] = d.graph.map(gtype)
    return d, len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seeds", default="5-9")
    a = ap.parse_args()
    seeds = range(10) if a.seeds == "all" else range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1)
    P = "E1pilot" if a.pilot else "E1"
    d, nrec = load(a.pilot, seeds)
    nsys = d.groupby(KEY).ngroups
    print(f"{nrec} systems in file, {nsys} used (seeds {a.seeds})")

    # E1a: mean test R^2 and gain over M0 with 95% bootstrap CI over systems
    rows = []
    for (g, c), s in d.groupby(["graph", "compiler"]):
        mu, lo, hi = boot_ci(s.gain.to_numpy())
        rows.append({"graph": g, "compiler": c, "R2": s.r2.mean(), "gain vs M0": mu, "lo": lo, "hi": hi,
                     "SHD": s.shd.mean(), "blanket F1": s.blanket_f1.mean(), "systems": len(s)})
    t = pd.DataFrame(rows).set_index(["graph", "compiler"])
    print(t.round(3).to_string())
    table(t, P + "a", f"E1: test R^2 and gain over XGBoost-all by graph and compiler (seeds {a.seeds})")

    # E1b: gain by sample size for the oracle and learned graphs
    t2 = d[d.type.isin(["oracle", "learned"])].groupby(["graph", "n", "compiler"]).gain.mean().unstack("compiler")
    print(t2.round(3).to_string())
    table(t2, P + "b", "E1: mean gain over M0 by graph and n")

    # E1c: rank correlation of gain with graph quality, per compiler
    from scipy.stats import spearmanr
    rc = []
    for c in SHOW:
        s = d[(d.compiler == c) & (d.type != "empty")]
        rc.append({"compiler": c, "rho(gain, -SHD)": spearmanr(s.gain, -s.shd).statistic,
                   "rho(gain, blanket F1)": spearmanr(s.gain, s.blanket_f1).statistic,
                   "rho(gain, parent F1)": spearmanr(s.gain, s.parent_f1).statistic})
    rc = pd.DataFrame(rc).set_index("compiler")
    print(rc.round(3).to_string())
    table(rc, P + "c", "E1: Spearman correlation between gain and graph quality (excluding the empty graph)")

    col = {"oracle": RED, "corrupted oracle": ORANGE, "random": MUTED, "learned": BLUE}
    fig, axes = plt.subplots(2, len(SHOW), figsize=(4 * len(SHOW), 7.5), sharey="row")
    for k, c in enumerate(SHOW):
        s = d[(d.compiler == c) & (d.type != "empty")]
        for row, (x, xl) in enumerate([("shd", "SHD to true graph"), ("blanket_f1", "outcome Markov-blanket F1")]):
            ax = axes[row, k]
            for ty, cc in col.items():
                q = s[s.type == ty]
                ax.scatter(q[x], q.gain.clip(-0.6, 0.3), s=7, alpha=0.45, color=cc, label=ty, lw=0)
            bins = pd.qcut(s[x], 8, duplicates="drop")
            bm = s.groupby(bins, observed=True).agg(x=(x, "mean"), g=("gain", "mean"))
            ax.plot(bm.x, bm.g, color=INK2, lw=1.6)
            ax.axhline(0, color=INK2, lw=0.7)
            ax.set_xlabel(xl)
            ax.grid(color=GRID, lw=0.5)
            if row == 0:
                ax.set_title(c)
    axes[0, 0].set_ylabel("R² gain over XGBoost-all (clipped)")
    axes[1, 0].set_ylabel("R² gain over XGBoost-all (clipped)")
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=2)
    fig.suptitle(f"E1: graph quality vs prediction gain ({nsys} systems, seeds {a.seeds}; line = binned mean)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "fig_e1_gain_vs_quality_pilot.png" if a.pilot else "fig_e1_gain_vs_quality.png")
    from experiments.common import MD
    (TABLES / ("TABLES_e1_pilot.md" if a.pilot else "TABLES_e1.md")).write_text("# E1 tables\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
