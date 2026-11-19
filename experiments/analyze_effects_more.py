"""Causal-effect error for every method on the same 96 systems: exp_effects.py (FENNEC, CAM, GES, true graph,
no effect) + exp_effects_more.py (BiDAG, PC, NOTEARS-MLP, DirectLiNGAM, DoubleML, g-computation).

Error = |estimated - true| effect in SD units, averaged over (target, outcome) queries; CIs are
system-cluster bootstraps of the paired difference to FENNEC's best graph.
Writes results/figures/fig16_effects_vs_literature.png and T16a-b (TABLES_effects_more.md).
"""
import json

import numpy as np
import pandas as pd

from experiments.exp_effects import OUT as RES
from experiments.common import AQUA, BLUE, GRID, INK2, MUTED, ORANGE, OUT, RED, plt, save_fig, table

MECH = ["linear", "additive", "mlp", "mixed"]
MLAB = {"linear": "linear", "additive": "nonlinear additive", "mlp": "nonlinear + interactions", "mixed": "nonlinear + mixed types"}
LAB = {"true": "true graph (reference)", "fennec_best": "FENNEC best graph", "fennec_pop": "FENNEC population",
       "cam": "CAM graph", "bidag": "BiDAG posterior (20 graphs)", "ges": "GES graph", "pc": "PC graph",
       "notears": "NOTEARS-MLP graph", "lingam": "DirectLiNGAM graph",
       "dml_truepa": "double ML | true parents of T", "gcomp_truepa": "g-computation | true parents of T",
       "dml_fennecpa": "double ML | FENNEC's parents of T", "gcomp_fennecpa": "g-computation | FENNEC's parents of T",
       "dml_all": "double ML | all other variables", "gcomp_all": "g-computation | all other variables",
       "empty": "no effect (all zero)"}
KIND = {**{k: "graph" for k in ("true", "fennec_best", "fennec_pop", "cam", "bidag", "ges", "pc", "notears", "lingam")},
        **{k: "graph-free" for k in LAB if k.startswith(("dml", "gcomp"))}, "empty": "reference"}
KCOL = {"graph": BLUE, "graph-free": ORANGE, "reference": MUTED}


def load(name):
    rows = []
    for l in (RES / name).read_text().splitlines():
        r = json.loads(l)
        if "error" in r:
            continue
        for q in r["rows"]:
            rows.append({**q, "mechanism": r["mechanism"], "system": f'{r["mechanism"]}-{r["d"]}-{r["n"]}-{r["seed"]}',
                         "d": r["d"], "n": r["n"]})
    return pd.DataFrame(rows)


def main():
    a, b = load("effects.jsonl"), load("effects_more.jsonl")
    g = load("effects_more_gcomp.jsonl") if (RES / "effects_more_gcomp.jsonl").exists() else None
    if g is not None and len(g):  # fixed g-computation rows replace the first version's
        b = pd.concat([b[~b.method.str.startswith("gcomp")], g[g.method.str.startswith("gcomp") & g.system.isin(set(b.system))]])
        b = b[~b.method.str.startswith("gcomp") | b.system.isin(set(g.system))]
    df = pd.concat([a[a.method.isin(["true", "fennec_best", "fennec_pop", "cam", "ges", "empty"])], b], ignore_index=True)
    systems = set(b.system)
    df = df[df.system.isin(systems)]
    print(f"{len(systems)} systems, methods: {sorted(df.method.unique())}")
    df["abs_err"] = (df.est - df.true).abs()
    per = df.groupby(["method", "mechanism", "system"]).abs_err.mean().reset_index()
    order = [m for m in LAB if m in set(df.method)]
    t = per.groupby(["method", "mechanism"]).abs_err.mean().unstack()[MECH].reindex(order)
    t["all"] = per.groupby("method").abs_err.mean().reindex(order)
    t.index = [LAB[m] for m in t.index]
    pd.set_option("display.width", 200)
    print("\nT16a mean absolute effect error (SD units)\n", t.round(3))
    table(t, "T16a", "Causal-effect error by method (mean |estimate - truth|, SD units; same systems and queries)")
    base = per[per.method == "fennec_best"].set_index(["mechanism", "system"]).abs_err
    rng = np.random.default_rng(0)
    rows = []
    for m in order:
        if m == "fennec_best":
            continue
        o = per[per.method == m].set_index(["mechanism", "system"]).abs_err
        for mech in MECH:
            dlt = (o.xs(mech) - base.xs(mech)).dropna()
            if not len(dlt):
                continue
            bs = [rng.choice(dlt.values, len(dlt)).mean() for _ in range(2000)]
            rows.append({"method": LAB[m], "mechanism": mech, "kind": KIND[m], "key": m, "diff": dlt.mean(),
                         "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975), "systems": len(dlt)})
    c = pd.DataFrame(rows)
    c["verdict"] = np.where(c.hi < 0, "better than FENNEC", np.where(c.lo > 0, "worse than FENNEC", "no clear difference"))
    tc = c.pivot(index="method", columns="mechanism", values="verdict")[MECH].reindex([LAB[m] for m in order if m != "fennec_best"])
    print("\nT16b vs FENNEC best graph (paired, 95% CI excludes 0)\n", tc)
    table(tc, "T16b", "Each method vs FENNEC's best graph: paired difference in effect error, per mechanism",
          "better / worse = the 95% system-bootstrap CI of the paired difference excludes 0.")
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.4), sharey=True)
    meths = [m for m in order if m != "fennec_best"]
    ys = np.arange(len(meths))[::-1]
    for ax, mech in zip(axes, MECH):
        s = c[c.mechanism == mech].set_index("key").reindex(meths)
        for y, m in zip(ys, meths):
            if pd.isna(s.loc[m, "diff"]):
                continue
            ax.plot([s.loc[m, "lo"], s.loc[m, "hi"]], [y, y], color=KCOL[KIND[m]], lw=2)
            ax.plot(s.loc[m, "diff"], y, "o", color=KCOL[KIND[m]], ms=7, mec="white", mew=1)
        ax.axvline(0, color=INK2, lw=0.9)
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_title(MLAB[mech], fontsize=10.5); ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_xlabel("error minus FENNEC's error (SD)\n← more accurate than FENNEC    less accurate →", fontsize=8.5)
    axes[0].set_yticks(ys, [LAB[m] for m in meths], fontsize=9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=KCOL[k], marker="o", lw=2, label=l) for k, l in
         (("graph", "graph-based (same causal model, different graph)"), ("graph-free", "graph-free estimator, given an adjustment set"),
          ("reference", "reference"))]
    fig.legend(handles=h, frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle(f"Causal-effect accuracy: FENNEC vs published methods ({len(systems)} synthetic systems, paired 95% CIs)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_fig(fig, "fig16_effects_vs_literature.png")
    from experiments.common import MD, TABLES
    (TABLES / "TABLES_effects_more.md").write_text("# Effects vs literature tables\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
