"""FENNEC heterogeneity tools vs CD-NOD and JCI-style FENNEC, at the level all of them report: which NODES have
a mechanism that changes with the environment / genetic context.

FENNEC tools (stored results, same systems):
    islands, light migration (exp_islands island_m2)   node score = max over incoming edges of max_k P - min_k P;
                                                          called if above the 95th percentile of the random-split null
    score-gain test (exp_islands)                      node p = min p over tested incoming edges; called if p < 0.05
    pathway-PGS clades (exp_genetic_clades A, 4 clades) node score as islands; null = random clades
    pooled interaction test (exp_genetic_clades A)     node p = min p over tested incoming edges
Baselines (exp_hetero_baselines.py): CD-NOD with Fisher-z or RCIT tests; JCI-style FENNEC.
Truth: changing nodes = cut nodes and children of environment- / genotype-specific edges. Genetic cohorts
also have context-shifted nodes (a genetic main effect, or a root whose mean differs by ancestry, which the
PGS track; no edge changes): reported separately, since a
context-variable method flags them by design while edge-level tools should not.
Writes results/figures/fig17_hetero_vs_cdnod.png and T17a-b (TABLES_hetero_baselines.md).
"""
import json
import textwrap

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.exp_effects import OUT as RES
from experiments.common import AQUA, BLUE, GRID, INK2, MUTED, ORANGE, OUT, RED, plt, save_fig, table

LAB = {"fennec_islands": "FENNEC islands", "score_gain": "score-gain test (FENNEC score)",
       "fennec_clades": "FENNEC pathway-PGS clades", "interaction": "pooled G x E interaction test",
       "cdnod_fisherz": "CD-NOD (Fisher-z)", "cdnod_rcit": "CD-NOD (RCIT)", "fennec_jci": "JCI-style FENNEC"}
COL = {"fennec_islands": BLUE, "fennec_clades": BLUE, "score_gain": "#6da7ec", "interaction": ORANGE,
       "cdnod_fisherz": MUTED, "cdnod_rcit": INK2, "fennec_jci": AQUA}
EDGE = {"fennec_islands": "yes, and which environment", "score_gain": "yes", "fennec_clades": "yes, and which clade",
        "interaction": "yes, but leaks to neighbours", "cdnod_fisherz": "no (node only)", "cdnod_rcit": "no (node only)",
        "fennec_jci": "no (node only)"}


def jl(name):
    return [r for r in map(json.loads, (RES / name).read_text().splitlines()) if "error" not in r]


def auroc(y, s):
    y = np.asarray(y, bool)
    return roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan


def presence_scores(P, Pn):
    """Node scores from per-group directed edge frequencies (K, d, d): the largest between-group change among
    incoming edges that are confidently present in at least one group (P >= 0.5), as in the edge-level analyses;
    the caller calls a node if its score exceeds the 95th percentile of null-group node scores pooled over
    all systems of the same cell (one system has too few nodes for a stable percentile)."""
    def node(P):
        P = np.asarray(P)
        H = np.where(P.max(0) >= 0.5, P.max(0) - P.min(0), 0.0)
        return H.max(0)
    return node(P), node(Pn)


def pvalue_scores(block, d):
    p = np.ones(d)
    for (i, j), pv in zip(block["edges"], block["p"]):
        p[j] = min(p[j], pv)
    return -np.log10(np.maximum(p, 1e-300)), p < 0.05


def main():
    base = {}
    for r in jl("hetero_baselines.jsonl"):
        k = (r["setting"], r.get("scenario", ""), r["mechanism"], r["n"], r["seed"])
        base[k] = r
    systems, nulls, seen = [], {}, set()
    for r in jl("islands.jsonl"):
        k = ("islands", "", r["mechanism"], r["n_env"], r["seed"])
        if k not in base or k in seen:  # islands.jsonl has re-run duplicates of some systems; keep the first
            continue
        seen.add(k)
        b = base[k]; d = len(r["truth"][0])
        y = np.zeros(d, bool); y[b["changing"]] = True
        s_, sn = presence_scores(r["arms"]["island_m2"]["P"], r["arms"]["null_m2"]["P"])
        cell = ("islands", r["mechanism"], r["n_env"])
        nulls.setdefault(cell, []).extend(sn)
        sc = {"fennec_islands": (s_, cell), "score_gain": pvalue_scores(r["score_gain"], d)}
        sc.update({m: (np.array(v["score"]), np.array(v["called"])) for m, v in b["methods"].items()})
        systems.append(("islands", "", r["mechanism"], y, np.zeros(d, bool), sc, b))
    for r in jl("genetic_clades.jsonl"):
        k = ("genetic", r["scenario"], r["mechanism"], r["n"], r["seed"])
        if k not in base:
            continue
        b = base[k]; d = len(r["adj"])
        y = np.zeros(d, bool); y[b["changing"]] = True
        main_ = np.zeros(d, bool); main_[b["main"]] = True
        main_ |= (np.asarray(r["adj"]).sum(0) == 0) & ~y  # roots: ancestry shifts their mean (also tracks the PGS)
        leaf = lambda t: [t[x]["P"] for x in sorted(t) if len(x) == 2]
        s_, sn = presence_scores(leaf(r["trees"]["pathway"]), leaf(r["trees"]["random"]))
        cell = ("genetic", r["scenario"], r["mechanism"], r["n"])
        nulls.setdefault(cell, []).extend(sn)
        sc = {"fennec_clades": (s_, cell), "interaction": pvalue_scores(r["interaction"], d)}
        sc.update({m: (np.array(v["score"]), np.array(v["called"])) for m, v in b["methods"].items()})
        systems.append(("genetic", r["scenario"], r["mechanism"], y, main_, sc, b))
    thr = {c: np.quantile(v, 0.95) for c, v in nulls.items()}
    rows = []
    for setting, scen, mech, y, main_, sc, b in systems:
        neg = ~y & ~main_
        for m, (s_, c) in sc.items():
            if isinstance(c, tuple):  # FENNEC presence score: call against the pooled null threshold of its cell
                c = s_ > thr[c]
            row = {"setting": setting, "scenario": scen, "mechanism": mech, "method": m,
                   "AUROC": auroc(y[~main_], s_[~main_]), "TPR": c[y].mean() if y.any() else np.nan, "FPR": c[neg].mean()}
            if setting == "islands":
                row["FPR, shuffled context"] = np.mean(b["null_methods"][m]["called"]) if m in b["null_methods"] else np.nan
            else:
                row["mean-shift-only nodes flagged"] = c[main_].mean() if main_.any() else np.nan
            rows.append(row)
    d = pd.DataFrame(rows)
    print(f"{len(d)} rows; systems per setting:", d.groupby("setting").size().to_dict())
    pd.set_option("display.width", 220)
    isl = d[d.setting == "islands"].groupby("method")[["AUROC", "TPR", "FPR", "FPR, shuffled context"]].mean()
    isl = isl.reindex([m for m in LAB if m in isl.index]); isl["edge-level answer"] = [EDGE[m] for m in isl.index]
    isl.index = [LAB[m] for m in isl.index]
    print("\nT17a islands (4 environments)\n", isl.round(3))
    table(isl, "T17a", "Which variables change across environments: FENNEC islands vs CD-NOD and JCI (80 systems)",
          "Node level. FPR, shuffled context: the same method with environment labels permuted (all calls false).")
    g = d[d.setting == "genetic"]
    gx = g[g.scenario == "gxe"].groupby("method")[["AUROC", "TPR", "FPR", "mean-shift-only nodes flagged"]].mean()
    gx["FPR, null scenario"] = g[g.scenario == "null"].groupby("method").FPR.mean()
    gx = gx.reindex([m for m in LAB if m in gx.index]); gx["edge-level answer"] = [EDGE[m] for m in gx.index]
    gx.index = [LAB[m] for m in gx.index]
    print("\nT17b genetic context\n", gx.round(3))
    table(gx, "T17b", "Which variables have genotype-dependent mechanisms: FENNEC clades vs CD-NOD and JCI (96 cohorts)",
          "Node level. Mean-shift-only nodes: a genetic main effect or ancestry shifts their mean; no edge changes.")
    print("\nby mechanism (TPR / FPR)\n", d.groupby(["setting", "mechanism", "method"])[["TPR", "FPR"]].mean().unstack("method").round(2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, (setting, t, ms) in zip(axes, (("islands", isl, ["fennec_islands", "score_gain", "fennec_jci", "cdnod_rcit", "cdnod_fisherz"]),
                                          ("genetic", gx, ["fennec_clades", "interaction", "fennec_jci", "cdnod_rcit", "cdnod_fisherz"]))):
        ms = [m for m in ms if LAB[m] in t.index]
        x = np.arange(len(ms))
        ax.bar(x - 0.2, [t.loc[LAB[m], "TPR"] for m in ms], 0.38, color=[COL[m] for m in ms], label="changing variables found (TPR)")
        ax.bar(x + 0.2, [t.loc[LAB[m], "FPR"] for m in ms], 0.38, color=[COL[m] for m in ms], alpha=0.35, hatch="//",
               edgecolor="white", label="unchanged variables flagged (FPR)")
        if setting == "genetic":
            ax.plot(x + 0.2, [t.loc[LAB[m], "mean-shift-only nodes flagged"] for m in ms], "D", color=RED, ms=7,
                    label="flagged where genetics/ancestry only shifts the mean")
        for xi, m in zip(x, ms):
            ax.text(xi, -0.1, f"AUROC {t.loc[LAB[m], 'AUROC']:.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x, [textwrap.fill(LAB[m], 16) for m in ms], fontsize=8.5)
        ax.set_ylim(-0.14, 1); ax.axhline(0, color=INK2, lw=0.6); ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title("4 environments (islands systems)" if setting == "islands" else "continuous genetic context (G x E cohorts)", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.suptitle("Which variables' mechanisms change? FENNEC heterogeneity tools vs CD-NOD and JCI",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "fig17_hetero_vs_cdnod.png")
    from experiments.common import MD, TABLES
    (TABLES / "TABLES_hetero_baselines.md").write_text("# Heterogeneity baselines tables\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
