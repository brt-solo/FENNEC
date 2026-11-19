"""Analysis of exp_outcome_causal_test.py: causal adaptive prediction vs graph-free feature dropping.
Prints tables; writes results/figures/fig10_causal_test.png and T11a-c (CSV)."""
import json

import numpy as np
import pandas as pd

from experiments.common import AQUA, BLUE, GRID, INK2, MUTED, ORANGE, OUT, RED, boot_ci, plt, save_fig, table
from experiments.exp_effects import OUT as RES

REG = ["iid", "child_break", "parent_mech_break", "other_break", "parent_shift", "both"]
RLAB = {"iid": "same distribution", "child_break": "children broken\n(should drop)",
        "parent_mech_break": "parents' mechanism broken\n(should keep)", "other_break": "non-blanket broken\n(irrelevant)",
        "parent_shift": "parents shifted\n(covariate shift)", "both": "children + parents\nbroken"}
PRED = ["mb_learned", "ensemble", "causal_v2", "v2_drop_any", "gf_conditional", "gf_marginal", "oracle_switch", "pa_learned"]
PLAB = {"all": "XGBoost, all features", "mb_learned": "learned Markov blanket", "ensemble": "graph-informed ensemble",
        "causal_v2": "causal adaptive (drop changed CHILDREN)", "v2_drop_any": "graph, no direction (drop any changed)",
        "gf_conditional": "graph-free: drop features whose conditional changed",
        "gf_marginal": "graph-free: drop features whose marginal changed", "oracle_switch": "oracle (drop true broken children)",
        "pa_learned": "parents only"}


def main():
    recs = [json.loads(l) for l in (RES / "outcome_causal_test.jsonl").read_text().splitlines()]
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    print(f"{len(recs)} systems, {len(errs)} errors")
    d = pd.DataFrame([{"mechanism": r["mechanism"], "n": r["n"], "seed": r["seed"], **x} for r in recs for x in r["rows"]])
    t = d.groupby(["predictor", "test"]).r2.mean().unstack().reindex(columns=REG).reindex(["all"] + PRED)
    t.index = [PLAB[i] for i in t.index]
    t.columns = [RLAB[c].replace("\n", " ") for c in t.columns]
    print(t.round(3)); table(t, "T11a", "Causal test: test R^2 by regime (80 systems)")

    key = ["mechanism", "n", "seed"]
    rows = []
    for reg in REG:
        s = d[d.test == reg].set_index(key + ["predictor"]).r2.unstack()
        for m in PRED:
            mu, lo, hi = boot_ci((s[m] - s["all"]).dropna())
            rows.append({"regime": RLAB[reg].replace("\n", " "), "predictor": PLAB[m], "gain vs all": mu, "lo": lo, "hi": hi})
        for a, b in [("causal_v2", "gf_conditional"), ("causal_v2", "v2_drop_any"), ("causal_v2", "gf_marginal")]:
            mu, lo, hi = boot_ci((s[a] - s[b]).dropna())
            rows.append({"regime": RLAB[reg].replace("\n", " "), "predictor": f"{PLAB[a]} MINUS {PLAB[b]}",
                         "gain vs all": mu, "lo": lo, "hi": hi})
    g = pd.DataFrame(rows).set_index(["regime", "predictor"])
    print(g.round(3)); table(g, "T11b", "Paired R^2 differences (95% CI over systems)")

    # what each detector flags, per regime
    fr = []
    for r in recs:
        ch, pa, oth = set(r["children_true"]), set(r["parents_true"]), set(r["other_broken"])
        for reg, fl in r["flags"].items():
            for det, f in fl.items():
                f = set(f)
                fr.append({"regime": reg, "detector": det, "true children": len(f & ch), "true parents": len(f & pa),
                           "broken non-blanket": len(f & oth), "other features": len(f - ch - pa - oth)})
    fr = pd.DataFrame(fr).groupby(["regime", "detector"]).mean().reindex(REG, level=0)
    print(fr.round(2)); table(fr, "T11c", "Features flagged as changed, mean per system")

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1.8, 1]})
    ax = axes[0]
    show = ["causal_v2", "v2_drop_any", "gf_conditional", "gf_marginal", "oracle_switch"]
    col = {"causal_v2": BLUE, "v2_drop_any": AQUA, "gf_conditional": ORANGE, "gf_marginal": "#f5b394", "oracle_switch": INK2}
    xs, w = np.arange(len(REG)), 0.16
    for k, m in enumerate(show):
        sub = g.xs(PLAB[m], level="predictor").reindex([RLAB[r].replace("\n", " ") for r in REG])
        v = sub["gain vs all"].to_numpy()
        ax.bar(xs + (k - 2) * w, np.maximum(v, -0.6), w * 0.92, color=col[m], label=PLAB[m],
               yerr=np.where(v > -0.6, [v - sub.lo, sub.hi - v], 0), error_kw=dict(lw=0.7, ecolor=INK2))
        for i, vi in enumerate(v):
            if vi < -0.6:  # clipped bar: print its value
                ax.text(xs[i] + (k - 2) * w, -0.585, f"{vi:.2f}", ha="center", va="bottom", fontsize=7, rotation=90, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-0.6, 0.45)
    ax.set_xticks(xs, [RLAB[r].replace("parents' mechanism", "parents'\nmechanism").replace("non-blanket broken", "non-blanket\nbroken")
                       for r in REG], fontsize=8)
    ax.set_ylabel("R² gain over XGBoost on all features")
    ax.set_title("Which features to drop when something changes?"); ax.legend(frameon=False, fontsize=8, loc="upper center",
                                                                                            bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax = axes[1]
    sub = fr.xs("parent_mech_break", level="regime")
    order = ["causal_v2", "v2_drop_any", "gf_conditional", "gf_marginal"]
    sub = sub.reindex(order)
    left = np.zeros(len(sub))
    for c, cc in [("true parents", RED), ("true children", BLUE), ("broken non-blanket", MUTED), ("other features", "#cde2fb")]:
        ax.barh(np.arange(len(sub)), sub[c], left=left, color=cc, label=c); left += sub[c].to_numpy()
    ax.set_yticks(np.arange(len(sub)), [PLAB[i].replace(": ", ":\n") for i in order], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("features flagged per system"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("When parents' mechanisms break: what gets flagged"); ax.grid(axis="x", color=GRID, lw=0.6)
    fig.suptitle("Is graph-guided adaptation causal, or just feature selection? (80 systems)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "fig10_causal_test.png")
    from experiments.common import MD, TABLES
    md = TABLES / "TABLES_causal_test.md"
    md.write_text("# Causal test tables\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
