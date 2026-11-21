"""Analysis of E4a/E4b (e4ab_inmodel.py) and E4c (e4c_missing.py): the graph used inside the model.

    python -m experiments.analyze_e4
Writes results/tables/E4a_*.csv, E4b_*.csv, E4c_*.csv, TABLES_e4.md and three figures:
results/figures/fig_e4a_latent_factors.png, fig_e4b_distance_penalties.png, fig_e4c_missing_block.png.
Colours: categorical slots of the reference palette in fixed order; grey = baselines; red = oracle (unattainable).
"""
import json

import numpy as np
import pandas as pd

from experiments.common import GRID, INK, INK2, MD, MUTED, TABLES, boot_ci, plt, save_fig, table
from experiments.e4ab_inmodel import RAW

S1, S2, S3, S4, S5, S7, S8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"
GREY, LGREY = "#8a8984", "#c3c2b7"


def ci_text(x):
    mu, lo, hi = boot_ci(np.asarray(x, float))
    return f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def load(prefix):
    return [json.loads(l) for p in sorted(RAW.glob(f"{prefix}_shard*.jsonl")) for l in p.read_text().splitlines()]


def bar_ci(ax, x, vals_list, color, width, label=None, hatch=None):
    m = [np.mean(v) for v in vals_list]
    ci = [boot_ci(np.asarray(v, float))[1:] for v in vals_list]
    err = np.array([[mi - lo for mi, (lo, hi) in zip(m, ci)], [hi - mi for mi, (lo, hi) in zip(m, ci)]])
    ax.bar(x, m, width, color=color, label=label, edgecolor="white", linewidth=1.5, hatch=hatch, zorder=2)
    ax.errorbar(x, m, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=3)
    return m


# ============================================================================ E4a
def e4a(recs):
    K = ["p", "n", "latent_y_weight", "seed"]
    d = pd.DataFrame([{**{k: r[k] for k in K}, **x} for r in recs for x in r["rows"] if x["exp"] == "E4a"])
    designs = ["causal", "causal+selfac", "causal+factors", "all", "all+factors", "oracle", "oracle+latent"]
    lab = {"causal": "causal set C\n(FENNEC, f>=0.3)", "causal+selfac": "C + factors\nFENNEC adds",
           "causal+factors": "C + all 10\nfactors", "all": "all features", "all+factors": "all features\n+ factors",
           "oracle": "true causes", "oracle+latent": "true causes +\ntrue latents"}
    # table: by proxy strength, predictor, design
    rows = []
    for (w, pr, des), g in d.groupby(["latent_y_weight", "predictor", "design"]):
        base = d[(d.latent_y_weight == w) & (d.predictor == pr) & (d.design == "causal")].set_index(K)
        allb = d[(d.latent_y_weight == w) & (d.predictor == pr) & (d.design == "all")].set_index(K)
        g = g.set_index(K)
        rows.append({"proxy strength": "weak (0.25)" if w < 0.5 else "real (1.0)", "model": pr, "design": des,
                     "R2 same": g.r2_iid.mean(), "R2 proxies break": g.r2_proxy_break.mean(),
                     "R2 causes shift": g.r2_cause_shift.mean(),
                     "vs causal set (same)": ci_text(g.r2_iid - base.r2_iid),
                     "vs all features (same)": ci_text(g.r2_iid - allb.r2_iid),
                     "drop when proxies break": ci_text(g.r2_proxy_break - g.r2_iid)})
    t = pd.DataFrame(rows)
    t["o"] = t.design.map({k: i for i, k in enumerate(designs)})
    t = t.sort_values(["proxy strength", "model", "o"]).drop(columns="o").set_index(["proxy strength", "model", "design"])
    print(t.round(3).to_string())
    table(t, "E4a_a", "E4a: latent factors. Test R^2 and paired differences (95% CI over systems)")
    fr = pd.DataFrame([{"p": r["p"], "n": r["n"], "proxy strength": r["latent_y_weight"],
                        "recovery latent 1": r["factor_recovery"][0], "recovery latent 2": r["factor_recovery"][1],
                        "factors FENNEC added": len(r["selected_factors"])} for r in recs])
    f = fr.groupby(["proxy strength", "p", "n"]).mean()
    print(f.round(3).to_string())
    table(f, "E4a_b", "E4a: how well 10 principal components recover Y's two latent causes (R^2), and how many factors FENNEC adds")

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.25, 1.25, 1])
    for col, w in enumerate([0.25, 1.0]):
        ax = fig.add_subplot(gs[0, col])
        sub = d[(d.latent_y_weight == w) & (d.predictor == "enet")]
        x = np.arange(len(designs))
        cols = [S1, S3, S3, GREY, GREY, S8, S8]
        hat = [None, None, "///", None, "///", None, "///"]
        for i, des in enumerate(designs):
            s = sub[sub.design == des]
            bar_ci(ax, [x[i] - 0.19], [s.r2_iid], cols[i], 0.36, hatch=hat[i])
            bar_ci(ax, [x[i] + 0.19], [s.r2_proxy_break], cols[i], 0.36, hatch=hat[i])
            ax.bar([x[i] + 0.19], [s.r2_proxy_break.mean()], 0.36, color="white", alpha=0.45, zorder=2.5)
        ax.set_xticks(x, [lab[k] for k in designs], fontsize=7.5, rotation=35, ha="right", rotation_mode="anchor")
        ax.axhline(sub[sub.design == "all"].r2_iid.mean(), color=INK2, lw=0.8, ls="--", zorder=1)
        ax.set_ylabel("test R² (elastic net)")
        ax.set_title(f"{'A' if col == 0 else 'B'}. {'Weak' if w < 0.5 else 'Real'} latent causes of Y "
                     f"(latent_y_weight = {w:g})")
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.text(0.01, 0.97, "left bar = same distribution · right (faded) = proxies break\n"
                "dashed line = all features, same distribution", transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
    ax = fig.add_subplot(gs[0, 2])
    for w, c, lab_ in [(0.25, LGREY, "weak latent causes"), (1.0, S3, "real latent causes")]:
        rr = [r for r in recs if r["latent_y_weight"] == w]
        rec_ = [np.mean(r["factor_recovery"]) for r in rr]
        gain = []
        for r in rr:
            rw = {x["design"]: x for x in r["rows"] if x["exp"] == "E4a" and x["predictor"] == "enet"}
            gain.append(rw["causal+factors"]["r2_iid"] - rw["causal"]["r2_iid"])
        ax.scatter(rec_, gain, s=22, color=c, label=lab_, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("how well the factors recover Y's latent causes (mean R²)")
    ax.set_ylabel("R² gain: C + factors minus C (same distribution)")
    ax.set_title("C. Factors help only when they carry the latent causes")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    # bottom: robustness trade-off, scatter of gain(same) vs drop(proxies break) per design
    ax = fig.add_subplot(gs[1, :2])
    pts = []
    for w, mk in [(0.25, "o"), (1.0, "s")]:
        for des, c in [("causal", S1), ("causal+selfac", S3), ("causal+factors", S4), ("all", GREY), ("all+factors", INK2),
                       ("oracle", S8), ("oracle+latent", S5)]:
            s = d[(d.latent_y_weight == w) & (d.predictor == "enet") & (d.design == des)]
            ax.scatter(s.r2_iid.mean(), s.r2_proxy_break.mean(), s=90, marker=mk, color=c, edgecolor="white",
                       linewidth=1.2, zorder=3, label=f"{lab[des].replace(chr(10), ' ')}" if w == 1.0 else None)
    lim = [0.05, 0.3]
    ax.plot(lim, lim, color=LGREY, lw=1, ls="--", zorder=1)
    ax.text(0.27, 0.272, "no loss under shift", fontsize=7.5, color=MUTED, rotation=30)
    ax.legend(frameon=False, fontsize=8, loc="upper left", title="design (colour)", title_fontsize=8)
    ax.text(0.99, 0.02, "circles = weak latent causes, squares = real latent causes", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("test R², same distribution")
    ax.set_ylabel("test R², proxies break")
    ax.set_title("D. Accuracy vs robustness (circles = weak latent causes, squares = real; elastic net). "
                 "Below the diagonal = loses accuracy when proxies break")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax = fig.add_subplot(gs[1, 2])
    sub = d[(d.predictor == "xgb_reg") & d.design.isin(["causal", "causal+selfac", "causal+factors", "all", "oracle"])]
    piv = sub.groupby(["design", "latent_y_weight"]).r2_iid.mean().unstack()
    piv = piv.reindex(["causal", "causal+selfac", "causal+factors", "all", "oracle"])
    y_ = np.arange(len(piv))
    ax.barh(y_ - 0.18, piv[0.25], 0.36, color=LGREY, label="weak latent causes", edgecolor="white")
    ax.barh(y_ + 0.18, piv[1.0], 0.36, color=S3, label="real latent causes", edgecolor="white")
    ax.set_yticks(y_, [lab[k].replace("\n", " ") for k in piv.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("test R², same distribution")
    ax.set_title("E. Same designs with regularized XGBoost")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    fig.suptitle(f"E4a. Latent factors from proxy clusters ({len(recs)} systems: p = 200/1000, n = 400/800, 10 seeds each)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "fig_e4a_latent_factors.png")


# ============================================================================ E4b
def e4b(recs):
    K = ["p", "n", "latent_y_weight", "seed"]
    d = pd.DataFrame([{**{k: r[k] for k in K}, **x} for r in recs for x in r["rows"] if x["exp"] == "E4b"])
    schemes = ["uniform", "adaptive", "dist_soft", "dist_strong", "hard_select_C", "oracle_dist"]
    lab = {"uniform": "uniform\n(plain EN)", "adaptive": "adaptive\n(frequency)", "dist_soft": "distance,\nsoft",
           "dist_strong": "distance,\nstrong", "hard_select_C": "hard select C\n(E2b best)", "oracle_dist": "distance, strong\nTRUE graph"}
    col = {"uniform": GREY, "adaptive": S2, "dist_soft": S1, "dist_strong": S7, "hard_select_C": S3, "oracle_dist": S8}
    rows = []
    for (pr, sch), g in d.groupby(["predictor", "design"]):
        base = d[(d.predictor == pr) & (d.design == "uniform")].set_index(K)
        g = g.set_index(K)
        row = {"model": pr, "scheme": sch, "R2 same": g.r2_iid.mean(), "R2 proxies break": g.r2_proxy_break.mean(),
               "R2 causes shift": g.r2_cause_shift.mean(), "vs uniform (same)": ci_text(g.r2_iid - base.r2_iid),
               "vs uniform (proxies break)": ci_text(g.r2_proxy_break - base.r2_proxy_break),
               "share of systems better (same)": float((g.r2_iid - base.r2_iid > 0).mean())}
        for c in ["nz_true_parent", "nz_true_grandparent", "nz_proxy", "nz_other"]:
            if c in g and g[c].notna().any():
                row[c] = g[c].mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    t["o"] = t.scheme.map({k: i for i, k in enumerate(schemes)})
    t = t.sort_values(["model", "o"]).drop(columns="o").set_index(["model", "scheme"])
    print(t.round(3).to_string())
    table(t, "E4b_a", "E4b: graph-distance penalties. Test R^2, paired differences vs plain elastic net / plain regularized XGBoost, nonzero coefficients by true class")
    for pr in ("enet", "xgb_reg"):
        s = d[d.predictor == pr].pivot_table(index=["p", "n", "latent_y_weight"], columns="design", values="r2_iid")
        s = s.sub(s["uniform"], axis=0).drop(columns="uniform").reindex(columns=[k for k in schemes if k != "uniform"])
        print(f"\n{pr}: gain vs uniform by setting\n", s.round(3).to_string())
        table(s, f"E4b_{'b' if pr == 'enet' else 'c'}", f"E4b: {pr} gain over the uniform version by p, n and proxy strength")
    gp = pd.DataFrame([{"p": r["p"], "n": r["n"], "|C|": r["n_C"], "C true causes": r["C_true"],
                        "|distance-2 set|": r["n_gp"], "of which true grandparents": r["gp_true_gp"],
                        "of which true parents (missed by C)": r["gp_true_parent"], "of which proxies": r["gp_proxy"],
                        "true grandparents exist": r["n_true_gp"]} for r in recs]).groupby(["p", "n"]).mean()
    print(gp.round(2).to_string())
    table(gp, "E4b_d", "E4b: what the learned distance classes contain")

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.3, 1])
    for k, (pr, title) in enumerate([("enet", "A. Elastic net"), ("xgb_reg", "B. Regularized XGBoost (feature_weights)")]):
        ax = fig.add_subplot(gs[0, k])
        ps = sorted(d.p.unique())
        x = np.arange(len(ps))
        w = 0.13
        for i, sch in enumerate(schemes):
            vals = [d[(d.predictor == pr) & (d.design == sch) & (d.p == p)].r2_iid for p in ps]
            bar_ci(ax, x + (i - 2.5) * w, vals, col[sch], w * 0.95, label=lab[sch].replace("\n", " "))
        ax.set_xticks(x, [f"p = {p} features" for p in ps])
        ax.set_ylabel("test R², same distribution")
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        if k == 0:
            ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    ax = fig.add_subplot(gs[0, 2])
    s = d[(d.predictor == "enet")].groupby("design")[["nz_true_parent", "nz_true_grandparent", "nz_proxy", "nz_other"]].mean()
    s = s.reindex([k for k in schemes if k != "hard_select_C"])
    left = np.zeros(len(s))
    for c, cc, ll in [("nz_true_parent", S8, "true causes"), ("nz_true_grandparent", S4, "causes of causes"),
                      ("nz_proxy", S5, "proxies"), ("nz_other", LGREY, "unrelated")]:
        ax.barh(np.arange(len(s)), s[c], left=left, color=cc, label=ll, edgecolor="white", linewidth=1.5)
        left += s[c].to_numpy()
    ax.set_yticks(np.arange(len(s)), [lab[k].replace("\n", " ") for k in s.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("features with a nonzero coefficient (mean per system)")
    ax.set_title("C. What the elastic net actually uses")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax = fig.add_subplot(gs[1, :2])
    s = d[d.predictor == "enet"].pivot_table(index=["p", "n"], columns="design", values="r2_iid")
    s = s.sub(s["uniform"], axis=0)
    labels = [f"p={p}\nn={n}" for p, n in s.index]
    x = np.arange(len(s))
    for i, sch in enumerate(["adaptive", "dist_soft", "dist_strong", "hard_select_C", "oracle_dist"]):
        ax.plot(x, s[sch], marker="o", ms=6, lw=2, color=col[sch], label=lab[sch].replace("\n", " "))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("R² gain over plain elastic net")
    ax.set_title("D. Where each scheme helps (elastic net, same distribution, both proxy strengths pooled)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax = fig.add_subplot(gs[1, 2])
    gp2 = pd.DataFrame([{"p": r["p"], "true grandparent": r["gp_true_gp"], "missed true cause": r["gp_true_parent"],
                         "proxy": r["gp_proxy"], "other": r["n_gp"] - r["gp_true_gp"] - r["gp_true_parent"] - r["gp_proxy"]}
                        for r in recs]).groupby("p").mean()
    left = np.zeros(len(gp2))
    for c, cc in [("missed true cause", S8), ("true grandparent", S4), ("proxy", S5), ("other", LGREY)]:
        ax.barh(np.arange(len(gp2)), gp2[c], left=left, color=cc, label=c, edgecolor="white", linewidth=1.5)
        left += gp2[c].to_numpy()
    ax.set_yticks(np.arange(len(gp2)), [f"p = {p}" for p in gp2.index])
    ax.set_xlabel("features in the learned distance-2 class (mean)")
    ax.set_title("E. What 'causes of the causes' FENNEC found")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    fig.suptitle(f"E4b. Graph-distance penalties: every feature stays in, penalised by its distance from Y in the learned graph "
                 f"({len(recs)} systems)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "fig_e4b_distance_penalties.png")


# ============================================================================ E4c
def e4c(recs):
    K = ["p", "n", "mechanism", "seed"]
    d = pd.DataFrame([{**{k: r[k] for k in K}, **x} for r in recs for x in r["rows"]])
    methods = ["xgb_nan", "meanind_enet", "pattern_enet", "pattern_xgb_reg", "mice_enet", "mice_C_enet", "scm_enet",
               "scm_xgb_reg", "oracle_scm_enet", "full_all_enet", "full_C_enet"]
    lab = {"xgb_nan": "XGBoost, NaN for missing", "meanind_enet": "mean impute + indicator (EN)",
           "pattern_enet": "pattern submodels (EN)", "pattern_xgb_reg": "pattern submodels (XGB)",
           "mice_enet": "MICE, all features (EN)", "mice_C_enet": "MICE, causal set (EN)",
           "scm_enet": "SCM imputation, causal set (EN)", "scm_xgb_reg": "SCM imputation, causal set (XGB)",
           "oracle_scm_enet": "SCM imputation, TRUE graph (EN)", "full_all_enet": "no missing data: all (EN)",
           "full_C_enet": "no missing data: causal set (EN)"}
    col = {"xgb_nan": S2, "meanind_enet": S4, "pattern_enet": S5, "pattern_xgb_reg": S5, "mice_enet": S7, "mice_C_enet": S7,
           "scm_enet": S1, "scm_xgb_reg": S1, "oracle_scm_enet": S8, "full_all_enet": GREY, "full_C_enet": GREY}
    ls = {"pattern_xgb_reg": ":", "mice_C_enet": ":", "scm_xgb_reg": ":", "full_C_enet": ":"}
    regs = ["same", "mcar", "none", "full"]
    rlab = {"same": "same sampling\nas training", "mcar": "30% at random", "none": "nobody has\nthe block",
            "full": "everybody has\nthe block"}
    rows = []
    for (mech, m), g in d.groupby(["mechanism", "method"]):
        base = d[(d.mechanism == mech) & (d.method == "mice_enet")].set_index(K)
        g = g.set_index(K)
        row = {"training sampling": mech, "method": m}
        for r_ in regs:
            row[f"R2 {r_}"] = g[f"r2_{r_}"].mean()
        row["worst-case R2"] = g[[f"r2_{r_}" for r_ in regs]].min(axis=1).mean()
        row["R2 range across regimes"] = (g[[f"r2_{r_}" for r_ in regs]].max(axis=1) - g[[f"r2_{r_}" for r_ in regs]].min(axis=1)).mean()
        row["vs MICE (30% at random)"] = ci_text(g.r2_mcar - base.r2_mcar)
        row["vs MICE (nobody has block)"] = ci_text(g.r2_none - base.r2_none)
        rows.append(row)
    t = pd.DataFrame(rows)
    t["o"] = t.method.map({k: i for i, k in enumerate(methods)})
    t = t.sort_values(["training sampling", "o"]).drop(columns="o").set_index(["training sampling", "method"])
    print(t.round(3).to_string())
    table(t, "E4c_a", "E4c: missing sub-study block. Test R^2 by deployment regime; paired differences vs MICE (95% CI)")
    imp = pd.DataFrame([{"p": r["p"], "n": r["n"], "mechanism": r["mechanism"], **v} for r in recs for v in r["imputation"].values()])
    ti = imp.groupby(["mechanism", "true_cause"])[["scm", "mice", "n_parents"]].mean() if len(imp) else pd.DataFrame()
    print(ti.round(3).to_string())
    table(ti, "E4c_b", "E4c: imputation accuracy (correlation with the true value) for block members of C")
    info = pd.DataFrame([{"mechanism": r["mechanism"], "p": r["p"], "n": r["n"], "rows with block (train)": r["n_obs_train"],
                          "mean y of those rows": r["y_mean_observed"], "|C|": r["n_C"], "C in block": r["n_C_block"],
                          "true causes in block": r["n_true_block_causes"]} for r in recs]).groupby(["mechanism", "p", "n"]).mean()
    print(info.round(2).to_string())
    table(info, "E4c_c", "E4c: training data and causal set")

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1])
    for k, mech in enumerate(["case_enriched", "mcar"]):
        ax = fig.add_subplot(gs[0, k])
        x = np.arange(len(regs))
        for m in methods:
            s = d[(d.mechanism == mech) & (d.method == m)]
            if not len(s):
                continue
            ax.plot(x, [s[f"r2_{r_}"].mean() for r_ in regs], marker="o", ms=6, lw=2, color=col[m], ls=ls.get(m, "-"),
                    label=lab[m])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, [rlab[r_] for r_ in regs], fontsize=8)
        ax.set_ylim(max(-0.6, ax.get_ylim()[0]), None)
        ax.set_ylabel("test R² (all test rows)")
        ax.set_title(("A. Training block case-enriched\n(low-outcome rows more likely to have it)" if mech == "case_enriched"
                      else "B. Training block missing completely at random"), fontsize=9.5)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel("who has the block at deployment")
        if k == 1:
            ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=1)
    ax = fig.add_subplot(gs[0, 2])
    if len(imp):
        for tc, c, ll in [(True, S1, "true causes of Y"), (False, LGREY, "other members of C")]:
            s = imp[imp.true_cause == tc]
            ax.scatter(s.mice, s.scm, s=20, color=c, edgecolor="white", linewidth=0.5, label=ll, zorder=3)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("MICE imputation accuracy (correlation with truth)")
    ax.set_ylabel("SCM imputation accuracy (correlation with truth)")
    ax.set_title("C. Filling in block causes: graph parents vs MICE")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax = fig.add_subplot(gs[1, :2])
    s = d[d.mechanism == "case_enriched"].groupby("method")[[f"r2_{r_}" for r_ in regs]].mean().reindex(methods).dropna()
    worst = s.min(axis=1)
    rng_ = s.max(axis=1) - s.min(axis=1)
    y_ = np.arange(len(s))
    ax.barh(y_ - 0.18, s["r2_same"], 0.36, color=LGREY, label="R² when deployment sampling = training sampling",
            edgecolor="white")
    ax.barh(y_ + 0.18, worst.clip(lower=-0.5), 0.36, color=[col[m] for m in s.index], label="worst R² over the 4 deployment regimes",
            edgecolor="white")
    for i, (v, w_) in enumerate(zip(worst, s["r2_same"])):
        ax.text(max(v, -0.5) - 0.01 if v < 0 else v + 0.005, i + 0.18, f"{v:.2f}", va="center",
                ha="right" if v < 0 else "left", fontsize=7.5, color=INK2)
    ax.set_yticks(y_, [lab[m] for m in s.index], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("test R²")
    ax.set_title("D. Case-enriched training: looks good on the same sampling, but how bad can it get? (clipped at -0.5)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax = fig.add_subplot(gs[1, 2])
    s = d[d.method.isin(["scm_enet", "mice_enet", "mice_C_enet", "pattern_enet", "full_C_enet"])]
    piv = s.groupby(["method", "n"]).r2_mcar.mean().unstack().reindex(["full_C_enet", "scm_enet", "mice_C_enet", "mice_enet", "pattern_enet"])
    y_ = np.arange(len(piv))
    for j, (nn, c) in enumerate(zip(piv.columns, [LGREY, S1])):
        ax.barh(y_ + (j - 0.5) * 0.36, piv[nn], 0.36, color=c, label=f"n = {nn}", edgecolor="white")
    ax.set_yticks(y_, [lab[m] for m in piv.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("test R², block 30% at random at deployment")
    ax.set_title("E. Elastic-net routes by training size (both samplings)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    fig.suptitle(f"E4c. A sub-study block (microbiome tier) missing for ~70% of participants ({len(recs)} systems)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "fig_e4c_missing_block.png")


def main():
    ab, c = load("e4ab"), load("e4c")
    print(f"E4a/b: {len(ab)} systems; E4c: {len(c)} systems")
    if ab:
        e4a(ab)
        e4b(ab)
    if c:
        e4c(c)
    (TABLES / "TABLES_e4.md").write_text("# E4 tables: the graph inside the model\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
