"""Analysis of exp_islands.py. Edges are evaluated as skeleton pairs (direction ignored), since the
question is which connections are shared vs environment-specific.

    1  per-environment accuracy   skeleton F1 of each environment's graph (P >= 0.5) vs that
                                  environment's truth: does sharing information help or hurt?
    2  heterogeneity detection    among pairs real in >= 1 environment: AUROC for VARIABLE vs SHARED
                                  (islands: H = max_k P - min_k P; score_gain: Cochran's Q), and
                                  calibrated calls (threshold = 95th percentile of the random-split null)
    3  where is it present        among variable pairs: AUROC of P[k] for the truth in environment k
    4  convergence                does "evolved independently on many islands" (L, local origins)
                                  identify real or shared pairs better than presence (P)?
Writes results/figures/fig11_islands.png and T12a-d.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.exp_effects import OUT as RES
from experiments.common import AQUA, BLUE, GRID, INK2, MUTED, ORANGE, OUT, boot_ci, plt, save_fig, table

ARMS = ["pooled", "island_m0", "island_m2", "island_m8"]
ALAB = {"pooled": "pooled search", "island_m0": "islands, no migration", "island_m2": "islands, light migration",
        "island_m8": "islands, heavy migration", "score_gain": "score-gain test (no evolution)"}
MECH = ["linear", "additive", "mlp", "mixed"]
iu = np.triu_indices(10, 1)


def skel(A):
    """(K, d, d) directed -> (K, n_pairs) undirected presence."""
    A = np.asarray(A, float)
    return (A + A.transpose(0, 2, 1))[:, iu[0], iu[1]].clip(0, 1)


def auroc(y, s):
    y = np.asarray(y, bool)
    return roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan


def f1(pred, true):
    tp = (pred & true).sum(); fp = (pred & ~true).sum(); fn = (~pred & true).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def q_pairs(sg):
    """Cochran's Q and p per skeleton pair (max over directions); pairs never tested get Q = 0, p = 1."""
    Q, p = np.zeros(len(iu[0])), np.ones(len(iu[0]))
    idx = {(a, b): n for n, (a, b) in enumerate(zip(*iu))}
    for (i, j), q, pv in zip(sg["edges"], sg["Q"], sg["p"]):
        n = idx[(min(i, j), max(i, j))]
        Q[n], p[n] = max(Q[n], q), min(p[n], pv)
    return Q, p


def main():
    recs = [json.loads(l) for l in (RES / "islands.jsonl").read_text().splitlines()]
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    print(f"{len(recs)} systems, {len(errs)} errors")
    per_sys, pair_rows = [], []
    for r in recs:
        T = skel(r["truth"]).astype(bool)                      # (K, pairs)
        real, shared = T.any(0), T.all(0)
        variable = real & ~shared
        base = {"mechanism": r["mechanism"], "n_env": r["n_env"], "seed": r["seed"],
                "n_shared": int(shared.sum()), "n_variable": int(variable.sum())}
        Qr, pr = q_pairs(r["score_gain"]); Qn, pn = q_pairs(r["score_gain_null"])
        row = dict(base)
        for a in ARMS:
            P = skel(r["arms"][a]["P"])
            row[f"F1 {a}"] = np.mean([f1(P[k] >= 0.5, T[k]) for k in range(len(T))])
            H = P.max(0) - P.min(0)
            row[f"het AUROC {a}"] = auroc(variable[real], H[real])
            pv = P[:, variable]
            row[f"where AUROC {a}"] = auroc(T[:, variable].ravel(), pv.ravel()) if variable.any() else np.nan
        row["het AUROC score_gain"] = auroc(variable[real], Qr[real])
        for a, nl in (("island_m0", "null_m0"), ("island_m2", "null_m2"), ("island_m8", "null_m2")):
            P, Pn = skel(r["arms"][a]["P"]), skel(r["arms"][nl]["P"])
            Hn = Pn.max(0) - Pn.min(0)  # all pairs (see analyze_genetic_clades)
            thr = np.quantile(Hn, 0.95) if len(Hn) else 0.5
            H = P.max(0) - P.min(0)
            row[f"TPR {a}"] = (H[variable] > thr).mean() if variable.any() else np.nan
            row[f"FPR {a}"] = (H[shared] > thr).mean() if shared.any() else np.nan
        row["TPR score_gain"] = (pr[variable] < 0.05).mean() if variable.any() else np.nan
        row["FPR score_gain"] = (pr[shared] < 0.05).mean() if shared.any() else np.nan
        tested_null = pn < 1
        row["null rejection rate score_gain"] = (pn[tested_null] < 0.05).mean() if tested_null.any() else np.nan
        for a in ("island_m0", "island_m2", "island_m8", "null_m2"):
            P, L = skel(r["arms"][a]["P"]), skel(r["arms"][a]["L"])
            cand = P.max(0) > 0.05
            row[f"real AUROC presence {a}"] = auroc(real[cand], P.mean(0)[cand])
            row[f"real AUROC local-origin {a}"] = auroc(real[cand], L.mean(0)[cand])
            row[f"real AUROC islands-converged {a}"] = auroc(real[cand], (L >= 0.5).sum(0)[cand])
            cr = cand & real
            row[f"shared AUROC presence {a}"] = auroc(shared[cr], P.mean(0)[cr])
            row[f"shared AUROC local-origin {a}"] = auroc(shared[cr], L.mean(0)[cr])
        per_sys.append(row)
    d = pd.DataFrame(per_sys)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)

    def by(cols, rename):
        t = d.groupby("n_env")[cols].mean().T
        t.index = [rename(c) for c in t.index]
        return t

    t1 = by([f"F1 {a}" for a in ARMS], lambda c: ALAB[c[3:]])
    t1m = d.groupby("mechanism")[[f"F1 {a}" for a in ARMS]].mean().T.reindex(columns=MECH)
    t1m.index = [ALAB[c[3:]] for c in t1m.index]
    t1 = pd.concat([t1, t1m], axis=1); t1.columns = [f"n/env = {c}" if isinstance(c, (int, np.integer)) else c for c in t1.columns]
    print("\n1. per-environment skeleton F1\n", t1.round(3)); table(t1, "T12a", "Islands: per-environment skeleton F1")
    print("   paired, islands-light minus no migration:", np.round(boot_ci(d["F1 island_m2"] - d["F1 island_m0"]), 3),
          " no migration minus pooled:", np.round(boot_ci(d["F1 island_m0"] - d["F1 pooled"]), 3))

    hc = [f"het AUROC {a}" for a in ARMS[1:]] + ["het AUROC score_gain"]
    t2 = by(hc, lambda c: ALAB[c.split()[-1]])
    cal = []
    for a in ("island_m0", "island_m2", "island_m8", "score_gain"):
        cal.append({"method": ALAB[a], "TPR (variable pairs called variable)": d[f"TPR {a}"].mean(),
                    "FPR (shared pairs called variable)": d[f"FPR {a}"].mean()})
    cal = pd.DataFrame(cal).set_index("method")
    cal["score-gain null rejection (should be 0.05)"] = [np.nan] * 3 + [d["null rejection rate score_gain"].mean()]
    t2 = t2.join(cal)
    print("\n2. heterogeneity detection (variable vs shared)\n", t2.round(3))
    table(t2, "T12b", "Detecting environment-specific edges (AUROC; calibrated calls vs random-split null)")

    t3 = by([f"where AUROC {a}" for a in ARMS], lambda c: ALAB[c.split()[-1]])
    print("\n3. which environment has a variable edge (AUROC)\n", t3.round(3))
    table(t3, "T12c", "Locating environment-specific edges: AUROC of P[k] for presence in environment k")

    rows = []
    for a in ("island_m0", "island_m2", "island_m8", "null_m2"):
        rows.append({"arm": ALAB.get(a, "random-split null, light migration"),
                     "real: presence": d[f"real AUROC presence {a}"].mean(),
                     "real: local origin": d[f"real AUROC local-origin {a}"].mean(),
                     "real: # islands converged": d[f"real AUROC islands-converged {a}"].mean(),
                     "shared vs variable: presence": d[f"shared AUROC presence {a}"].mean(),
                     "shared vs variable: local origin": d[f"shared AUROC local-origin {a}"].mean()})
    t4 = pd.DataFrame(rows).set_index("arm")
    print("\n4. convergence: AUROC of presence vs independent local origins\n", t4.round(3))
    table(t4, "T12d", "Convergent evolution across islands: does independent re-evolution identify real / shared edges?")
    for a in ("island_m2", "island_m8"):
        print(f"   {a}: local-origin minus presence (real):",
              np.round(boot_ci((d[f"real AUROC local-origin {a}"] - d[f"real AUROC presence {a}"]).dropna()), 3))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    col = {"pooled": MUTED, "island_m0": "#9ec5f4", "island_m2": BLUE, "island_m8": "#184f95", "score_gain": ORANGE}
    ns = sorted(d.n_env.unique())
    xs = np.arange(len(ns))
    ax = axes[0]; w = 0.2
    for k, a in enumerate(ARMS):
        ax.bar(xs + (k - 1.5) * w, [d[d.n_env == n][f"F1 {a}"].mean() for n in ns], w * 0.92, color=col[a], label=ALAB[a])
    ax.set_xticks(xs, [f"{n} rows / environment" for n in ns]); ax.set_ylim(0, 1); ax.set_ylabel("skeleton F1 per environment")
    ax.set_title("Accuracy of each environment's graph"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax = axes[1]; w = 0.2
    for k, a in enumerate(["island_m0", "island_m2", "island_m8", "score_gain"]):
        ax.bar(xs + (k - 1.5) * w, [d[d.n_env == n][f"het AUROC {a}"].mean() for n in ns], w * 0.92, color=col[a], label=ALAB[a])
    ax.axhline(0.5, color=INK2, ls="--", lw=0.8); ax.text(xs[-1] + 0.45, 0.51, "chance", fontsize=7.5, ha="right")
    ax.set_xticks(xs, [f"{n} rows / environment" for n in ns]); ax.set_ylim(0.3, 1); ax.set_ylabel("AUROC")
    ax.set_title("Telling environment-specific from shared edges"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax = axes[2]
    lab = ["presence\n(share of population)", "independent local origin", "# islands where\nit evolved locally"]
    cols_ = ["real AUROC presence", "real AUROC local-origin", "real AUROC islands-converged"]
    for k, (a, c) in enumerate([("island_m0", "#9ec5f4"), ("island_m2", BLUE), ("island_m8", "#184f95"), ("null_m2", MUTED)]):
        ax.bar(np.arange(3) + (k - 1.5) * 0.2, [d[f"{cc} {a}"].mean() for cc in cols_], 0.18, color=c,
               label=ALAB.get(a, "random-split null"))
    ax.set_xticks(np.arange(3), lab, fontsize=8); ax.set_ylim(0.5, 1); ax.set_ylabel("AUROC: real vs false edges")
    ax.set_title("Does convergent evolution identify real edges?"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle(f"Environments as islands: shared vs environment-specific causal structure ({len(d)} synthetic systems, 4 environments each)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "fig11_islands.png")
    from experiments.common import MD, TABLES
    (TABLES / "TABLES_islands.md").write_text("# Islands tables\n\n" + "\n".join(MD))


if __name__ == "__main__":
    main()
