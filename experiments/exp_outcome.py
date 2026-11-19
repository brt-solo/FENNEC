"""Can the WHOLE learned causal structure make an outcome predictor better than plain XGBoost?

Systems: truth.TrueSystem with 12 causal variables + 12 irrelevant noise variables (pure noise,
so feature selection matters), 4 mechanisms, n_train in {100, 250, 500, 2000}, seeds 0-9.
Outcome Y = the causal variable with the most parents + children (has both, so the graph matters).
Test sets (5000 rows each):
    iid           same distribution as training
    child_break   Y's children are set to independent draws from their own marginals: the
                  children no longer carry information about Y (a mechanism shift, e.g. a
                  measurement or treatment change downstream of the outcome)
    parent_shift  Y's parents shifted by +1.5 SD (covariate shift; Y's own mechanism unchanged)
Predictors (all XGBoost, default settings, unless noted):
    all            all 23 other variables (the standard ML baseline)
    lasso          LassoCV on all variables (linear baseline)
    mb_learned     the learned Markov blanket of Y (FENNEC best graph)
    pa_learned     learned parents of Y only (causally invariant, ignores children)
    weighted       all variables, XGBoost feature sampling weighted by the FENNEC population's
                   probability that each variable is in Y's Markov blanket (colsample_bynode 0.5)
    graph_feats    all variables + graph-derived features: cross-fitted prediction of Y from its
                   learned parents, and each learned child's residual given its OTHER parents
                   (the part of a child explained only by Y)
    mb_graph       learned Markov blanket + the same graph-derived features
    ensemble       average of `all` and `mb_graph`
    adaptive       causal test-time adaptation: for each learned child c of Y, a model predicts c
                   from the inputs observable at test time that stand in for its mechanism (its
                   other parents + Y's parents); if the distribution of its residuals on the
                   unlabeled test batch differs from training (KS test, Bonferroni p < 0.01), c's
                   mechanism has changed and c (with co-parents that only enter through c) is
                   dropped; XGBoost is refit on the remaining blanket. Uses no test labels.
    mb_true        oracle: XGBoost on the true Markov blanket
Metric: test R^2 (and MSE). Results: results/outcome.jsonl
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import traceback
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from experiments.exp_effects import OUT
from experiments.truth import TrueSystem

RESULTS = OUT / "outcome.jsonl"
D_CAUSAL, D_NOISE, N_TEST = 12, 12, 5000
XGB = dict(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0, verbosity=0, n_jobs=1)


def blanket(adj, y):
    pa = set(np.flatnonzero(adj[:, y])); ch = set(np.flatnonzero(adj[y, :]))
    co = {int(p) for c in ch for p in np.flatnonzero(adj[:, c])}
    return sorted((pa | ch | co) - {y})


def make_data(sysm, n, seed, rng_noise, do=None):
    Xc = sysm.sample(n, seed=seed, do=do)
    return np.hstack([Xc, rng_noise.normal(size=(n, D_NOISE))])


def shifted_children(Xtr, Xte, adj, y, alpha=0.01):
    """Children of y whose mechanism appears changed on the unlabeled test batch."""
    from scipy.stats import ks_2samp
    ch = list(np.flatnonzero(adj[y, :]))
    pa_y = list(np.flatnonzero(adj[:, y]))
    out = []
    for c in ch:
        inputs = sorted(set(p for p in np.flatnonzero(adj[:, c]) if p != y) | set(pa_y))
        if inputs:
            oof = np.zeros(len(Xtr))
            for tr, va in KFold(5, shuffle=True, random_state=0).split(Xtr):
                oof[va] = XGBRegressor(**XGB).fit(Xtr[tr][:, inputs], Xtr[tr][:, c]).predict(Xtr[va][:, inputs])
            r_tr = Xtr[:, c] - oof
            r_te = Xte[:, c] - XGBRegressor(**XGB).fit(Xtr[:, inputs], Xtr[:, c]).predict(Xte[:, inputs])
        else:
            r_tr, r_te = Xtr[:, c], Xte[:, c]
        if ks_2samp(r_tr, r_te).pvalue < alpha / max(len(ch), 1):
            out.append(int(c))
    return out


def adaptive_blanket(adj, y, drop):
    """Markov blanket of y after removing shifted children (and co-parents reached only through them)."""
    pa = set(np.flatnonzero(adj[:, y]))
    ch = set(np.flatnonzero(adj[y, :])) - set(drop)
    co = {int(p) for c in ch for p in np.flatnonzero(adj[:, c])}
    return sorted((pa | ch | co) - {y})


def graph_features(Xtr, Xte, adj, y, folds=5):
    """Cross-fitted (train) / full-fit (test) features: yhat from parents; child residuals."""
    pa = list(np.flatnonzero(adj[:, y]))
    ch = list(np.flatnonzero(adj[y, :]))
    specs = []
    if pa:
        specs.append(("yhat_pa", y, pa))
    for c in ch:
        other = [p for p in np.flatnonzero(adj[:, c]) if p != y]
        specs.append((f"res_{c}", c, other))
    Ftr = np.zeros((len(Xtr), len(specs))); Fte = np.zeros((len(Xte), len(specs)))
    for k, (name, tgt, inputs) in enumerate(specs):
        def fit_pred(A, B):
            if not inputs:
                return np.full(len(B), A[:, tgt].mean())
            return XGBRegressor(**XGB).fit(A[:, inputs], A[:, tgt]).predict(B[:, inputs])
        oof = np.zeros(len(Xtr))
        for tr, va in KFold(folds, shuffle=True, random_state=0).split(Xtr):
            oof[va] = fit_pred(Xtr[tr], Xtr[va])
        full = fit_pred(Xtr, Xte)
        if name == "yhat_pa":
            Ftr[:, k], Fte[:, k] = oof, full
        else:
            Ftr[:, k], Fte[:, k] = Xtr[:, tgt] - oof, Xte[:, tgt] - full
    return Ftr, Fte


def run(mech, n, seed):
    from fennec_search import FennecSearch
    t0 = time.time()
    sysm = TrueSystem(D_CAUSAL, mech, seed)
    G = sysm.G
    y = max(range(D_CAUSAL), key=lambda j: (G.in_degree(j) > 0 and G.out_degree(j) > 0, G.in_degree(j) + G.out_degree(j)))
    rng = np.random.default_rng(seed + 50)
    Xtr = make_data(sysm, n, seed + 1, rng)
    feats = [j for j in range(Xtr.shape[1]) if j != y]
    tests = {"iid": make_data(sysm, N_TEST, seed + 2, rng)}
    chi = list(np.flatnonzero(sysm.adj[y, :])); par = list(np.flatnonzero(sysm.adj[:, y]))
    base = sysm.sample(N_TEST, seed=seed + 3)
    tests["child_break"] = make_data(sysm, N_TEST, seed + 4, rng,
                                     do={c: rng.permutation(base[:, c]) for c in chi})
    tests["parent_shift"] = make_data(sysm, N_TEST, seed + 5, rng,
                                      do={p: base[:, p] + 1.5 * (1.0 if p not in sysm.discrete else 0.0) for p in par}) \
        if par else tests["iid"]
    df = pd.DataFrame(Xtr, columns=[f"X{i}" for i in range(Xtr.shape[1])])
    fs = FennecSearch(n_jobs=1, seed=seed, me_generations=25, me_init=200, me_batch=40, max_graphs=30).fit(df)
    A = fs.best_graph_.astype(int)
    mb_l, pa_l = blanket(A, y), list(np.flatnonzero(A[:, y]))
    pop = [g for g, _ in fs.population_]
    w = np.array([np.mean([j in blanket(g, y) for g in pop]) for j in feats]) + 0.05
    mb_t = blanket(sysm.adj.astype(int), y)
    ytr = Xtr[:, y]
    rows = []
    gf = {name: graph_features(Xtr, Xte, A, y) for name, Xte in tests.items()}

    def xgb_on(cols, extra_tr=None, extra_te=None, test=None, weights=None):
        Ftr = Xtr[:, cols] if cols else np.zeros((len(Xtr), 0))
        Fte = test[:, cols] if cols else np.zeros((len(test), 0))
        if extra_tr is not None:
            Ftr, Fte = np.hstack([Ftr, extra_tr]), np.hstack([Fte, extra_te])
        if Ftr.shape[1] == 0:
            return np.full(len(test), ytr.mean())
        if weights is not None:
            m = XGBRegressor(**XGB, colsample_bynode=0.5).fit(Ftr, ytr, feature_weights=weights)
        else:
            m = XGBRegressor(**XGB).fit(Ftr, ytr)
        return m.predict(Fte)

    adapt_log = {}
    for tname, Xte in tests.items():
        yte = Xte[:, y]
        Gtr, Gte = gf[tname]
        preds = {
            "all": xgb_on(feats, test=Xte),
            "lasso": LassoCV(cv=5, random_state=0).fit(Xtr[:, feats], ytr).predict(Xte[:, feats]),
            "mb_learned": xgb_on(mb_l, test=Xte),
            "pa_learned": xgb_on(pa_l, test=Xte),
            "weighted": xgb_on(feats, test=Xte, weights=w),
            "graph_feats": xgb_on(feats, Gtr, Gte, test=Xte),
            "mb_graph": xgb_on(mb_l, Gtr, Gte, test=Xte),
            "mb_true": xgb_on(mb_t, test=Xte),
        }
        preds["ensemble"] = (preds["all"] + preds["mb_graph"]) / 2
        drop = shifted_children(Xtr, Xte, A, y)
        preds["adaptive"] = xgb_on(adaptive_blanket(A, y, drop), test=Xte)
        adapt_log[tname] = drop
        for name, p in preds.items():
            rows.append({"test": tname, "predictor": name, "r2": float(r2_score(yte, p)),
                         "mse": float(np.mean((yte - p) ** 2))})
    true_mb = set(mb_t)
    return {"mechanism": mech, "n": n, "seed": seed, "outcome": int(y), "seconds": time.time() - t0,
            "mb_true": [int(v) for v in mb_t], "mb_learned": [int(v) for v in mb_l],
            "mb_precision": len(set(mb_l) & true_mb) / max(len(mb_l), 1), "mb_recall": len(set(mb_l) & true_mb) / max(len(true_mb), 1),
            "noise_in_mb": int(sum(v >= D_CAUSAL for v in mb_l)), "pop_size": len(pop),
            "children_true": [int(c) for c in chi], "adaptive_dropped": adapt_log, "rows": rows}


def safe(*a):
    try:
        return run(*a)
    except Exception:  # noqa: BLE001
        return {"mechanism": a[0], "n": a[1], "seed": a[2], "error": traceback.format_exc()[-1500:]}


if __name__ == "__main__":
    done = set()
    if RESULTS.exists():
        for l in RESULTS.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add((r["mechanism"], r["n"], r["seed"]))
    grid = [g for g in product(["linear", "additive", "mlp", "mixed"], [2000, 500, 250, 100], range(10)) if g not in done]
    print(f"{len(grid)} systems", flush=True)
    with open(RESULTS, "a") as fh:
        for r in Parallel(n_jobs=24, backend="loky", return_as="generator_unordered")(delayed(safe)(*g) for g in grid):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["mechanism"], r["n"], r["seed"], "ERROR" if "error" in r else f"{r['seconds']:.0f}s", flush=True)
    print("done", flush=True)
