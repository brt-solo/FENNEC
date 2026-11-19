"""Adaptive causal outcome prediction, v2: an effect-size test for broken child mechanisms.

In exp_outcome.py the adaptive predictor flagged a child of Y as "mechanism changed" with a KS
test on its residuals. That fired falsely (a) under a pure PARENT shift, where XGBoost
extrapolates and the residual MEAN moves, and (b) for discrete children with large test sets,
where KS detects negligible differences. Measured residual-variance ratios (test / train):
broken child 2.3-5.7, unchanged mechanism 0.95-1.16. A broken child no longer carries Y's
contribution, so its residual variance grows by exactly that contribution. v2 therefore flags a
child iff  var(test residuals) / var(train out-of-fold residuals) > VR_MIN = 1.5.

Systems: as exp_outcome (12 causal + 12 noise variables), 4 mechanisms, n {500, 2000}, seeds 0-4.
Test sets: iid, child_break, parent_shift, both (parents shifted AND children broken).
Predictors: all, mb_learned, ensemble, ensemble_ctrl (all + a differently seeded/subsampled
            all-features XGBoost: the averaging-only control), adaptive_v1 (old KS test), adaptive_v2 (variance-ratio test),
            pa_learned, oracle_switch (drops exactly the truly broken children: upper bound).
The learned graph is saved. Results: results/outcome_adaptive.jsonl
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from experiments.exp_effects import OUT
from experiments.exp_outcome import (D_CAUSAL, N_TEST, XGB, TrueSystem, adaptive_blanket, blanket, graph_features, make_data,
                         shifted_children)

RESULTS = OUT / "outcome_adaptive.jsonl"
VR_MIN = 1.5


def shifted_children_v2(Xtr, Xte, adj, y, vr_min=VR_MIN):
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
            r_tr, r_te = Xtr[:, c] - Xtr[:, c].mean(), Xte[:, c] - Xtr[:, c].mean()
        if r_te.var() / max(r_tr.var(), 1e-12) > vr_min:
            out.append(int(c))
    return out


def run(mech, n, seed):
    from fennec_search import FennecSearch
    t0 = time.time()
    sysm = TrueSystem(D_CAUSAL, mech, seed)
    G = sysm.G
    y = max(range(D_CAUSAL), key=lambda j: (G.in_degree(j) > 0 and G.out_degree(j) > 0, G.in_degree(j) + G.out_degree(j)))
    rng = np.random.default_rng(seed + 50)
    Xtr = make_data(sysm, n, seed + 1, rng)
    feats = [j for j in range(Xtr.shape[1]) if j != y]
    chi = list(np.flatnonzero(sysm.adj[y, :])); par = list(np.flatnonzero(sysm.adj[:, y]))
    base = sysm.sample(N_TEST, seed=seed + 3)
    brk = {c: rng.permutation(base[:, c]) for c in chi}
    shf = {p: base[:, p] + 1.5 * (p not in sysm.discrete) for p in par}
    tests = {"iid": make_data(sysm, N_TEST, seed + 2, rng),
             "child_break": make_data(sysm, N_TEST, seed + 4, rng, do=brk),
             "parent_shift": make_data(sysm, N_TEST, seed + 5, rng, do=shf),
             "both": make_data(sysm, N_TEST, seed + 6, rng, do={**shf, **brk})}
    broken = {"iid": [], "child_break": chi, "parent_shift": [], "both": chi}
    df = pd.DataFrame(Xtr, columns=[f"X{i}" for i in range(Xtr.shape[1])])
    fs = FennecSearch(n_jobs=1, seed=seed, me_generations=25, me_init=200, me_batch=40, max_graphs=30).fit(df)
    A = fs.best_graph_.astype(int)
    mb_l, pa_l = blanket(A, y), list(np.flatnonzero(A[:, y]))
    ytr = Xtr[:, y]

    def xgb_on(cols, Xte, extra=None):
        if not cols and extra is None:
            return np.full(len(Xte), ytr.mean())
        Ftr, Fte = Xtr[:, cols], Xte[:, cols]
        if extra is not None:
            Ftr, Fte = np.hstack([Ftr, extra[0]]), np.hstack([Fte, extra[1]])
        return XGBRegressor(**XGB).fit(Ftr, ytr).predict(Fte)

    rows, flags = [], {}
    for tname, Xte in tests.items():
        yte = Xte[:, y]
        f1, f2 = shifted_children(Xtr, Xte, A, y), shifted_children_v2(Xtr, Xte, A, y)
        flags[tname] = {"v1": f1, "v2": f2}
        Gtr, Gte = graph_features(Xtr, Xte, A, y)
        preds = {"all": xgb_on(feats, Xte), "mb_learned": xgb_on(mb_l, Xte), "pa_learned": xgb_on(pa_l, Xte),
                 "adaptive_v1": xgb_on(adaptive_blanket(A, y, f1), Xte),
                 "adaptive_v2": xgb_on(adaptive_blanket(A, y, f2), Xte),
                 "oracle_switch": xgb_on(adaptive_blanket(A, y, [c for c in broken[tname] if A[y, c]]), Xte)}
        preds["ensemble"] = (preds["all"] + xgb_on(mb_l, Xte, (Gtr, Gte))) / 2
        # control: is the ensemble gain from the graph, or just from averaging two models?
        alt = XGBRegressor(**{**XGB, "random_state": 1}, colsample_bynode=0.7, subsample=0.8).fit(Xtr[:, feats], ytr)
        preds["ensemble_ctrl"] = (preds["all"] + alt.predict(Xte[:, feats])) / 2
        for k, p in preds.items():
            rows.append({"test": tname, "predictor": k, "r2": float(r2_score(yte, p))})
    return {"mechanism": mech, "n": n, "seed": seed, "outcome": int(y), "seconds": time.time() - t0,
            "children_true": [int(c) for c in chi], "children_learned": [int(c) for c in np.flatnonzero(A[y])],
            "flags": flags, "graph": A.tolist(), "rows": rows}


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
    grid = [g for g in product(["linear", "additive", "mlp", "mixed"], [2000, 500], range(5)) if g not in done]
    print(f"{len(grid)} systems", flush=True)
    with open(RESULTS, "a") as fh:
        for r in Parallel(n_jobs=24, backend="loky", return_as="generator_unordered")(delayed(safe)(*g) for g in grid):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["mechanism"], r["n"], r["seed"], "ERROR" if "error" in r else f"{r['seconds']:.0f}s", flush=True)
    print("done", flush=True)
