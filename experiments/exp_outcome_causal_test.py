"""Is graph-guided adaptive prediction CAUSAL, or just feature selection?

The adaptive v2 predictor (exp_outcome_adaptive.py) drops children of Y whose mechanism changed.
A graph-free model could do the same by detecting which features changed and dropping them. The
graph should matter only where DIRECTION matters: if a PARENT of Y changes its own mechanism
(how it is generated from its causes), Y still responds to it the same way, so it must be KEPT;
if a CHILD of Y changes its mechanism, it no longer reflects Y and must be DROPPED. Every
mechanism break below permutes the variable, so its marginal distribution is unchanged and only
its relationships move.

Systems: exp_outcome (12 causal + 12 noise variables), 4 mechanisms, n {500, 2000}, seeds 0-9.
Y = a variable with children AND a parent that has its own parents (so a parent can break).
Test regimes (5000 rows):
    iid               same distribution
    child_break       Y's children permuted: no longer depend on Y               -> drop them
    parent_mech_break Y's non-root parents permuted: cut from their own causes,
                      Y's mechanism intact                                       -> keep them
    other_break       up to 2 non-root variables outside Y's true blanket permuted -> irrelevant
    parent_shift      Y's parents +1.5 SD (covariate shift, mechanisms intact)
    both              child_break + parent_mech_break
Predictors (XGBoost):
    all               all 23 other features
    mb_learned        learned Markov blanket (FENNEC best graph)
    causal_v2         learned blanket minus CHILDREN whose residual variance (given their other
                      parents + Y's parents) rises > 1.5x  [graph + direction]
    v2_drop_any       same test applied to EVERY blanket member (parents given their own learned
                      parents), drop any flagged one   [graph, direction ignored in the decision]
    gf_marginal       graph-free: drop features whose marginal changed (KS, Bonferroni 0.01)
    gf_conditional    graph-free: predict each feature from all other features; drop features
                      whose residual variance rises > 1.5x (same rule as causal_v2)
    oracle_switch     learned blanket minus exactly the truly broken children
    pa_learned, ensemble (references)
Results: results/outcome_causal_test.jsonl
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from experiments.exp_effects import OUT
from experiments.exp_outcome import D_CAUSAL, N_TEST, XGB, TrueSystem, adaptive_blanket, blanket, graph_features, make_data
from experiments.exp_outcome_adaptive import VR_MIN, shifted_children_v2

RESULTS = OUT / "outcome_causal_test.jsonl"


def _resid_ratio(Xtr, Xte, target, inputs):
    """var(test residual) / var(train out-of-fold residual) of `target` given `inputs`."""
    if inputs:
        oof = np.zeros(len(Xtr))
        for tr, va in KFold(5, shuffle=True, random_state=0).split(Xtr):
            oof[va] = XGBRegressor(**XGB).fit(Xtr[tr][:, inputs], Xtr[tr][:, target]).predict(Xtr[va][:, inputs])
        r_tr = Xtr[:, target] - oof
        r_te = Xte[:, target] - XGBRegressor(**XGB).fit(Xtr[:, inputs], Xtr[:, target]).predict(Xte[:, inputs])
    else:
        r_tr, r_te = Xtr[:, target] - Xtr[:, target].mean(), Xte[:, target] - Xtr[:, target].mean()
    return r_te.var() / max(r_tr.var(), 1e-12)


def drop_any_v2(Xtr, Xte, adj, y):
    """The v2 variance-ratio test applied to every learned blanket member, ignoring direction."""
    pa_y = set(np.flatnonzero(adj[:, y]))
    out = []
    for v in blanket(adj, y):
        inputs = sorted((set(np.flatnonzero(adj[:, v])) | pa_y) - {y, v})
        if _resid_ratio(Xtr, Xte, v, inputs) > VR_MIN:
            out.append(int(v))
    return out


class GraphFreeConditional:
    """Each feature predicted from all other features (Y excluded: unknown at test time).
    Train out-of-fold residual variances and full-data models are fitted once and reused."""

    def __init__(self, Xtr, feats):
        self.feats, self.models, self.var_tr = feats, {}, {}
        for j in feats:
            inp = [f for f in feats if f != j]
            oof = np.zeros(len(Xtr))
            for tr, va in KFold(5, shuffle=True, random_state=0).split(Xtr):
                oof[va] = XGBRegressor(**XGB).fit(Xtr[tr][:, inp], Xtr[tr][:, j]).predict(Xtr[va][:, inp])
            self.var_tr[j] = max((Xtr[:, j] - oof).var(), 1e-12)
            self.models[j] = (inp, XGBRegressor(**XGB).fit(Xtr[:, inp], Xtr[:, j]))

    def flags(self, Xte):
        out = []
        for j, (inp, m) in self.models.items():
            if (Xte[:, j] - m.predict(Xte[:, inp])).var() / self.var_tr[j] > VR_MIN:
                out.append(int(j))
        return out


def gf_marginal_flags(Xtr, Xte, feats, alpha=0.01):
    return [int(j) for j in feats if ks_2samp(Xtr[:, j], Xte[:, j]).pvalue < alpha / len(feats)]


def pick_outcome(sysm):
    G, adj = sysm.G, sysm.adj

    def key(j):
        nonroot_par = [p for p in np.flatnonzero(adj[:, j]) if adj[:, p].any()]
        return (G.out_degree(j) > 0 and len(nonroot_par) > 0, G.in_degree(j) > 0 and G.out_degree(j) > 0,
                G.in_degree(j) + G.out_degree(j))
    return max(range(D_CAUSAL), key=key)


def run(mech, n, seed):
    from fennec_search import FennecSearch
    t0 = time.time()
    sysm = TrueSystem(D_CAUSAL, mech, seed)
    adj_t = sysm.adj.astype(int)
    y = pick_outcome(sysm)
    rng = np.random.default_rng(seed + 50)
    Xtr = make_data(sysm, n, seed + 1, rng)
    feats = [j for j in range(Xtr.shape[1]) if j != y]
    chi = [int(c) for c in np.flatnonzero(adj_t[y])]
    par = [int(p) for p in np.flatnonzero(adj_t[:, y])]
    par_nr = [p for p in par if adj_t[:, p].any()]
    mb_t = set(blanket(adj_t, y))
    other = [int(j) for j in range(D_CAUSAL) if j != y and j not in mb_t and adj_t[:, j].any()]
    other = list(rng.choice(other, size=min(2, len(other)), replace=False)) if other else []
    base = sysm.sample(N_TEST, seed=seed + 3)
    perm = lambda nodes: {int(v): rng.permutation(base[:, v]) for v in nodes}
    brk_c, brk_p = perm(chi), perm(par_nr)
    regimes = {"iid": ({}, []), "child_break": (brk_c, chi), "parent_mech_break": (brk_p, []),
               "other_break": (perm(other), []),
               "parent_shift": ({p: base[:, p] + 1.5 * (p not in sysm.discrete) for p in par}, []),
               "both": ({**brk_c, **brk_p}, chi)}
    if not par_nr:
        regimes.pop("parent_mech_break"); regimes.pop("both")
    if not other:
        regimes.pop("other_break")
    tests = {k: make_data(sysm, N_TEST, seed + 4 + i, rng, do=do) for i, (k, (do, _)) in enumerate(regimes.items())}
    broken_children = {k: b for k, (_, b) in regimes.items()}

    df = pd.DataFrame(Xtr, columns=[f"X{i}" for i in range(Xtr.shape[1])])
    fs = FennecSearch(n_jobs=1, seed=seed, me_generations=25, me_init=200, me_batch=40, max_graphs=30).fit(df)
    A = fs.best_graph_.astype(int)
    mb_l, pa_l = blanket(A, y), [int(p) for p in np.flatnonzero(A[:, y])]
    ytr = Xtr[:, y]
    gfc = GraphFreeConditional(Xtr, feats)

    def xgb_on(cols, Xte, extra=None):
        cols = list(cols)
        if not cols and extra is None:
            return np.full(len(Xte), ytr.mean())
        Ftr, Fte = Xtr[:, cols], Xte[:, cols]
        if extra is not None:
            Ftr, Fte = np.hstack([Ftr, extra[0]]), np.hstack([Fte, extra[1]])
        return XGBRegressor(**XGB).fit(Ftr, ytr).predict(Fte)

    rows, flags = [], {}
    for tname, Xte in tests.items():
        yte = Xte[:, y]
        f = {"causal_v2": shifted_children_v2(Xtr, Xte, A, y), "v2_drop_any": drop_any_v2(Xtr, Xte, A, y),
             "gf_marginal": gf_marginal_flags(Xtr, Xte, feats), "gf_conditional": gfc.flags(Xte)}
        flags[tname] = f
        Gtr, Gte = graph_features(Xtr, Xte, A, y)
        preds = {"all": xgb_on(feats, Xte), "mb_learned": xgb_on(mb_l, Xte), "pa_learned": xgb_on(pa_l, Xte),
                 "causal_v2": xgb_on(adaptive_blanket(A, y, f["causal_v2"]), Xte),
                 "v2_drop_any": xgb_on([v for v in mb_l if v not in f["v2_drop_any"]], Xte),
                 "gf_marginal": xgb_on([j for j in feats if j not in f["gf_marginal"]], Xte),
                 "gf_conditional": xgb_on([j for j in feats if j not in f["gf_conditional"]], Xte),
                 "oracle_switch": xgb_on(adaptive_blanket(A, y, [c for c in broken_children[tname] if A[y, c]]), Xte)}
        preds["ensemble"] = (preds["all"] + xgb_on(mb_l, Xte, (Gtr, Gte))) / 2
        for k, p in preds.items():
            rows.append({"test": tname, "predictor": k, "r2": float(r2_score(yte, p))})
    return {"mechanism": mech, "n": n, "seed": seed, "outcome": int(y), "seconds": time.time() - t0,
            "children_true": chi, "parents_true": par, "parents_nonroot": par_nr, "other_broken": [int(o) for o in other],
            "blanket_true": sorted(int(v) for v in mb_t), "blanket_learned": [int(v) for v in mb_l],
            "children_learned": [int(c) for c in np.flatnonzero(A[y])], "parents_learned": pa_l,
            "flags": flags, "graph": A.tolist(), "rows": rows}


def safe(*a):
    try:
        return run(*a)
    except Exception:  # noqa: BLE001
        return {"mechanism": a[0], "n": a[1], "seed": a[2], "error": traceback.format_exc()[-1500:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    done = set()
    if RESULTS.exists():
        for l in RESULTS.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add((r["mechanism"], r["n"], r["seed"]))
    grid = [g for g in product(["linear", "additive", "mlp", "mixed"], [2000, 500], range(args.seeds))]
    grid = [g for i, g in enumerate(grid) if i % args.nshards == args.shard and g not in done]
    print(f"{len(grid)} systems", flush=True)
    for r in Parallel(n_jobs=args.jobs, backend="loky", return_as="generator_unordered")(delayed(safe)(*g) for g in grid):
        with open(RESULTS, "a") as fh:  # one line per write: safe with several shards appending
            fh.write(json.dumps(r) + "\n")
        print(r["mechanism"], r["n"], r["seed"], "ERROR" if "error" in r else f"{r['seconds']:.0f}s", flush=True)
    print("done", flush=True)
