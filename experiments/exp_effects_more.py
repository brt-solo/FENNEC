"""More baselines for causal-effect accuracy, on the same 96 systems as exp_effects.py.

Re-creates each system, its data, targets and true effects exactly as exp_effects.run_system does,
then adds (FENNEC, CAM, GES, the true graph and the empty graph are already in results/effects.jsonl):

Graph sources (same DoWhy-GCM model per graph, as in exp_effects):
    bidag           BiDAG partition-MCMC posterior (BGe score), 20 posterior samples, equal weights
    pc              PC (CPDAG extended to a DAG)
    notears         NOTEARS-MLP (nonlinear continuous optimisation)
    lingam          DirectLiNGAM
Graph-free effect estimators for each (target T, outcome Y), given an adjustment set Z:
    dml_*           double ML partially linear model (partialling-out score, as DoubleMLPLR; implemented
                    here because doubleml 0.10 is incompatible with scikit-learn 1.8), 5-fold cross-fitted XGBoost;
                    effect = theta x (a - b)  (assumes the effect is linear in T)
    gcomp_*         regression adjustment / g-computation: XGBoost of Y on (T, Z), averaged
                    prediction at T = a minus at T = b  (allows nonlinear dose-response)
    adjustment sets:  *_truepa   true parents of T (needs perfect prior knowledge of T's causes)
                      *_fennecpa FENNEC's learned parents of T (graph used only for adjustment)
                      *_all      every other variable (no graph; adjusts for mediators too)
    If Y is itself in Z, the effect is 0 by construction (Y is a cause of T, not an effect).
Results: results/effects_more.jsonl (rows in the same format as effects.jsonl).
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import time
import traceback
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.exp_effects import N_TARGETS, OUT
from experiments.truth import TrueSystem

RESULTS = OUT / "effects_more.jsonl"
GCOMP_RESULTS = OUT / "effects_more_gcomp.jsonl"  # g-computation rows recomputed with the fixed estimator
N_BIDAG = 20


def xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, n_jobs=1, verbosity=0)


def dml_effect(X, t, y, Z, a, b, seed):
    from sklearn.model_selection import KFold
    if y in Z:
        return 0.0
    if not Z:
        T, Y = X[:, t], X[:, y]
        return float(np.cov(T, Y)[0, 1] / np.var(T, ddof=1) * (a - b))
    # partialling-out score of DoubleMLPLR: residualise Y and T on Z out of fold, regress residual on residual
    W, T, Y = X[:, Z], X[:, t], X[:, y]
    rt, ry = np.zeros(len(X)), np.zeros(len(X))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(W):
        rt[te] = T[te] - xgb().fit(W[tr], T[tr]).predict(W[te])
        ry[te] = Y[te] - xgb().fit(W[tr], Y[tr]).predict(W[te])
    return float((rt @ ry) / (rt @ rt) * (a - b))


def gcomp_effect(X, t, y, Z, a, b, discrete=False):
    """S-learner g-computation. A regularised XGBoost and a +-0.25 SD window around a and b keep the
    contrast from reading noise off one overfit curve at two points (the first version did; see GCOMP_RESULTS)."""
    from xgboost import XGBRegressor
    if y in Z:
        return 0.0
    F = X[:, [t] + list(Z)]
    f = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=30, reg_lambda=5,
                     subsample=0.8, n_jobs=1, verbosity=0).fit(F, X[:, y])
    offs = [0.0] if discrete else np.linspace(-0.25, 0.25, 5)

    def mean_at(v):
        out = []
        for o in offs:
            G = F.copy()
            G[:, 0] = v + o
            out.append(f.predict(G).mean())
        return np.mean(out)
    return float(mean_at(a) - mean_at(b))


def run_system(mech, d, n, seed, gcomp_only=False):
    from benchmarks.methods import bidag_samples, lingam, notears_mlp, pc
    from fennec_search import FennecSearch, GraphEnsemble, infer_types, to_dag
    t0 = time.time()
    sysm = TrueSystem(d, mech, seed)
    cols = [f"X{i}" for i in range(d)]
    df = pd.DataFrame(sysm.sample(n, seed=seed + 1), columns=cols)
    X = df.to_numpy()
    types = infer_types(df)
    rng = np.random.default_rng(seed + 5)  # identical target choice to exp_effects.run_system
    cand = [j for j in range(d) if nx.descendants(sysm.G, j)]
    disc = [j for j in cand if j in sysm.discrete]
    targets = list(rng.choice(cand, size=min(N_TARGETS, len(cand)), replace=False))
    if disc and not any(t in sysm.discrete for t in targets):
        targets[-1] = int(rng.choice(disc))
    targets = [int(t) for t in targets]
    queries = [(cols[t], *sysm.levels(t)) for t in targets]
    truth = {t: sysm.true_effects(t, *sysm.levels(t)) for t in targets}
    info, graphs = {}, {}
    for name, f in [] if gcomp_only else [("pc", pc), ("notears", notears_mlp), ("lingam", lingam)]:
        t1 = time.time()
        try:
            graphs[name] = [to_dag(f(X))]
        except Exception as e:  # noqa: BLE001
            info[f"{name}_error"] = repr(e)[:200]
        info[f"{name}_seconds"] = time.time() - t1
    t1 = time.time()
    try:
        if gcomp_only:
            raise RuntimeError("skipped")
        graphs["bidag"] = bidag_samples(X, n_samples=N_BIDAG)
    except Exception as e:  # noqa: BLE001
        info["bidag_error"] = repr(e)[:200]
    info["bidag_seconds"] = time.time() - t1
    t1 = time.time()
    fs = FennecSearch(n_jobs=1, seed=seed, max_graphs=30).fit(df)
    fpa = {t: [int(p) for p in np.flatnonzero(fs.best_graph_[:, t])] for t in targets}
    info["fennec_seconds"] = time.time() - t1
    rows = []

    def add(method, t, y, est, lo=np.nan, hi=np.nan):
        rows.append({"method": method, "target": cols[t], "outcome": cols[y], "target_discrete": t in sysm.discrete,
                     "outcome_discrete": y in sysm.discrete, "true": float(truth[t][y]),
                     "is_descendant": y in nx.descendants(sysm.G, t), "est": float(est), "lo": lo, "hi": hi})

    for method, adjs in graphs.items():
        ens = GraphEnsemble(adjs, [0.0] * len(adjs), n_obs=n, weighting="uniform")
        r = ens.query(df, queries, types=types, n=3000, n_jobs=1, seed=seed)
        for _, q in r.iterrows():
            add(method, int(q.target[1:]), int(q.outcome[1:]), q["mean"], q.lo, q.hi)
    for t, (_, a, b) in zip(targets, queries):
        sets = {"truepa": [int(p) for p in np.flatnonzero(sysm.adj[:, t])], "fennecpa": fpa[t]}
        for y in range(d):
            if y == t:
                continue
            sets["all"] = [j for j in range(d) if j not in (t, y)]
            for sname, Z in sets.items():
                if not gcomp_only:
                    add(f"dml_{sname}", t, y, dml_effect(X, t, y, Z, a, b, seed))
                add(f"gcomp_{sname}", t, y, gcomp_effect(X, t, y, Z, a, b, t in sysm.discrete))
    return {"mechanism": mech, "d": d, "n": n, "seed": seed, "seconds": time.time() - t0, **info, "rows": rows}


def safe_run(*args):
    try:
        return run_system(*args)
    except Exception:  # noqa: BLE001
        return dict(zip(("mechanism", "d", "n", "seed"), args), error=traceback.format_exc()[-1500:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gcomp-only", action="store_true", help="recompute only the g-computation rows")
    args = ap.parse_args()
    if args.smoke:
        r = run_system("additive", 10, 500, 0)
        print("error" in r and r["error"] or {k: v for k, v in r.items() if k != "rows"}, len(r.get("rows", [])))
        raise SystemExit
    # the 96 systems of results/effects.jsonl
    done_eff = [json.loads(l) for l in (OUT / "effects.jsonl").read_text().splitlines()]
    grid = sorted({(r["mechanism"], r["d"], r["n"], r["seed"]) for r in done_eff if "error" not in r})
    out_file = GCOMP_RESULTS if args.gcomp_only else RESULTS
    done = set()
    if out_file.exists():
        for l in out_file.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add((r["mechanism"], r["d"], r["n"], r["seed"]))
    grid.sort(key=lambda g: -(g[1] ** 2 * g[2]))
    grid = [g for i, g in enumerate(grid) if i % args.nshards == args.shard and g not in done]
    print(f"{len(grid)} systems", flush=True)
    for r in Parallel(n_jobs=args.jobs, backend="loky", return_as="generator_unordered")(delayed(safe_run)(*g, args.gcomp_only) for g in grid):
        with open(out_file, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(r["mechanism"], r["d"], r["n"], r["seed"], "ERROR" if "error" in r else f"{r['seconds']:.0f}s", flush=True)
    print("done", flush=True)
