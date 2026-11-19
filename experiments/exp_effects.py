"""Does FENNEC-Search give correct causal effects, with honest uncertainty?

For each ground-truth system (truth.TrueSystem): sample observational data, learn graphs, fit a
mixed-type causal model (fennec_search.causal_model, DoWhy-GCM) per graph, and estimate
E[Y | do(T=a)] - E[Y | do(T=b)] for 4 targets T and every other variable Y. Compare with the
true effect (Monte Carlo from the true mechanisms).

Graph sources (mechanisms always learned from the same observational data):
    true            the true graph                                  (single; upper reference)
    fennec_best     FENNEC score, hill climbing                     (single)
    fennec_pop      FENNEC near-optimal population: MAP-Elites pool within the bootstrap noise band, uniform weights
    fennec_pop_folds  same, but band from CV-split noise only (narrower)
    fennec_pop_bma  same population, BMA-style weights exp(n * score difference)
    bootstrap       FENNEC hill climbing on 10 bootstrap resamples  (standard ensemble baseline)
    cam, ges        single graphs from CAM / GES (CPDAG extended to a DAG)
    empty           no edges: every effect estimated as 0
Grid: mechanism {linear, additive, mlp, mixed} x d {10, 20} x n {500, 2000} x seeds 0-9.
Systems run in parallel (each single-threaded); results appended to results/effects.jsonl.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
from experiments.truth import TrueSystem  # noqa: E402

OUT = HERE.parent / "results" / "raw"   # raw result files (git-ignored)
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "effects.jsonl"
N_TARGETS = 4
N_BOOT = 10
MAX_POP = 30


def learn_graphs(df, seed, n_jobs=1):
    from benchmarks.methods import cam, ges
    from fennec_search import FennecSearch, hill_climb, to_dag
    from fennec_search.score import CVScore
    X = df.to_numpy()
    out, info = {}, {}
    t = time.time()
    fs = FennecSearch(n_jobs=n_jobs, seed=seed, max_graphs=MAX_POP).fit(df)
    out["fennec_best"] = [(fs.best_graph_, fs.best_score_)]
    out["fennec_pop"] = fs.population_
    out["fennec_pop_folds"] = fs.population_folds_
    info.update(fennec_seconds=time.time() - t, band=fs.band_, band_folds=fs.band_folds_, pop_size=len(fs.population_),
                pop_folds_size=len(fs.population_folds_), n_fits=fs.score_.n_fits)
    t = time.time()
    rng = np.random.default_rng(seed + 99)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(X), len(X), replace=True)
        a, s = hill_climb(CVScore(X[idx], n_jobs=n_jobs))
        boots.append((a, s))
    out["bootstrap"] = boots
    info["bootstrap_seconds"] = time.time() - t
    for name, f in [("cam", cam), ("ges", ges)]:
        t = time.time()
        try:
            out[name] = [(to_dag(f(X)), 0.0)]
        except Exception as e:  # noqa: BLE001
            info[f"{name}_error"] = repr(e)[:200]
        info[f"{name}_seconds"] = time.time() - t
    return out, info


def run_system(mech, d, n, seed):
    from fennec_search import GraphEnsemble, infer_types
    t0 = time.time()
    sysm = TrueSystem(d, mech, seed)
    cols = [f"X{i}" for i in range(d)]
    df = pd.DataFrame(sysm.sample(n, seed=seed + 1), columns=cols)
    types = infer_types(df)
    rng = np.random.default_rng(seed + 5)
    cand = [j for j in range(d) if nx.descendants(sysm.G, j)]
    disc = [j for j in cand if j in sysm.discrete]
    targets = list(rng.choice(cand, size=min(N_TARGETS, len(cand)), replace=False))
    if disc and not any(t in sysm.discrete for t in targets):  # ensure a discrete exposure in mixed systems
        targets[-1] = int(rng.choice(disc))
    targets = [int(t) for t in targets]
    queries = [(cols[t], *sysm.levels(t)) for t in targets]
    truth = {cols[t]: sysm.true_effects(t, *sysm.levels(t)) for t in targets}
    graphs, info = learn_graphs(df, seed)
    graphs["true"] = [(sysm.adj, 0.0)]
    rows = []
    for method, pop in list(graphs.items()) + [("fennec_pop_bma", graphs["fennec_pop"])]:
        adjs, scores = zip(*pop)
        weighting = "bma" if method == "fennec_pop_bma" else "uniform"
        ens = GraphEnsemble(adjs, scores, n_obs=n, weighting=weighting)
        r = ens.query(df, queries, types=types, n=3000, n_jobs=1, seed=seed)
        for _, q in r.iterrows():
            t = int(q.target[1:]); y = int(q.outcome[1:])
            rows.append({"method": method, "target": q.target, "outcome": q.outcome,
                         "target_discrete": t in sysm.discrete, "outcome_discrete": y in sysm.discrete,
                         "true": float(truth[q.target][y]), "is_descendant": y in nx.descendants(sysm.G, t),
                         "est": q["mean"], "lo": q.lo, "hi": q.hi, "p_ancestor": q.p_ancestor,
                         "n_graphs": int(q.n_graphs), "ess": float(1 / np.sum(ens.weights ** 2))})
    for t, a, b in queries:  # empty graph: every effect is 0
        y_true = truth[t]
        for y in range(d):
            if cols[y] != t:
                rows.append({"method": "empty", "target": t, "outcome": cols[y], "target_discrete": int(t[1:]) in sysm.discrete,
                             "outcome_discrete": y in sysm.discrete, "true": float(y_true[y]),
                             "is_descendant": y in nx.descendants(sysm.G, int(t[1:])), "est": 0.0, "lo": 0.0, "hi": 0.0,
                             "p_ancestor": 0.0, "n_graphs": 1, "ess": 1.0})
    return {"mechanism": mech, "d": d, "n": n, "seed": seed, "seconds": time.time() - t0, **info, "rows": rows}


def safe_run(*args):
    try:
        return run_system(*args)
    except Exception:  # noqa: BLE001
        return {"mechanism": args[0], "d": args[1], "n": args[2], "seed": args[3], "error": traceback.format_exc()[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--d", nargs="*", type=int, default=[10, 20])
    ap.add_argument("--n", nargs="*", type=int, default=[500, 2000])
    ap.add_argument("--mechanisms", nargs="*", default=["linear", "additive", "mlp", "mixed"])
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["mechanism"], r["d"], r["n"], r["seed"]))
    grid = [g for g in product(a.mechanisms, a.d, a.n, a.seeds) if g not in done]
    grid.sort(key=lambda g: -(g[1] ** 2 * g[2]))  # biggest systems first, for load balance
    print(f"{len(grid)} systems to run on {a.jobs} workers", flush=True)
    with open(RESULTS, "a") as fh:
        for r in Parallel(n_jobs=a.jobs, backend="loky", return_as="generator_unordered")(
                delayed(safe_run)(*g) for g in grid):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            tag = "ERROR" if "error" in r else f"{r['seconds']:.0f}s pop={r.get('pop_size')}"
            print(f"  {r['mechanism']} d={r['d']} n={r['n']} seed={r['seed']}: {tag}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
