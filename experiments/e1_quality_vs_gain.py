"""E1: does a better graph give a better predictor? Prediction gain vs graph quality.

For each synthetic system (benchmarks.generate) one outcome node is chosen (seeded; preferring a node
with >= 2 parents and >= 1 child). Graphs given to every compiler (fennec_search.compilers):
    oracle          the true DAG
    corrupt_<k>     the true DAG with each edge deleted or reversed with probability k, plus
                    round(k * |E|) spurious edges (kept acyclic): a continuum of graph error
    random_<r>      a random DAG with the true edge count
    empty           no edges (compilers fall back to a mean / all-feature model)
    <learner>       learned on the TRAINING rows only: PC, GES, NOTEARS-MLP, CAM, FENNEC
Each compiler is fitted on the training rows and scored on an independent test set of n_test rows
from the same system (no shift; shift is E3). Nothing is tuned: XGBoost settings are fixed and shared.

    python -m experiments.e1_quality_vs_gain                       # whole grid, all cores
    python -m experiments.e1_quality_vs_gain --shard 0 --nshards 10 --jobs 8
    python -m experiments.e1_quality_vs_gain --pilot              # 4 small systems, quick check
Resumable: systems already in the output file (results/raw/e1.jsonl, or e1_shard<k>.jsonl) are skipped.
Note: FENNEC's 4x penalty was chosen on seeds 0-4, so its learned graphs are reported on seeds 5-9.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from benchmarks.generate import simulate
from benchmarks.metrics import graph_metrics
from fennec_search.compilers import make_compiler, markov_blanket, parents
from fennec_search.graph import is_dag

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


def choose_outcome(adj, rng):
    d = len(adj)
    npa, nch = adj.sum(0), adj.sum(1)
    for cond in (lambda j: npa[j] >= 2 and nch[j] >= 1, lambda j: npa[j] >= 1 and nch[j] >= 1, lambda j: npa[j] >= 1):
        cands = [j for j in range(d) if cond(j)]
        if cands:
            return int(rng.choice(cands))
    return int(rng.integers(d))


def _add_edges(a, k, rng):
    d = len(a)
    tries = 0
    while k > 0 and tries < 50 * d * d:
        tries += 1
        i, j = rng.integers(d, size=2)
        if i == j or a[i, j] or a[j, i]:
            continue
        a[i, j] = 1
        if is_dag(a):
            k -= 1
        else:
            a[i, j] = 0
    return a


def corrupt(adj, k, rng):
    a = adj.copy()
    for i, j in zip(*np.nonzero(adj)):
        if rng.random() < k:
            a[i, j] = 0
            if rng.random() < 0.5:
                a[j, i] = 1
                if not is_dag(a):
                    a[j, i] = 0
    return _add_edges(a, int(round(k * adj.sum())), rng)


def random_graph(d, n_edges, rng):
    return _add_edges(np.zeros((d, d), int), int(n_edges), rng)


def f1(pred, true):
    pred, true = set(pred), set(true)
    tp = len(pred & true)
    if not pred and not true:
        return 1.0
    return 2 * tp / (len(pred) + len(true)) if tp else 0.0


def learn(name, X):
    from benchmarks import methods
    if name == "fennec":
        return methods.fennec(X, n_jobs=1)
    return getattr(methods, name)(X)


def run_system(ds, cfg):
    d, n, mech, seed = ds["d"], ds["n"], ds["mechanism"], ds["seed"]
    df, adj, _ = simulate(d, n + cfg["n_test"], mech, seed, noise_sd=cfg["noise_sd"])
    X = df.to_numpy()
    mu, sd = X[:n].mean(0), X[:n].std(0)
    sd[sd < 1e-12] = 1.0
    X = (X - mu) / sd  # standardize with training statistics only
    tr, te = X[:n], X[n:]
    rng = np.random.default_rng(10_000 + seed * 101 + d * 7 + n)
    t = choose_outcome(adj, rng)

    graphs = {"oracle": (adj, 0.0), "empty": (np.zeros_like(adj), 0.0)}
    for k in cfg["corruption"]:
        graphs[f"corrupt_{k:g}"] = (corrupt(adj, k, rng), 0.0)
    for r in range(cfg["n_random"]):
        graphs[f"random_{r}"] = (random_graph(d, adj.sum(), rng), 0.0)
    errors = {}
    for name in cfg["learners"]:
        t0 = time.time()
        try:
            graphs[name] = (np.asarray(learn(name, tr)).astype(int), time.time() - t0)
        except Exception as e:  # record and continue
            errors[name] = f"{type(e).__name__}: {e}"

    true_pa, true_mb = parents(adj, t), markov_blanket(adj, t)
    rows = []
    for gname, (g, secs) in graphs.items():
        m = graph_metrics(g, adj)
        info = {"graph": gname, "learn_seconds": secs, "shd": int(m["shd"]), "skel_f1": float(m["skel_f1"]),
                "n_edges": int(m["n_edges"]), "parent_f1": f1(parents(g, t), true_pa),
                "blanket_f1": f1(markov_blanket(g, t), true_mb)}
        for cname in cfg["compilers"]:
            try:
                yhat = make_compiler(cname, g, t, xgb_params=cfg["xgb"], seed=seed).fit(tr).predict(te)
                mse = float(np.mean((yhat - te[:, t]) ** 2))
                rows.append({**info, "compiler": cname, "r2": 1 - mse / float(te[:, t].var()), "rmse": mse ** 0.5})
            except Exception as e:
                rows.append({**info, "compiler": cname, "error": f"{type(e).__name__}: {e}"})
    return {**ds, "outcome": t, "n_parents": len(true_pa), "n_children": int(adj[t].sum()),
            "blanket_size": len(true_mb), "true_edges": int(adj.sum()), "learner_errors": errors, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e1.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.pilot:
        cfg.update(mechanisms=["mlp", "linear"], d=[10], n=[200], seeds=[0, 5], learners=["pc", "fennec"])
    RAW.mkdir(parents=True, exist_ok=True)
    # one file per shard: concurrent appends to a shared file can interleave on network filesystems
    tag = "_pilot" if a.pilot else (f"_shard{a.shard}" if a.nshards > 1 else "")
    out = RAW / f"{cfg['name']}{tag}.jsonl"
    systems = [{"mechanism": m, "d": d, "n": n, "seed": s}
               for m, d, n, s in product(cfg["mechanisms"], cfg["d"], cfg["n"], cfg["seeds"])][a.shard::a.nshards]
    key = lambda r: (r["mechanism"], r["d"], r["n"], r["seed"])
    done = {key(json.loads(l)) for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in systems if key(s) not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(systems)} systems, {len(todo)} to do -> {out}", flush=True)
    gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(delayed(run_system)(s, cfg) for s in todo)
    for k, rec in enumerate(gen, 1):
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  [{k}/{len(todo)}] {key(rec)} done", flush=True)


if __name__ == "__main__":
    main()
