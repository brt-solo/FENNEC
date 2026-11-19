"""Run the synthetic structure-learning benchmark (resumable).

Grid: mechanism in {linear, additive, mlp, mixed} x d in {10, 20} x n in {200, 1000} x 10 seeds (0-4 exploratory, 5-9 confirmatory)
(the same random graph for every mechanism at a given (d, seed), so comparisons are paired).
Every method in benchmarks.methods.METHODS is run on every dataset. Results are appended to
benchmarks/results/results.jsonl; (dataset, method) pairs already present are skipped.

    python -m benchmarks.run                       # full grid, all methods
    python -m benchmarks.run --pilot               # d = 10, n = 200 only, 2 seeds
    python -m benchmarks.run --methods fennec pc   # selected methods
    python -m benchmarks.run --noise-sd 1.0        # signal-to-noise 1:1 (writes results_noise1.jsonl)
    python -m benchmarks.run --generator gcastle   # outside generator, ER2 + SF2 graphs (results_gcastle.jsonl)

Metrics: skeleton precision/recall/F1, Markov-blanket F1, correctly oriented edges, SHD,
runtime; plus each dataset's R^2-sortability (Kendall tau between how predictable a variable
is and its position in the causal order; ~0 means no order leakage) and varsortability (0.5 = none).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from pathlib import Path

from benchmarks.generate import MECHANISMS, simulate
from benchmarks.generate_gcastle import GC_MECHANISMS, simulate_gcastle
from benchmarks.methods import METHODS, PARALLEL_SAFE
from benchmarks.metrics import graph_metrics
from benchmarks.sortability import r2_sortability, varsortability

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
RESULTS = OUT / "results.jsonl"


def blanket_f1(pred, truth):
    """Markov-blanket F1 summed over nodes. Undirected predicted edges count as neighbours;
    co-parents only come from directed edges."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    d = len(truth)

    def blankets(a):
        directed = a & (1 - a.T)
        nb = (a | a.T).astype(bool)
        out = []
        for v in range(d):
            ch = np.flatnonzero(directed[v])
            co = set(np.flatnonzero(directed[:, ch].any(1))) if len(ch) else set()
            out.append((set(np.flatnonzero(nb[v])) | co) - {v})
        return out

    bp, bt = blankets(pred), blankets(truth)
    tp = sum(len(p & t) for p, t in zip(bp, bt))
    fp = sum(len(p - t) for p, t in zip(bp, bt))
    fn = sum(len(t - p) for p, t in zip(bp, bt))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return 2 * prec * rec / (prec + rec) if tp else 0.0


def make_data(ds, noise_sd=0.5):
    if "graph" in ds:  # gCastle dataset
        return simulate_gcastle(ds["d"], ds["n"], ds["mechanism"], ds["seed"], graph=ds["graph"])
    return simulate(ds["d"], ds["n"], ds["mechanism"], ds["seed"], noise_sd=noise_sd)


def run_one(ds, method, noise_sd=0.5):
    df, adj, info = make_data(ds, noise_sd)
    X = df.to_numpy()
    t = time.time()
    try:
        pred = METHODS[method](X)
        err = None
    except Exception as e:  # record failures instead of crashing the grid
        pred, err = np.zeros_like(adj), f"{type(e).__name__}: {e}"
    secs = time.time() - t
    m = graph_metrics(pred, adj)
    return {**ds, "method": method, "seconds": secs, "error": err, "true_edges": int(adj.sum()),
            "blanket_f1": blanket_f1(pred, adj), **{k: m[k] for k in
            ["n_edges", "n_undirected", "skel_tp", "skel_precision", "skel_recall", "skel_f1", "dir_tp", "shd"]},
            "r2_sortability": r2_sortability(X, adj), "varsortability": varsortability(X, adj),
            **({"varsortability_raw": info["varsortability_raw"]} if "varsortability_raw" in info else {}),
            "pred_edges": [[int(i), int(j)] for i, j in zip(*np.nonzero(pred))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--n", nargs="*", type=int, default=[200, 1000])
    ap.add_argument("--d", nargs="*", type=int, default=[10, 20])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)),
                    help="graph seeds; 5-9 = confirmatory set (penalty fixed before it was run)")
    ap.add_argument("--noise-sd", type=float, default=0.5,
                    help="noise SD on the standardized signal (0.5 = signal-to-noise 4:1; 1.0 = 1:1); "
                         "non-default values write to results_noise<sd>.jsonl")
    ap.add_argument("--generator", choices=["fennec", "gcastle"], default="fennec",
                    help="gcastle = outside generator (gCastle IIDSimulation), ER2 and SF2 graphs")
    args = ap.parse_args()
    global RESULTS
    if args.generator == "gcastle":
        RESULTS = OUT / "results_gcastle.jsonl"
    elif args.noise_sd != 0.5:
        RESULTS = OUT / f"results_noise{args.noise_sd:g}.jsonl"
    grid = (dict(d=10, n=200, seeds=[0, 1]) if args.pilot else dict(d=args.d, n=args.n, seeds=args.seeds))
    if args.generator == "gcastle":
        ds_list = [{"mechanism": m, "graph": g, "d": d, "n": n, "seed": s} for g, m, d, n, s in
                   product(["ER", "SF"], GC_MECHANISMS, np.atleast_1d(grid["d"]), np.atleast_1d(grid["n"]), grid["seeds"])]
    else:
        ds_list = [{"mechanism": m, "d": d, "n": n, "seed": s} for m, d, n, s in
                   product(MECHANISMS, np.atleast_1d(grid["d"]), np.atleast_1d(grid["n"]), grid["seeds"])]
    ds_list = [{k: (int(v) if isinstance(v, np.integer) else v) for k, v in ds.items()} for ds in ds_list]
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            r = json.loads(line)
            done.add((r.get("graph"), r["mechanism"], r["d"], r["n"], r["seed"], r["method"]))
    todo = [(ds, m) for ds in ds_list for m in args.methods
            if (ds.get("graph"), ds["mechanism"], ds["d"], ds["n"], ds["seed"], m) not in done]
    print(f"{len(ds_list)} datasets, {len(todo)} (dataset, method) runs to do", flush=True)

    def write(rows):
        with open(RESULTS, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    par = [(ds, m) for ds, m in todo if m in PARALLEL_SAFE]
    seq = [(ds, m) for ds, m in todo if m not in PARALLEL_SAFE]
    # single-threaded baselines: many at once, in chunks so progress is saved
    for k in range(0, len(par), 48):
        chunk = par[k:k + 48]
        write(Parallel(n_jobs=-1, backend="loky")(delayed(run_one)(ds, m, args.noise_sd) for ds, m in chunk))
        print(f"  baselines: {min(k + 48, len(par))}/{len(par)} done", flush=True)
    # our XGBoost methods: one at a time, each parallel inside
    for k, (ds, m) in enumerate(seq):
        r = run_one(ds, m, args.noise_sd)
        write([r])
        print(f"  {m} {ds}: skelF1 {r['skel_f1']:.2f} SHD {r['shd']} ({r['seconds']:.0f}s)  [{k + 1}/{len(seq)}]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
