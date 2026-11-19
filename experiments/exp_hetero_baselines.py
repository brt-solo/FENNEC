"""Heterogeneity baselines: CD-NOD and a joint-causal-inference (JCI) FENNEC variant, on the same systems as
exp_islands.py (4 discrete environments) and exp_genetic_clades.py experiment A (continuous genetic context).

The FENNEC heterogeneity tools (islands, genetic clades) were already run on these systems; this script
regenerates each system's data exactly and adds the standard alternatives, which add the context to
the pooled data as extra variable(s) that nothing can cause:
    cdnod_fisherz   CD-NOD (Huang et al. 2020) phase 1: PC on [X, context] with no edges into the context,
                    Fisher-z tests (linear); a node "changes" if it stays adjacent to the context
    cdnod_rcit      the same with the randomized conditional independence test (RCIT; nonlinear)
    fennec_jci      JCI-style FENNEC (Mooij et al. 2020): FENNEC hill climbing on [X, context] with edges
                    out of the context only; a node changes if the context is among its parents
Context: islands  the environment index (0-3);  genetic  the 3 observed pathway PGS.
Per node: called (bool) and a continuous score for AUROC
    CD-NOD      -log10 p of context _||_ X_j | X_j's other neighbours (min over contexts)
    FENNEC-JCI  local-score gain of adding the context to X_j's final parents
Islands also get the same methods with shuffled environment labels (a null: every call is false).
These methods report which NODES change; the FENNEC tools also say which EDGE changes and where.
Results: results/hetero_baselines.jsonl
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
import traceback
from itertools import product

import numpy as np
from joblib import Parallel, delayed

from experiments.exp_effects import OUT

RESULTS = OUT / "hetero_baselines.jsonl"
ALPHA = 0.05


def cdnod(X, C, test):
    """PC on [X, C] with nothing pointing into C. Returns (called, score) per node of X."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import CIT
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    d, m = X.shape[1], C.shape[1]
    if test == "rcit":  # RCIT's random-feature covariance can fail to decompose with a discrete context; tiny jitter fixes it
        C = C + np.random.default_rng(0).normal(0, 1e-3, C.shape)
    data = np.column_stack([X, C])
    cg0 = pc(data, ALPHA, test, show_progress=False)  # node objects for the background knowledge
    nodes = cg0.G.get_nodes()
    bk = BackgroundKnowledge()
    for c in range(d, d + m):
        for j in range(d + m):
            if j != c:
                bk.add_forbidden_by_node(nodes[j], nodes[c])
    cg = pc(data, ALPHA, test, background_knowledge=bk, show_progress=False)
    A = cg.G.graph != 0
    cit = CIT(data, test)
    called, score = np.zeros(d, bool), np.zeros(d)
    for j in range(d):
        called[j] = A[j, d:].any() or A[d:, j].any()
        nb = [k for k in range(d) if k != j and (A[j, k] or A[k, j])]
        ps = [cit(c, j, nb) for c in range(d, d + m)]
        score[j] = -np.log10(max(min(ps), 1e-300))
    return called, score


def fennec_jci(X, C):
    from fennec_search.score import CVScore
    from fennec_search.search import full_mask, hill_climb
    d, m = X.shape[1], C.shape[1]
    S = CVScore(np.column_stack([X, C]), n_jobs=1)
    mask = full_mask(d + m)
    mask[:, d:] = False
    A, _ = hill_climb(S, mask=mask)
    ctx = tuple(range(d, d + m))
    called, score = np.zeros(d, bool), np.zeros(d)
    for j in range(d):
        pa = tuple(int(p) for p in np.flatnonzero(A[:d, j]))
        called[j] = A[d:, j].any()
        score[j] = S.local(j, tuple(sorted(pa + ctx))) - S.local(j, pa)
    return called, score


def methods(X, C):
    out = {}
    for name, f in (("cdnod_fisherz", lambda: cdnod(X, C, "fisherz")), ("cdnod_rcit", lambda: cdnod(X, C, "rcit")),
                    ("fennec_jci", lambda: fennec_jci(X, C))):
        t = time.time()
        called, score = f()
        out[name] = {"called": called.tolist(), "score": score.round(5).tolist(), "seconds": round(time.time() - t, 1)}
    return out


def run_islands(mech, n_env, seed):
    from experiments.exp_islands import D, K, EnvSystem
    es = EnvSystem(D, mech, seed)
    datas = [es.sample(k, n_env, seed * 10 + k + 1) for k in range(K)]
    X = np.vstack(datas)
    C = np.repeat(np.arange(K), n_env)[:, None].astype(float)
    pos = sorted({int(c) for e in es.envs for c in e["cut"]} | {int(c) for e in es.envs for _, c, _ in e["extra"]})
    Cs = np.random.default_rng(seed + 123).permutation(C)
    return {"setting": "islands", "mechanism": mech, "n": n_env, "seed": seed, "changing": pos, "main": [],
            "methods": methods(X, C), "null_methods": methods(X, Cs)}


def run_genetic(scenario, mech, n, seed):
    from experiments.exp_genetic_clades import GenSystem
    gs = GenSystem(mech, seed, gxe=scenario == "gxe")
    dat = gs.sample(n, seed * 7 + 3)
    pos = sorted({gs.cut} | {c for _, c, _ in gs.extra}) if scenario == "gxe" else []
    main = sorted({t for _, t, _ in gs.main} - set(pos))
    return {"setting": "genetic", "scenario": scenario, "mechanism": mech, "n": n, "seed": seed, "changing": pos,
            "main": main, "methods": methods(dat["X"], dat["pgs"])}


def safe(kind, *a):
    t0 = time.time()
    try:
        r = run_islands(*a) if kind == "islands" else run_genetic(*a)
        r["seconds"] = time.time() - t0
        return r
    except Exception:  # noqa: BLE001
        return {"setting": kind, "args": list(a), "error": traceback.format_exc()[-1500:]}


def key(r):
    return (r["setting"], r.get("scenario", ""), r["mechanism"], r["n"], r["seed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        for r in (safe("islands", "additive", 300, 0), safe("genetic", "gxe", "additive", 2000, 0)):
            print(r.get("error") or {k: (v if k not in ("methods", "null_methods") else
                                         {m: (x["called"], x["seconds"]) for m, x in v.items()}) for k, v in r.items()})
        raise SystemExit
    grid = [("islands", m, n, s) for m, n, s in product(["linear", "additive", "mlp", "mixed"], [1000, 300], range(10))]
    grid += [("genetic", sc, m, n, s) for sc, m, n, s in product(["gxe", "null"], ["linear", "additive", "mlp"], [2000, 4000], range(8))]
    done = set()
    if RESULTS.exists():
        done = {key(r) for r in map(json.loads, RESULTS.read_text().splitlines()) if "error" not in r}
    todo = []
    for i, g in enumerate(grid):
        k = (g[0], "", *g[1:]) if g[0] == "islands" else g
        if i % args.nshards == args.shard and k not in done:
            todo.append(g)
    print(f"{len(todo)} systems", flush=True)
    for r in Parallel(n_jobs=args.jobs, backend="loky", return_as="generator_unordered")(delayed(safe)(*g) for g in todo):
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(r["setting"], "ERROR" if "error" in r else f"{r['mechanism']} {r['n']} {r['seed']} {r['seconds']:.0f}s", flush=True)
    print("done", flush=True)
