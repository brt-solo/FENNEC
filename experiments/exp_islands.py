"""Environments as islands: which causal structure is shared and which is environment-specific?

Each environment's data evolves its own MAP-Elites island (FENNEC score on that environment only),
with optional migration of top elites between islands. Migrants carry their edge genealogy, and
origin ids are tagged with the island where the edge was born, so "edge independently evolved on
island k" (convergent / analogous structure) is separated from "edge arrived by migration".

Synthetic systems with known heterogeneity (EnvSystem): a TrueSystem (d = 10) shared by K = 4
environments. Environment 0 is the reference; in each environment k >= 1
    cut    one non-root node loses all its incoming edges (its mechanism is replaced by noise:
           a soft intervention, as in Sachs' reagent conditions)
    extra  one new edge a -> b (consistent with the causal order) exists only there
so each edge is SHARED (present in all environments), VARIABLE (present in some) or ABSENT.

Arms per system (same search budget per island):
    pooled        one search on all environments' data together (cannot express differences)
    island_m0     one island per environment, no migration (= fully separate searches)
    island_m2     2 top elites from every other island each epoch
    island_m8     8 top elites from every other island each epoch
    null_m0/m2    the pooled data randomly split into K parts of the same sizes: differences
                  between these islands are sampling noise only (calibrates heterogeneity calls)
    score_gain    no evolution: for each candidate edge i -> j, the gain in j's local score from
                  adding i (given j's pooled-graph parents) in each environment, with subsample
                  standard errors; Cochran's Q tests whether the gain differs between environments
                  (also computed on the null split)
Per island: population = near-optimal set (bootstrap noise band) after hill-climbing polish;
P[k, i, j] = share of the population with edge i -> j; L[k, i, j] = share with that edge AND a
locally born origin. Results: results/islands.jsonl (analysed by analyze_islands.py).
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
from scipy.stats import chi2

from experiments.exp_effects import OUT
from experiments.truth import NOISE_SD, TrueSystem

RESULTS = OUT / "islands.jsonl"
K, D = 4, 10
EPOCHS, GENS, INIT, BATCH = 5, 6, 200, 40
OFFSET = 10 ** 9
MIGRATION = {"m0": 0, "m2": 2, "m8": 8}
SUB_FRAC, SUB_REPS = 0.8, 5


class EnvSystem:
    def __init__(self, d, mechanism, seed, k_env=K, extra_strength=1.2):
        self.base = b = TrueSystem(d, mechanism, seed)
        rng = np.random.default_rng(seed + 999)
        pos = {int(v): i for i, v in enumerate(b.order)}
        nonroot = [j for j in range(d) if b.mech[j] is not None]
        cuts = list(rng.choice(nonroot, size=min(k_env - 1, len(nonroot)), replace=False))
        self.envs = [{"cut": set(), "extra": []}]
        used = set()
        for k in range(1, k_env):
            cut = {int(cuts[k - 1])} if k - 1 < len(cuts) else set()
            cand = [(a, c) for a in range(d) for c in range(d) if pos[a] < pos[c] and not b.adj[a, c]
                    and (a, c) not in used and c not in cut]
            a, c = cand[rng.integers(len(cand))]
            used.add((a, c))
            self.envs.append({"cut": cut, "extra": [(int(a), int(c), float(extra_strength * rng.choice([-1, 1])))]})
        self.k_env, self.d = k_env, d

    def adj(self, k):
        A = self.base.adj.astype(int).copy()
        for j in self.envs[k]["cut"]:
            A[:, j] = 0
        for a, c, _ in self.envs[k]["extra"]:
            A[a, c] = 1
        return A

    def sample(self, k, n, seed):
        b, env = self.base, self.envs[k]
        rng = np.random.default_rng(seed)
        X = np.zeros((n, self.d))
        for j in b.order:
            if b.mech[j] is None or j in env["cut"]:
                raw = rng.normal(0, 1, n)
                if b.mech[j] is not None:
                    rng.normal(0, NOISE_SD, n)  # keep the random stream aligned with other environments
            else:
                f = b._signal(j, X)
                raw = (f - b.sig[j][0]) / b.sig[j][1] + rng.normal(0, NOISE_SD, n)
            for a, c, w in env["extra"]:
                if c == j:
                    xa = X[:, a] - (np.mean(X[:, a]) if a in b.discrete else 0.0)
                    raw = raw + w * np.tanh(1.5 * xa)
            if j in b.discrete:
                X[:, j] = np.digitize(raw, b.cuts[j])
            else:
                X[:, j] = (raw - b.std[j][0]) / b.std[j][1]
        return X


def top_elites(me, m):
    if m == 0:
        return []
    return [a for a, _ in sorted(me.archive.values(), key=lambda t: -t[1])[:m]]


def evolve(datas, m, seed, max_graphs=30):
    """Island-model FENNEC. Returns P, L (K x d x d) and the per-island best graphs."""
    from fennec_search.score import CVScore
    from fennec_search.search import MapElites, hill_climb, near_optimal, noise_band
    scores = [CVScore(X, n_jobs=1) for X in datas]
    isl = []
    for k, s in enumerate(scores):
        me = MapElites(s, seed=seed * 100 + k)
        me._event = k * OFFSET
        isl.append(me)
    for k, me in enumerate(isl):
        me.run(n_generations=GENS, init_population=INIT, batch_size=BATCH, seeds=[hill_climb(scores[k])[0]])
    for _ in range(1, EPOCHS):
        tops = [top_elites(me, m) for me in isl]
        for k, me in enumerate(isl):
            migr = []
            for k2, t in enumerate(tops):
                if k2 == k:
                    continue
                for a in t:
                    key = a.astype(np.int8).tobytes()
                    if key not in me.origins:
                        me.origins[key] = isl[k2].origins[key]
                    migr.append(a)
            me.run(n_generations=GENS, init_population=0, batch_size=BATCH, seeds=migr)
    d = datas[0].shape[1]
    P, L, best = np.zeros((len(datas), d, d)), np.zeros((len(datas), d, d)), []
    for k, (me, X) in enumerate(zip(isl, datas)):
        b0, _ = me.best()
        pol, _ = hill_climb(scores[k], start=b0)
        me._insert([pol], [[b0]])
        bk, _ = me.best()
        cands = [np.frombuffer(key, dtype=np.int8).reshape(d, d) for key, _ in
                 sorted(me.pool.items(), key=lambda kv: -kv[1])[1:200]]
        band = noise_band(X, bk, cands[::10], kind="bootstrap", n_jobs=1, seed=seed)
        pop = near_optimal(me.pool, d, band, max_graphs)
        for g, _ in pop:
            P[k] += g
            org = me.origins.get(g.astype(np.int8).tobytes(), {})
            for (i, j), ids in org.items():
                if any(o // OFFSET == k for o in ids):
                    L[k, i, j] += 1
        P[k] /= len(pop); L[k] /= len(pop)
        best.append(bk.astype(int))
    return P, L, best


def score_gain(datas, edges, parents_of, seed):
    """Per environment: mean and standard error of the local-score gain of each edge i -> j."""
    from fennec_search.score import CVScore
    G = np.zeros((len(datas), len(edges))); SE = np.zeros_like(G)
    rng = np.random.default_rng(seed)
    for k, X in enumerate(datas):
        reps = []
        for r in range(SUB_REPS):
            Xs = X[rng.choice(len(X), int(SUB_FRAC * len(X)), replace=False)]
            S = CVScore(Xs, n_jobs=1, random_state=r)
            reps.append([S.local(j, tuple(sorted(set(parents_of[j]) - {i} | {i}))) -
                         S.local(j, tuple(sorted(set(parents_of[j]) - {i}))) for i, j in edges])
        reps = np.array(reps)
        G[k] = reps.mean(0)
        # subsamples of fraction f vary around the full-data value with variance (1 - f) / f x Var(full)
        SE[k] = np.maximum(reps.std(0, ddof=1) * np.sqrt(SUB_FRAC / (1 - SUB_FRAC)), 1e-6)
    w = 1 / SE ** 2
    mu = (w * G).sum(0) / w.sum(0)
    Q = (w * (G - mu) ** 2).sum(0)
    return G, SE, Q, chi2.sf(Q, len(datas) - 1)


def run(mech, n_env, seed):
    t0 = time.time()
    es = EnvSystem(D, mech, seed)
    datas = [es.sample(k, n_env, seed * 10 + k + 1) for k in range(K)]
    T = np.array([es.adj(k) for k in range(K)])
    pooled = np.vstack(datas)
    perm = np.random.default_rng(seed + 77).permutation(len(pooled))
    nulls = [pooled[perm[k * n_env:(k + 1) * n_env]] for k in range(K)]
    out = {"mechanism": mech, "n_env": n_env, "seed": seed, "truth": T.tolist(),
           "envs": [{"cut": sorted(int(c) for c in e["cut"]), "extra": e["extra"]} for e in es.envs], "arms": {}}
    Pp, _, bp = evolve([pooled], 0, seed)
    out["arms"]["pooled"] = {"P": np.repeat(Pp, K, 0).round(4).tolist()}
    for name, m in MIGRATION.items():
        P, L, _ = evolve(datas, m, seed)
        out["arms"][f"island_{name}"] = {"P": P.round(4).tolist(), "L": L.round(4).tolist()}
    for name in ("m0", "m2"):
        P, L, _ = evolve(nulls, MIGRATION[name], seed)
        out["arms"][f"null_{name}"] = {"P": P.round(4).tolist(), "L": L.round(4).tolist()}
    # score-gain heterogeneity test on every edge any arm found (> 0.2 anywhere), given pooled parents
    anyP = np.max([np.max(np.array(a["P"]), 0) for a in out["arms"].values()], 0)
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(anyP > 0.2))]
    parents_of = {j: [int(p) for p in np.flatnonzero(bp[0][:, j])] for j in range(D)}
    for tag, dd in (("score_gain", datas), ("score_gain_null", nulls)):
        G, SE, Q, p = score_gain(dd, edges, parents_of, seed)
        out[tag] = {"edges": edges, "gain": G.round(5).tolist(), "se": SE.round(5).tolist(), "Q": Q.round(4).tolist(),
                    "p": p.tolist()}
    out["seconds"] = time.time() - t0
    return out


def safe(*a):
    try:
        return run(*a)
    except Exception:  # noqa: BLE001
        return {"mechanism": a[0], "n_env": a[1], "seed": a[2], "error": traceback.format_exc()[-1500:]}


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
                done.add((r["mechanism"], r["n_env"], r["seed"]))
    grid = list(product(["linear", "additive", "mlp", "mixed"], [1000, 300], range(args.seeds)))
    grid = [g for i, g in enumerate(grid) if i % args.nshards == args.shard and g not in done]
    print(f"{len(grid)} systems", flush=True)
    for r in Parallel(n_jobs=args.jobs, backend="loky", return_as="generator_unordered")(delayed(safe)(*g) for g in grid):
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(r["mechanism"], r["n_env"], r["seed"], "ERROR" if "error" in r else f"{r['seconds']:.0f}s", flush=True)
    print("done", flush=True)
