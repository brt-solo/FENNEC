"""Genetic clades: does grouping people by genetic similarity reveal genotype-specific causal structure?

Idea under test: split the sample into clades of genetically similar individuals (a tree, like a
phylogeny), learn a causal graph per clade with FENNEC, and read differences between clades as
genotype-dependent causal structure. The danger: clades also differ in ancestry, exposures and
trait means, which can make graphs differ without any genotype x mechanism interaction.

Synthetic cohort (GenSystem): 500 SNPs in 3 ancestral populations (Balding-Nichols, Fst = 0.1).
Three pathway scores g1, g2, g3 (30 causal SNPs each) are the true genetic factors; each child
also has an observed noisy pathway PGS (correlation 0.7 with the true score). Phenotypes: a
TrueSystem graph (d = 10) with
    main effects   g1 -> one node, g2 -> one node (0.3 SD), g3 -> outcome (0.6 SD); in both scenarios
    ancestry       each ancestral population shifts the mean of every root (exposure) node
    G x E gates    scenario "gxe" only, gate s_k = sigmoid(4 g_k):
                     g1 switches ON an extra edge e1 (1.2 tanh, as in exp_islands)
                     g2 switches ON an extra edge e2 and switches OFF all inputs of node c
                   scenario "null": no gates, so every difference between clades is spurious
The single "trait PGS" is 0.3 pgs1 + 0.3 pgs2 + 0.9 pgs3: like a real trait PGS it is dominated by
the pathway with the largest main effect, which here modifies no edge.

Clade trees: recursive median bisection on the first principal component of a feature space
    random     random halves (null: calibrates how much clades differ by chance)
    pgs_total  the single trait PGS (depth 2 = quartiles; the "stratify by PGS" idea)
    pathway    the 3 observed pathway PGS ("finer genetics")
    ancestry   top 10 genome-wide genotype PCs ("genetically similar" in the usual sense)
    oracle     true g1 at depth 1, true g2 at depth 2 (best possible split)
Each tree node gets its own FENNEC search (MAP-Elites, then hill-climb polish, near-optimal
population), seeded with its parent clade's top elites ("inheritance", the phylogeny), so
P[node, i, j] = share of the node's population with edge i -> j. Root = pooled search.

Also: interaction  no clades: for each candidate edge i -> j, an F-test of x_i x PGS terms in a
regression of x_j on a flexible basis of its pooled parents, the 3 PGS and the interactions
(the standard pooled G x E test).

Experiment A (results/genetic_clades.jsonl): depth-2 trees for all schemes.
Experiment B (results/genetic_clades_depth.jsonl): depth-4 trees (2-16 clades) for random and
pathway, plus a 3 x 3 grid of pgs1 x pgs2 terciles evolved as islands with neighbour migration
("MAP-Elites cells defined by genetic scores") and a size-matched random grid.
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
from scipy.stats import f as f_dist

from experiments.exp_effects import OUT
from experiments.exp_islands import top_elites
from experiments.truth import NOISE_SD, TrueSystem

RESULTS = {"A": OUT / "genetic_clades.jsonl", "B": OUT / "genetic_clades_depth.jsonl"}
D, N_SNP, N_POP, FST, N_CAUSAL, N_PC = 10, 500, 3, 0.1, 30, 10
GENS, INIT, BATCH, INHERIT = 30, 200, 40, 4
EPOCHS, EPOCH_GENS, MIGRANTS = 5, 6, 2
PGS_R = 0.7
TOTAL_W = np.array([0.3, 0.3, 0.9])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GenSystem:
    def __init__(self, mechanism, seed, gxe):
        self.base = b = TrueSystem(D, mechanism, seed)
        self.gxe = gxe
        rng = np.random.default_rng(seed + 4242)
        p0 = rng.uniform(0.1, 0.9, N_SNP)
        c = (1 - FST) / FST
        self.freq = rng.beta(p0 * c, (1 - p0) * c, (N_POP, N_SNP)).clip(0.01, 0.99)
        snps = rng.permutation(N_SNP)
        self.path = [snps[k * N_CAUSAL:(k + 1) * N_CAUSAL] for k in range(3)]
        self.beta = [rng.normal(0, 1, N_CAUSAL) for _ in range(3)]
        _, G = self._genotypes(np.random.default_rng(seed + 1), 20_000)
        raw = self._raw_scores(G)
        self.g_mu, self.g_sd = raw.mean(0), raw.std(0)
        pos = {int(v): i for i, v in enumerate(b.order)}
        nonroot = [j for j in range(D) if b.mech[j] is not None]
        self.cut = int(rng.choice(nonroot))
        self.extra, used = [], set()
        for _ in range(2):
            cand = [(a, c) for a in range(D) for c in range(D) if pos[a] < pos[c] and not b.adj[a, c]
                    and c != self.cut and (a, c) not in used]
            a, c = cand[rng.integers(len(cand))]
            used.add((a, c))
            self.extra.append((int(a), int(c), float(1.2 * rng.choice([-1, 1]))))
        self.outcome = int(b.order[-1])
        self.main = [(0, int(rng.integers(D)), 0.3), (1, int(rng.integers(D)), 0.3), (2, self.outcome, 0.6)]
        self.pop_shift = {j: rng.normal(0, 0.5, N_POP) for j in range(D) if b.mech[j] is None}

    def _genotypes(self, rng, n):
        pop = rng.integers(N_POP, size=n)
        return pop, rng.binomial(2, self.freq[pop]).astype(float)

    def _raw_scores(self, G):
        return np.column_stack([G[:, s] @ w for s, w in zip(self.path, self.beta)])

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        pop, G = self._genotypes(rng, n)
        g = (self._raw_scores(G) - self.g_mu) / self.g_sd
        s = sigmoid(4 * g[:, :2]) if self.gxe else np.zeros((n, 2))
        b = self.base
        X = np.zeros((n, D))
        for j in b.order:
            if b.mech[j] is None:
                raw = rng.normal(0, 1, n) + self.pop_shift[j][pop]
            else:
                f = (b._signal(j, X) - b.sig[j][0]) / b.sig[j][1]
                eps = rng.normal(0, 1, n)
                if j == self.cut:
                    f = (1 - s[:, 1]) * f + s[:, 1] * eps
                raw = f + rng.normal(0, NOISE_SD, n)
            for k, (a, c, w) in enumerate(self.extra):
                if c == j:
                    raw = raw + w * s[:, k] * np.tanh(1.5 * X[:, a])
            for k, t, m in self.main:
                if t == j:
                    raw = raw + m * g[:, k]
            X[:, j] = (raw - b.std[j][0]) / b.std[j][1]
        X = (X - X.mean(0)) / X.std(0)
        pgs = g + rng.normal(0, np.sqrt(1 / PGS_R ** 2 - 1), g.shape)
        return {"X": X, "G": G, "pop": pop, "g": g, "pgs": pgs, "gate": s}

    def truth(self, gate_mean):
        """Adjacency of a clade given its mean gates (s1, s2): an edge counts if its gate is > 0.5 on average."""
        A = self.base.adj.astype(int).copy()
        if not self.gxe:
            return A
        if gate_mean[1] > 0.5:
            A[:, self.cut] = 0
        for k, (a, c, _) in enumerate(self.extra):
            A[a, c] = int(gate_mean[k] > 0.5)
        return A

    def gated_pairs(self):
        """Skeleton pairs whose presence depends on genotype (empty in the null scenario)."""
        if not self.gxe:
            return []
        pairs = {tuple(sorted((a, c))) for a, c, _ in self.extra}
        pairs |= {tuple(sorted((int(p), self.cut))) for p in np.flatnonzero(self.base.adj[:, self.cut])}
        return sorted(pairs)


def bisect(F, idx, rng):
    """Split rows idx at the median of the first principal component of F[idx]; random if F is None."""
    if F is None:
        p = rng.permutation(idx)
    else:
        Z = F[idx] - F[idx].mean(0)
        pc = Z[:, 0] if Z.shape[1] == 1 else Z @ np.linalg.svd(Z, full_matrices=False)[2][0]
        p = idx[np.argsort(pc, kind="stable")]
    h = len(p) // 2
    return [np.sort(p[:h]), np.sort(p[h:])]


def genome_pcs(G):
    Z = (G - G.mean(0)) / np.maximum(G.std(0), 1e-9)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :N_PC] * S[:N_PC]


def splitters(dat, seed):
    """scheme -> function(idx, level) giving the two child clades."""
    rng = np.random.default_rng(seed + 31)
    total = (dat["pgs"] @ TOTAL_W)[:, None]
    pcs = genome_pcs(dat["G"])
    return {
        "random": lambda idx, lev: bisect(None, idx, rng),
        "pgs_total": lambda idx, lev: bisect(total, idx, rng),
        "pathway": lambda idx, lev: bisect(dat["pgs"], idx, rng),
        "ancestry": lambda idx, lev: bisect(pcs, idx, rng),
        "oracle": lambda idx, lev: bisect(dat["g"][:, [min(lev, 1)]], idx, rng),
    }


def finish(me, S, X, seed, max_graphs=30):
    """Polish the best graph, then the near-optimal population's edge frequencies."""
    from fennec_search.search import hill_climb, near_optimal, noise_band
    b0, _ = me.best()
    pol, _ = hill_climb(S, start=b0)
    me._insert([pol], [[b0]])
    bk, _ = me.best()
    cands = [np.frombuffer(key, dtype=np.int8).reshape(D, D) for key, _ in
             sorted(me.pool.items(), key=lambda kv: -kv[1])[1:200]]
    band = noise_band(X, bk, cands[::10], kind="bootstrap", n_jobs=1, seed=seed)
    pop = near_optimal(me.pool, D, band, max_graphs)
    return np.mean([g for g, _ in pop], 0), bk


def search_node(X, seeds, seed):
    from fennec_search.score import CVScore
    from fennec_search.search import MapElites, hill_climb
    S = CVScore(X, n_jobs=1)
    me = MapElites(S, seed=seed)
    me.run(n_generations=GENS, init_population=INIT, batch_size=BATCH, seeds=[hill_climb(S)[0]] + list(seeds))
    P, bk = finish(me, S, X, seed)
    return P, bk, top_elites(me, INHERIT)


def evolve_tree(X, split, depth, seed, root, gate):
    """Clade tree; node ids are bit strings ("" = root). Each child search is seeded by its parent's elites."""
    nodes = {"": {"idx": np.arange(len(X)), "res": root}}
    frontier = [""]
    for lev in range(depth):
        nxt = []
        for path in frontier:
            for b, sub in enumerate(split(nodes[path]["idx"], lev)):
                cid = path + str(b)
                nodes[cid] = {"idx": sub, "res": search_node(X[sub], nodes[path]["res"][2], seed * 1000 + int("1" + cid, 2))}
                nxt.append(cid)
        frontier = nxt
    return {k: {"P": v["res"][0].round(4).tolist(), "n": len(v["idx"]), "gate": gate[v["idx"]].mean(0).round(4).tolist()}
            for k, v in nodes.items()}


def evolve_grid(X, cells, nbrs, seed):
    """One MAP-Elites island per cell; each epoch every island receives its neighbours' top elites."""
    from fennec_search.score import CVScore
    from fennec_search.search import MapElites, hill_climb
    Ss = [CVScore(X[c], n_jobs=1) for c in cells]
    isl = [MapElites(S, seed=seed * 100 + k) for k, S in enumerate(Ss)]
    for S, me in zip(Ss, isl):
        me.run(n_generations=EPOCH_GENS, init_population=INIT, batch_size=BATCH, seeds=[hill_climb(S)[0]])
    for _ in range(1, EPOCHS):
        tops = [top_elites(me, MIGRANTS) for me in isl]
        for k, me in enumerate(isl):
            migr = []
            for k2 in nbrs[k]:
                for a in tops[k2]:
                    key = a.astype(np.int8).tobytes()
                    if key not in me.origins:
                        me.origins[key] = isl[k2].origins[key]
                    migr.append(a)
            me.run(n_generations=EPOCH_GENS, init_population=0, batch_size=BATCH, seeds=migr)
    return [finish(me, S, X[c], seed)[0] for me, S, c in zip(isl, Ss, cells)]


def interaction_test(X, pgs, edges, parents_of):
    """F-test p-value of the x_i x PGS block for each edge i -> j, given j's pooled parents."""
    basis = lambda x: [x, x ** 2, x ** 3, np.tanh(1.5 * x), np.abs(x), np.sin(x), np.exp(-x ** 2)]
    n, out = len(X), []
    for i, j in edges:
        pa = sorted(set(parents_of[j]) | {i})
        base = [np.ones(n)] + [b for p in pa for b in basis(X[:, p])] + list(pgs.T)
        inter = [X[:, i] * pgs[:, k] for k in range(pgs.shape[1])] + \
                [np.tanh(1.5 * X[:, i]) * pgs[:, k] for k in range(pgs.shape[1])]
        Z0, Z1 = np.column_stack(base), np.column_stack(base + inter)
        rss = [np.sum((X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]) ** 2) for Z in (Z0, Z1)]
        q, df = len(inter), n - Z1.shape[1]
        F = ((rss[0] - rss[1]) / q) / (rss[1] / df)
        out.append(float(f_dist.sf(F, q, df)))
    return out


def grid_cells(F2, rng):
    """3 x 3 grid of terciles of two scores; neighbours share a side. Also a random grid of the same sizes."""
    t = [np.digitize(F2[:, k], np.quantile(F2[:, k], [1 / 3, 2 / 3])) for k in range(2)]
    cells = [np.flatnonzero((t[0] == a) & (t[1] == b)) for a in range(3) for b in range(3)]
    nbrs = [[3 * a2 + b2 for a2, b2 in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)) if 0 <= a2 < 3 and 0 <= b2 < 3]
            for a in range(3) for b in range(3)]
    perm = rng.permutation(len(F2))
    cuts = np.cumsum([len(c) for c in cells])[:-1]
    return cells, [np.sort(c) for c in np.split(perm, cuts)], nbrs


def run(exp, scenario, mech, n, seed):
    t0 = time.time()
    gs = GenSystem(mech, seed, gxe=scenario == "gxe")
    dat = gs.sample(n, seed * 7 + 3)
    X = dat["X"]
    out = {"exp": exp, "scenario": scenario, "mechanism": mech, "n": n, "seed": seed,
           "adj": gs.base.adj.astype(int).tolist(), "cut": gs.cut, "extra": gs.extra, "main": gs.main,
           "gated_pairs": gs.gated_pairs(), "trees": {}}
    root = search_node(X, [], seed * 1000 + 1)
    sp = splitters(dat, seed)
    schemes, depth = (("random", "pgs_total", "pathway", "ancestry", "oracle"), 2) if exp == "A" else (("random", "pathway"), 4)
    for sc in schemes:
        out["trees"][sc] = evolve_tree(X, sp[sc], depth, seed, root, dat["gate"])
    # clade-level truth, so the analysis needs no access to the generator
    out["truth"] = {sc: {k: gs.truth(v["gate"]).tolist() for k, v in t.items()} for sc, t in out["trees"].items()}
    if exp == "A":
        anyP = np.max([np.array(v["P"]) for t in out["trees"].values() for v in t.values()], 0)
        edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(anyP > 0.2))]
        parents_of = {j: [int(p) for p in np.flatnonzero(root[1][:, j])] for j in range(D)}
        out["interaction"] = {"edges": edges, "p": interaction_test(X, dat["pgs"], edges, parents_of)}
    else:
        cells, rcells, nbrs = grid_cells(dat["pgs"][:, :2], np.random.default_rng(seed + 55))
        for name, cs in (("grid", cells), ("grid_random", rcells)):
            P = evolve_grid(X, cs, nbrs, seed)
            gm = [dat["gate"][c].mean(0) for c in cs]
            out["trees"][name] = {str(k): {"P": P[k].round(4).tolist(), "n": len(c), "gate": gm[k].round(4).tolist()}
                                  for k, c in enumerate(cs)}
            out["truth"][name] = {str(k): gs.truth(gm[k]).tolist() for k in range(len(cs))}
        out["nbrs"] = nbrs
    out["seconds"] = time.time() - t0
    return out


def safe(*a):
    try:
        return run(*a)
    except Exception:  # noqa: BLE001
        return dict(zip(("exp", "scenario", "mechanism", "n", "seed"), a), error=traceback.format_exc()[-1500:])


GRIDS = {"A": lambda s: list(product(["gxe", "null"], ["linear", "additive", "mlp"], [2000, 4000], range(s))),
         "B": lambda s: list(product(["gxe", "null"], ["additive", "mlp"], [4000], range(s)))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B"], default="A")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="one tiny system, short search, no output file")
    args = ap.parse_args()
    if args.smoke:
        GENS, INIT, EPOCH_GENS, EPOCHS = 2, 20, 1, 2
        r = run(args.exp, "gxe", "additive", 400, 0)
        print(json.dumps({k: r[k] for k in ("gated_pairs", "seconds")}), {sc: list(t) for sc, t in r["trees"].items()})
        raise SystemExit
    res = RESULTS[args.exp]
    done = set()
    if res.exists():
        for l in res.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add((r["scenario"], r["mechanism"], r["n"], r["seed"]))
    grid = [g for i, g in enumerate(GRIDS[args.exp](args.seeds)) if i % args.nshards == args.shard and g not in done]
    print(f"{len(grid)} systems", flush=True)
    for r in Parallel(n_jobs=args.jobs, backend="loky", return_as="generator_unordered")(
            delayed(safe)(args.exp, *g) for g in grid):
        with open(res, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(r["scenario"], r["mechanism"], r["n"], r["seed"], "ERROR" if "error" in r else f"{r['seconds']:.0f}s", flush=True)
    print("done", flush=True)
