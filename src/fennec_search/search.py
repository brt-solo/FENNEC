"""Search over DAGs with the FENNEC score.

hill_climb      best-improvement add / delete / reverse, from a start graph
MapElites       quality-diversity evolution (MAP-Elites) over DAGs; every evaluated graph is kept
                in `pool`, so the near-optimal population can be read off afterwards
near_optimal    graphs within a noise band of the best score, estimated by re-scoring on
                different cross-validation splits (differences smaller than that are noise)
"""
import numpy as np

from .graph import crossover, full_mask, mutate, random_dag
from .score import CVScore


def _reach(adj):
    r = adj.astype(bool)
    for k in range(len(adj)):
        r = r | (r[:, [k]] & r[[k], :])
    return r


def hill_climb(score, mask=None, start=None, max_iter=2000):
    d = score.d
    mask = full_mask(d) if mask is None else mask
    adj = np.zeros((d, d), dtype=np.int8) if start is None else np.array(start, dtype=np.int8)
    pa = lambda a, j: tuple(int(p) for p in np.flatnonzero(a[:, j]))
    for _ in range(max_iter):
        reach = _reach(adj)
        moves = []
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if adj[i, j]:
                    moves.append(("del", i, j, [(j, tuple(p for p in pa(adj, j) if p != i))]))
                    if mask[j, i]:
                        a2 = adj.copy(); a2[i, j] = 0
                        if not _reach(a2)[i, j]:
                            moves.append(("rev", i, j, [(j, tuple(p for p in pa(adj, j) if p != i)),
                                                        (i, tuple(sorted(pa(adj, i) + (j,))))]))
                elif mask[i, j] and not reach[j, i]:
                    moves.append(("add", i, j, [(j, tuple(sorted(pa(adj, j) + (i,))))]))
        score._compute([k for m in moves for k in m[3]] + score._keys(adj))
        best, best_gain = None, 1e-9
        for kind, i, j, keys in moves:
            gain = sum(score.cache[(node, new)] - score.cache[(node, pa(adj, node))] for node, new in keys)
            gain += {"add": -score.penalty, "del": score.penalty, "rev": 0.0}[kind]
            if gain > best_gain:
                best, best_gain = (kind, i, j), gain
        if best is None:
            break
        kind, i, j = best
        if kind == "add":
            adj[i, j] = 1
        elif kind == "del":
            adj[i, j] = 0
        else:
            adj[i, j], adj[j, i] = 0, 1
    return adj, score.score(adj)


class MapElites:
    """Archive cells: (edge density bin) x (max in-degree bin). Uniform elite selection;
    children are random immigrants, mutations (log-uniform rate) or crossovers.

    Edge genealogy (always on): every graph carries, per edge, the set of birth events at which
    that edge was created independently. An edge present in a parent inherits the parent's
    origin ids (union over both parents after crossover); an edge no parent had gets a fresh id.
    `origins[key]` maps each evaluated graph (adjacency bytes) to {edge: frozenset(origin ids)}."""

    def __init__(self, score, mask=None, n_density_bins=10, n_indeg_bins=6, seed=0):
        self.score, self.d = score, score.d
        self.mask = full_mask(self.d) if mask is None else mask
        self.max_edges = self.d * (self.d - 1) / 2
        self.nb, self.ni = n_density_bins, n_indeg_bins
        self.rng = np.random.default_rng(seed)
        self.archive, self.pool, self.origins = {}, {}, {}
        self._event = 0

    def _cell(self, a):
        dens = min(int(a.sum() / self.max_edges * self.nb), self.nb - 1)
        indeg = min(int(a.sum(0).max()), self.ni - 1)
        return dens, indeg

    def _birth(self, child, parents):
        """Assign edge origins to `child` given its parents' origins."""
        out = {}
        pk = [self.origins.get(p.astype(np.int8).tobytes(), {}) for p in parents]
        for i, j in zip(*np.nonzero(child)):
            e = (int(i), int(j))
            srcs = [o[e] for o in pk if e in o]
            if srcs:
                out[e] = frozenset().union(*srcs)
            else:
                out[e] = frozenset({self._event}); self._event += 1
        return out

    def _insert(self, adjs, parents=None):
        parents = parents or [[] for _ in adjs]
        for a, par, f in zip(adjs, parents, self.score.score_many(adjs)):
            k = a.astype(np.int8).tobytes()
            if k not in self.origins:
                self.origins[k] = self._birth(a, par)
            self.pool.setdefault(k, f)
            c = self._cell(a)
            if c not in self.archive or f > self.archive[c][1]:
                self.archive[c] = (a, f)

    def run(self, n_generations=60, init_population=300, batch_size=50, seeds=(), immigrant_rate=0.1,
            crossover_rate=0.3, mutation_rate_range=(0.01, 0.2)):
        rng, mask, d = self.rng, self.mask, self.d
        lo, hi = np.log(mutation_rate_range)
        immigrant = lambda: random_dag(d, rng.uniform(0.0, 3.0 / d), rng, mask)
        self._insert([np.asarray(s, np.int8) for s in seeds] + [immigrant() for _ in range(init_population)])
        for _ in range(n_generations):
            cells = list(self.archive)
            elite = lambda: self.archive[cells[rng.integers(len(cells))]][0]
            kids, pars = [], []
            for _ in range(batch_size):
                u = rng.random()
                if u < immigrant_rate:
                    kids.append(immigrant()); pars.append([])
                elif u < immigrant_rate + crossover_rate and len(cells) > 1:
                    a, b = elite(), elite()
                    kids.append(mutate(crossover(a, b, rng, mask), mutation_rate_range[0], rng, mask)); pars.append([a, b])
                else:
                    a = elite()
                    kids.append(mutate(a, float(np.exp(rng.uniform(lo, hi))), rng, mask)); pars.append([a])
            self._insert(kids, pars)
        return self

    def best(self):
        k = max(self.pool, key=self.pool.get)
        return np.frombuffer(k, dtype=np.int8).reshape(self.d, self.d).copy(), self.pool[k]


def noise_band(data, best_adj, candidates, n_reps=5, model="xgb", n_jobs=-1, k=2.0, kind="bootstrap", seed=0,
               **score_kw):
    """k x median SD of the score difference between the best graph and up to 20 candidates,
    across `n_reps` re-scorings: score differences below this cannot be told apart.
        kind="folds"      re-score on different cross-validation splits of the same data
                          (only the split noise; narrow)
        kind="bootstrap"  re-score on bootstrap resamples of the data (finite-sample uncertainty;
                          the relevant notion for "which graphs could have come out best")"""
    cands = candidates[:20]
    if not cands:
        return 0.0
    X = np.asarray(data, float)
    rng = np.random.default_rng(seed)
    diffs = []
    for s in range(n_reps):
        Xs = X[rng.choice(len(X), len(X), replace=True)] if kind == "bootstrap" else X
        S = CVScore(Xs, model=model, random_state=1000 + s, n_jobs=n_jobs, **score_kw)
        f = S.score_many([best_adj] + list(cands))
        diffs.append(f[0] - np.array(f[1:]))
    return float(k * np.median(np.std(np.array(diffs), axis=0)))


def near_optimal(pool, d, band, max_graphs=50):
    """Graphs within `band` of the best score in a pool {bytes: score}, best first."""
    items = sorted(pool.items(), key=lambda kv: -kv[1])
    top = items[0][1]
    out = [(np.frombuffer(k, dtype=np.int8).reshape(d, d).copy(), f) for k, f in items if f >= top - band]
    return out[:max_graphs]
