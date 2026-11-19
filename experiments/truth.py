"""Ground-truth systems with fixed mechanisms, mixed variable types and exact interventions.

Data units: continuous variables are standardized with the observational mean/SD (fixed once
from a large calibration sample); discrete variables are integer codes (0/1 or 0/1/2).
Children receive their parents in data units, so do(X = v) is specified in the same units the
methods see. True effects are computed by Monte Carlo with common random numbers.

Mechanisms (as in the structure benchmark): linear, additive (random nonlinear function per parent),
mlp (random tanh network of all parents: nonlinear with interactions), mixed (mlp, with 40% of
nodes discretised into 2 or 3 ordered levels by fixed quantile thresholds).
"""
import networkx as nx
import numpy as np

NOISE_SD = 0.5


def _random_dag(d, rng, expected_degree=2.0):
    order = rng.permutation(d)
    p = min(1.0, expected_degree / (d - 1))
    adj = np.zeros((d, d), dtype=np.int8)
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < p:
                adj[order[a], order[b]] = 1
    return adj, order


def _rand_fn(rng):
    kind = rng.integers(5)
    a = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    if kind == 0:
        return lambda x: np.tanh(a * x)
    if kind == 1:
        return lambda x: np.sin(a * x)
    if kind == 2:
        return lambda x: a * (x ** 2 - 1)
    if kind == 3:
        return lambda x: a * (np.abs(x) - 0.8)
    c = rng.normal()
    return lambda x: a * np.exp(-((x - c) ** 2))


class TrueSystem:
    def __init__(self, d, mechanism, seed, n_calib=100_000):
        rng = np.random.default_rng(seed)
        self.d, self.mechanism = d, mechanism
        self.adj, self.order = _random_dag(d, rng)
        self.discrete = {}
        if mechanism == "mixed":
            for j in rng.choice(d, size=int(round(0.4 * d)), replace=False):
                self.discrete[int(j)] = int(rng.choice([2, 3]))
        self.mech = {}
        for j in self.order:
            pa = np.flatnonzero(self.adj[:, j])
            if len(pa) == 0:
                self.mech[j] = None
            elif mechanism == "linear":
                self.mech[j] = ("lin", pa, rng.uniform(0.5, 2.0, len(pa)) * rng.choice([-1, 1], len(pa)))
            elif mechanism == "additive":
                self.mech[j] = ("add", pa, [_rand_fn(rng) for _ in pa])
            else:
                W1 = rng.uniform(0.5, 2.0, (len(pa), 10)) * rng.choice([-1, 1], (len(pa), 10))
                W2 = rng.uniform(0.5, 2.0, 10) * rng.choice([-1, 1], 10)
                self.mech[j] = ("mlp", pa, (W1, W2))
        self.G = nx.DiGraph([(int(i), int(j)) for i, j in zip(*np.nonzero(self.adj))])
        self.G.add_nodes_from(range(d))
        self._calibrate(n_calib, np.random.default_rng(seed + 7_000_000))

    def _signal(self, j, X):
        kind, pa, p = self.mech[j]
        if kind == "lin":
            return X[:, pa] @ p
        if kind == "add":
            return sum(f(X[:, k]) for f, k in zip(p, pa))
        return np.tanh(X[:, pa] @ p[0]) @ p[1]

    def _calibrate(self, n, rng):
        self.sig, self.std, self.cuts = {}, {}, {}
        X = np.zeros((n, self.d))
        for j in self.order:
            if self.mech[j] is None:
                raw = rng.normal(0, 1, n)
            else:
                f = self._signal(j, X)
                self.sig[j] = (f.mean(), f.std() if f.std() > 1e-12 else 1.0)
                raw = (f - self.sig[j][0]) / self.sig[j][1] + rng.normal(0, NOISE_SD, n)
            if j in self.discrete:
                k = self.discrete[j]
                self.cuts[j] = np.quantile(raw, np.linspace(0, 1, k + 1)[1:-1])
                X[:, j] = np.digitize(raw, self.cuts[j])
            else:
                self.std[j] = (raw.mean(), raw.std())
                X[:, j] = (raw - self.std[j][0]) / self.std[j][1]

    def sample(self, n, seed, do=None):
        """n samples in data units; do = {node: value in data units}."""
        rng = np.random.default_rng(seed)
        do = do or {}
        X = np.zeros((n, self.d))
        for j in self.order:  # draw every node's noise even if intervened (common random numbers)
            if self.mech[j] is None:
                raw = rng.normal(0, 1, n)
            else:
                f = self._signal(j, X)
                raw = (f - self.sig[j][0]) / self.sig[j][1] + rng.normal(0, NOISE_SD, n)
            if j in do:
                X[:, j] = do[j]
            elif j in self.discrete:
                X[:, j] = np.digitize(raw, self.cuts[j])
            else:
                X[:, j] = (raw - self.std[j][0]) / self.std[j][1]
        return X

    def true_effects(self, target, a, b, n=200_000, seed=12345):
        return self.sample(n, seed, {target: a}).mean(0) - self.sample(n, seed, {target: b}).mean(0)

    def levels(self, j):
        """(a, b) contrast for an intervention on node j: top vs bottom level, or +1 vs -1 SD."""
        if j in self.discrete:
            return float(self.discrete[j] - 1), 0.0
        return 1.0, -1.0
