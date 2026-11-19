"""Synthetic benchmark data: random DAGs with controlled mechanisms.

simulate(d, n, mechanism, seed) -> (data DataFrame, true adjacency, info)

Graph: Erdős–Rényi DAG over a random node order, expected degree 2 (about d edges).
Each node X_j = f_j(parents) + noise, generated in topological order:
    linear       f = sum_k w_k * x_k,                  |w| ~ U[0.5, 2], random sign
    additive     f = sum_k g_k(x_k), g_k a random nonlinearity (tanh, sin, square, abs, sigmoid
                 bump) with random scale and sign
    mlp          f = random 1-hidden-layer tanh network of the parents (10 units): nonlinear and
                 non-additive (interactions)
    mixed        as mlp, then 40% of nodes are discretised (half binary, half 3-level ordinal)
                 *before* their children are generated, so children see the discrete value
The signal f is standardized and noise N(0, 0.5^2) added (signal-to-noise variance ratio 4).
Finally every column is standardized, so marginal variance carries no information about the
causal order (avoids the "varsortability" artefact of naive simulations).
"""
import numpy as np
import pandas as pd

NOISE_SD = 0.5
MECHANISMS = ["linear", "additive", "mlp", "mixed"]


def random_dag(d, rng, expected_degree=2.0):
    order = rng.permutation(d)
    p = min(1.0, expected_degree / (d - 1))
    adj = np.zeros((d, d), dtype=int)
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < p:
                adj[order[a], order[b]] = 1
    return adj, order


def _std(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x - x.mean()


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


def simulate(d, n, mechanism, seed, noise_sd=NOISE_SD):
    rng = np.random.default_rng(seed)
    adj, order = random_dag(d, rng)
    X = np.zeros((n, d))
    discrete = set()
    if mechanism == "mixed":
        discrete = set(rng.choice(d, size=int(round(0.4 * d)), replace=False).tolist())
    for j in order:
        pa = np.flatnonzero(adj[:, j])
        if len(pa) == 0:
            f = np.zeros(n)
        elif mechanism == "linear":
            w = rng.uniform(0.5, 2.0, len(pa)) * rng.choice([-1, 1], len(pa))
            f = X[:, pa] @ w
        elif mechanism == "additive":
            f = sum(_rand_fn(rng)(X[:, k]) for k in pa)
        else:  # mlp / mixed
            W1 = rng.uniform(0.5, 2.0, (len(pa), 10)) * rng.choice([-1, 1], (len(pa), 10))
            W2 = rng.uniform(0.5, 2.0, 10) * rng.choice([-1, 1], 10)
            f = np.tanh(X[:, pa] @ W1) @ W2
        x = (_std(f) if len(pa) else f) + rng.normal(0, noise_sd if len(pa) else 1.0, n)
        if j in discrete:
            if rng.random() < 0.5:
                x = (x > np.median(x)).astype(float)
            else:
                x = np.digitize(x, np.quantile(x, [1 / 3, 2 / 3])).astype(float)
        X[:, j] = _std(x)
    cols = [f"X{i}" for i in range(d)]
    info = {"d": d, "n": n, "mechanism": mechanism, "seed": seed, "n_edges": int(adj.sum()),
            "discrete": sorted(discrete)}
    return pd.DataFrame(X, columns=cols), adj, info


if __name__ == "__main__":
    for m in MECHANISMS:
        df, adj, info = simulate(10, 500, m, 0)
        print(m, info["n_edges"], "edges", df.shape, df.std().round(2).tolist()[:4], "discrete", info["discrete"])
