"""Order-leakage diagnostics for simulated data (Reisach et al. 2021, 2023).

varsortability(X, adj): fraction of causally ordered pairs (i an ancestor of j, counted once per
    directed path length) in which the descendant has the larger marginal variance; ties count 1/2.
    0.5 means variance carries no order information; near 1 means sorting by variance recovers the order.
r2_sortability(X, adj): Kendall tau between each variable's R^2 from all others (linear) and its
    causal depth. About 0 means predictability does not leak the order.
"""
import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression


def varsortability(X, adj, tol=1e-9):
    v = np.asarray(X, float).var(0)
    a = (np.asarray(adj) != 0).astype(float)
    d = len(a)
    num = den = 0.0
    Ek = np.eye(d)
    for _ in range(d - 1):
        Ek = (Ek @ a > 0).astype(float)
        i, j = np.nonzero(Ek)
        if len(i) == 0:
            break
        diff = v[j] - v[i]
        num += (diff > tol).sum() + 0.5 * (np.abs(diff) <= tol).sum()
        den += len(i)
    return float(num / den) if den else float("nan")


def causal_depth(adj):
    d = len(adj)
    depth = np.zeros(d)
    for _ in range(d):
        for i, j in zip(*np.nonzero(adj)):
            depth[j] = max(depth[j], depth[i] + 1)
    return depth


def r2_sortability(X, adj):
    X = np.asarray(X, float)
    d = X.shape[1]
    r2 = [LinearRegression().fit(np.delete(X, j, 1), X[:, j]).score(np.delete(X, j, 1), X[:, j]) for j in range(d)]
    return float(kendalltau(r2, causal_depth(adj)).statistic)
