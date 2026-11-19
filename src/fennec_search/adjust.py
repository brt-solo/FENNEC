"""Effect estimation by the backdoor adjustment formula, with cross-fitting.

For an exposure T and outcome Y in a DAG, the parents of T block every backdoor path, so
    E[Y | do(T = t)] = E_Z[ mu(t, Z) ],   mu(t, z) = E[Y | T = t, Z = z],   Z = Pa(T)
mu is learned by XGBoost with K-fold cross-fitting (each row is predicted by a model that did
not see it), which limits overfitting bias. If Y is not a descendant of T in the graph, the
effect is 0 by the graph. Compared with simulating the whole causal model, only ONE model is
fitted per effect, so errors do not compound along long causal chains.
"""
import networkx as nx
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from .score import XGB_PARAMS


def backdoor_effect(X, adj, t, y, a, b, xgb_params=None, folds=5, seed=0):
    """E[X_y | do(X_t = a)] - E[X_y | do(X_t = b)] for column indices t, y of array X."""
    X = np.asarray(X, float)
    G = nx.DiGraph([(int(i), int(j)) for i, j in zip(*np.nonzero(adj))])
    G.add_nodes_from(range(X.shape[1]))
    if y not in nx.descendants(G, t):
        return 0.0
    Z = sorted(G.predecessors(t))
    F = X[:, [t] + Z]
    target = X[:, y]
    pred_a, pred_b = np.zeros(len(X)), np.zeros(len(X))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = XGBRegressor(**(xgb_params or XGB_PARAMS)).fit(F[tr], target[tr])
        Fa, Fb = F[te].copy(), F[te].copy()
        Fa[:, 0], Fb[:, 0] = a, b
        pred_a[te], pred_b[te] = m.predict(Fa), m.predict(Fb)
    return float(np.mean(pred_a - pred_b))
