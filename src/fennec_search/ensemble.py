"""Causal answers averaged over a population of graphs (structural uncertainty).

GraphEnsemble(graphs, scores) fits one causal model per graph and answers effect queries
under each. Weights over graphs:
    "uniform"  equal weight (e.g. all graphs within the score's noise band)
    "bma"      w_g ∝ exp(n * (score_g - max score)): the score is a per-sample log-likelihood,
               so n * score approximates each graph's log evidence (Bayesian model averaging)
The answer is the weighted mean, with a weighted (lo, hi) quantile interval across graphs and
the weighted share of graphs in which the target is an ancestor of the outcome at all.
"""
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .causal_model import build_causal_model, effects, infer_types


def _weighted_quantile(x, w, q):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c - w / (2 * w.sum()), x))


def _fit_and_query(adj, df, types, queries, n, seed):
    m = build_causal_model(adj, df, types, seed=seed)
    return {(t, a, b): effects(m, t, a, b, n=n, seed=seed) for t, a, b in queries}


class GraphEnsemble:
    def __init__(self, graphs, scores=None, n_obs=None, weighting="uniform"):
        self.graphs = [np.asarray(g) for g in graphs]
        s = np.zeros(len(graphs)) if scores is None else np.asarray(scores, float)
        if weighting == "bma":
            if n_obs is None:
                raise ValueError("bma weighting needs n_obs")
            w = np.exp(n_obs * (s - s.max()))
        else:
            w = np.ones(len(graphs))
        self.weights = w / w.sum()

    def query(self, df, queries, types=None, n=4000, n_jobs=1, seed=0):
        """queries: list of (target, a, b). Returns a long DataFrame:
        target, a, b, outcome, mean, lo, hi, p_ancestor, n_graphs, per-graph estimates."""
        types = types or infer_types(df)
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_and_query)(g, df, types, queries, n, seed) for g in self.graphs) if n_jobs != 1 else \
            [_fit_and_query(g, df, types, queries, n, seed) for g in self.graphs]
        cols = list(df.columns)
        anc = []
        for g in self.graphs:
            G = nx.DiGraph([(cols[i], cols[j]) for i, j in zip(*np.nonzero(g))]); G.add_nodes_from(cols)
            anc.append({c: nx.descendants(G, c) for c in cols})
        rows = []
        for (t, a, b) in queries:
            E = pd.DataFrame([r[(t, a, b)] for r in res])  # graphs x variables
            for y in cols:
                if y == t:
                    continue
                v = E[y].to_numpy()
                rows.append({"target": t, "a": a, "b": b, "outcome": y,
                             "mean": float(np.dot(self.weights, v)),
                             "lo": _weighted_quantile(v, self.weights, 0.05),
                             "hi": _weighted_quantile(v, self.weights, 0.95),
                             "p_ancestor": float(sum(w for w, an in zip(self.weights, anc) if y in an[t])),
                             "n_graphs": len(self.graphs), "per_graph": v.tolist()})
        return pd.DataFrame(rows)
