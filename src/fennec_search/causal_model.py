"""From a DAG to a fitted causal model (DoWhy-GCM) that answers interventional queries.

Mechanisms per node:
    no parents          empirical distribution of the observed values
    continuous node     additive noise model, XGBoost regressor of the parents (same settings as the score)
    categorical node    classifier mechanism, histogram gradient-boosting classifier of the parents
A column is treated as categorical if it has <= `max_levels` distinct values (override with `types`).
DoWhy-GCM needs categorical values as strings: they are converted to str(float(value)) for
fitting and sampling, and back to floats in every returned quantity.

effect(model, target, outcome, a, b) = E[outcome | do(target = a)] - E[outcome | do(target = b)],
estimated by Monte Carlo sampling from the intervened model.
"""
import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBRegressor

from .score import XGB_PARAMS

gcm.config.disable_progress_bars()


def _cat(v):
    return str(float(v))


def infer_types(df, max_levels=5):
    return {c: ("categorical" if df[c].nunique() <= max_levels else "continuous") for c in df.columns}


XGB_BIG = dict(XGB_PARAMS, n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8)


def _regressor(params=None):
    return gcm.ml.SklearnRegressionModel(XGBRegressor(**(params or XGB_PARAMS)))


def _classifier():
    return gcm.ml.SklearnClassificationModel(HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1,
                                                                            random_state=42))


def build_causal_model(adj, df, types=None, seed=0, xgb_params=None):
    """Fit a DoWhy-GCM StructuralCausalModel for the DAG `adj` over the columns of `df`."""
    cols = list(df.columns)
    types = types or infer_types(df)
    g = nx.DiGraph()
    g.add_nodes_from(cols)
    g.add_edges_from((cols[i], cols[j]) for i, j in zip(*np.nonzero(adj)))
    model = gcm.StructuralCausalModel(g)
    for c in cols:
        if g.in_degree(c) == 0:
            model.set_causal_mechanism(c, gcm.EmpiricalDistribution())
        elif types[c] == "categorical":
            model.set_causal_mechanism(c, gcm.ClassifierFCM(_classifier()))
        else:
            model.set_causal_mechanism(c, gcm.AdditiveNoiseModel(_regressor(xgb_params)))
    data = df.copy()
    for c in cols:
        if types[c] == "categorical":
            data[c] = data[c].astype(float).map(_cat)
    np.random.seed(seed)
    gcm.fit(model, data)
    model._fennec_types = types
    return model


def intervened_means(model, target, values, n=4000, seed=0):
    """{value: mean of every variable under do(target = value)}."""
    types = model._fennec_types
    out = {}
    for v in values:
        val = _cat(v) if types[target] == "categorical" else float(v)
        np.random.seed(seed)
        s = gcm.interventional_samples(model, {target: (lambda x, val=val: val)}, num_samples_to_draw=n)
        out[v] = s.astype(float).mean()
    return out


def effects(model, target, a, b, n=4000, seed=0):
    """E[X | do(target=a)] - E[X | do(target=b)] for every variable X (a pandas Series)."""
    m = intervened_means(model, target, [a, b], n=n, seed=seed)
    return m[a] - m[b]
