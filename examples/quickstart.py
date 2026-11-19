"""Quickstart: learn a population of graphs on a small nonlinear system and query a causal effect.

    python examples/quickstart.py      (about 15 seconds on 4 cores)
"""
import numpy as np
import pandas as pd

from fennec_search import FennecSearch

# A -> B -> C, A -> D, with nonlinear mechanisms
rng = np.random.default_rng(0)
n = 1000
a = rng.normal(size=n)
b = np.tanh(2 * a) + 0.3 * rng.normal(size=n)
c = b ** 2 + 0.3 * rng.normal(size=n)
d = np.sin(2 * a) + 0.3 * rng.normal(size=n)
df = pd.DataFrame({"A": a, "B": b, "C": c, "D": d})

fs = FennecSearch(n_jobs=4, me_generations=10, me_init=50, me_batch=20, max_graphs=10).fit(df)

edges = [f"{df.columns[i]} -> {df.columns[j]}" for i, j in zip(*np.nonzero(fs.best_graph_))]
print("best graph:", ", ".join(edges))
print("graphs in the near-optimal population:", len(fs.population_))

# E[Y | do(A = 1)] - E[Y | do(A = -1)] for every other variable, averaged over the population
eff = fs.effects(df, [("A", 1.0, -1.0)], n=2000)
print(eff[["outcome", "mean", "lo", "hi", "p_ancestor"]].round(2).to_string(index=False))
