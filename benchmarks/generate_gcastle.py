"""Outside benchmark generator (E0): gCastle's IIDSimulation, which we did not write.

simulate_gcastle(d, n, mechanism, seed, graph="ER", edges_per_node=2, standardize=True)
    -> (data DataFrame, true adjacency, info)

Graphs: Erdős–Rényi or scale-free, passing n_edges = edges_per_node * d to gCastle ("ER2"/"SF2";
the realised ER edge count is lower, about 1.2 d at d = 10).
Mechanisms (gCastle nonlinear SEMs, Gaussian noise of scale 1):
    mlp        random 1-hidden-layer sigmoid network of the parents (non-additive)
    mim        multiple index model (sums of tanh/cos/sin of random projections)
    gp         a draw from a Gaussian process in all parents jointly
    quadratic  quadratic polynomial of the parents (interactions included)
Raw gCastle data are NOT standardized and are typically varsortable; info records varsortability
before standardization. With standardize=True (default) every column is z-scored afterwards,
as in the main benchmark.
"""
import numpy as np
import pandas as pd

from benchmarks.sortability import varsortability

GC_MECHANISMS = ["mlp", "mim", "gp", "quadratic"]


def simulate_gcastle(d, n, mechanism, seed, graph="ER", edges_per_node=2, standardize=True):
    from castle.datasets import DAG, IIDSimulation
    np.random.seed(seed)  # gCastle draws from the global numpy RNG
    make = DAG.erdos_renyi if graph == "ER" else DAG.scale_free
    W = make(n_nodes=d, n_edges=edges_per_node * d, weight_range=(0.5, 2.0), seed=seed)
    sim = IIDSimulation(W=W, n=n, method="nonlinear", sem_type=mechanism, noise_scale=1.0)
    X = np.asarray(sim.X, float)
    adj = (np.asarray(sim.B) != 0).astype(int)
    vs_raw = varsortability(X, adj)
    if standardize:
        X = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    info = {"d": d, "n": n, "mechanism": mechanism, "seed": seed, "graph": graph, "n_edges": int(adj.sum()),
            "varsortability_raw": vs_raw, "standardized": standardize}
    return pd.DataFrame(X, columns=[f"X{i}" for i in range(d)]), adj, info


if __name__ == "__main__":
    for g in ("ER", "SF"):
        for m in GC_MECHANISMS:
            df, adj, info = simulate_gcastle(10, 500, m, 0, graph=g)
            print(g, m, info["n_edges"], "edges; raw varsortability", round(info["varsortability_raw"], 2),
                  "; after standardizing", round(varsortability(df.to_numpy(), adj), 2))
