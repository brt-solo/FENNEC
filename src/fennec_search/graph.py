"""DAG utilities: acyclicity, repair, random graphs, variation operators.

Graphs are (d x d) 0/1 numpy arrays, adj[i, j] = 1 meaning i -> j. An optional boolean
`mask` restricts which edges may ever exist (background knowledge, e.g. time ordering).
"""
import networkx as nx
import numpy as np


def full_mask(d):
    return ~np.eye(d, dtype=bool)


def is_dag(adj):
    return nx.is_directed_acyclic_graph(nx.from_numpy_array(np.asarray(adj), create_using=nx.DiGraph))


def edges(adj):
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(adj))]


def repair(adj, rng):
    """Delete a random edge from each remaining cycle until the graph is acyclic."""
    a = np.asarray(adj).astype(np.int8).copy()
    g = nx.from_numpy_array(a, create_using=nx.DiGraph)
    while True:
        try:
            cyc = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            return a
        u, v = cyc[rng.integers(len(cyc))][:2]
        g.remove_edge(u, v)
        a[u, v] = 0


def random_dag(d, edge_prob, rng, mask=None):
    mask = full_mask(d) if mask is None else mask
    a = ((rng.random((d, d)) < edge_prob) & mask).astype(np.int8)
    return repair(a, rng)


def mutate(adj, rate, rng, mask=None):
    mask = full_mask(len(adj)) if mask is None else mask
    flip = (rng.random(adj.shape) < rate) & mask
    return repair(np.where(flip, 1 - adj, adj), rng)


def crossover(a, b, rng, mask=None):
    take_b = rng.random(a.shape) < 0.5
    child = np.where(take_b, b, a)
    if mask is not None:
        child = child & mask
    return repair(child, rng)


def to_dag(adj, rng=None):
    """Orient undirected (two-way) edges consistently with the directed part (CPDAG -> a member DAG)."""
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.asarray(adj).astype(int).copy()
    d = len(a)
    und = [(i, j) for i in range(d) for j in range(i + 1, d) if a[i, j] and a[j, i]]
    for i, j in und:
        a[i, j] = a[j, i] = 0
    g = nx.DiGraph()
    g.add_nodes_from(range(d))
    g.add_edges_from(edges(a))
    while not nx.is_directed_acyclic_graph(g):
        u, v = nx.find_cycle(g)[0][:2]
        g.remove_edge(u, v)
        a[u, v] = 0
    order = {n: k for k, n in enumerate(nx.lexicographical_topological_sort(g, key=lambda n: rng.random()))}
    for i, j in und:
        if order[i] < order[j]:
            a[i, j] = 1
        else:
            a[j, i] = 1
    return a.astype(np.int8)
