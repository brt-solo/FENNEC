"""Tests for the graph -> predictor compilers."""
import numpy as np
import pytest

from fennec_search.compilers import COMPILERS, Constrained, children, make_compiler, markov_blanket, parents

# 0 -> 2 <- 1, 2 -> 3 <- 4, 5 isolated; outcome 2
ADJ = np.zeros((6, 6), int)
for i, j in [(0, 2), (1, 2), (2, 3), (4, 3)]:
    ADJ[i, j] = 1


def data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 2] = np.tanh(X[:, 0]) * X[:, 1] + 0.3 * rng.normal(size=n)
    X[:, 3] = X[:, 2] + np.sin(X[:, 4]) + 0.3 * rng.normal(size=n)
    return X


def test_blanket_parts():
    assert parents(ADJ, 2) == [0, 1]
    assert children(ADJ, 2) == [3]
    assert markov_blanket(ADJ, 2) == [0, 1, 3, 4]


def test_undirected_neighbour_counts_as_parent():
    a = ADJ.copy()
    a[5, 2] = a[2, 5] = 1
    assert parents(a, 2) == [0, 1, 5]


def test_interaction_groups():
    g = Constrained(ADJ, 2).interaction_groups()
    assert [0, 1] in g          # the outcome's parent set may interact
    assert [3, 4] in g          # child with its other parent
    assert [5] in g             # isolated feature on its own
    assert not any(2 in grp for grp in g)


@pytest.mark.parametrize("name", list(COMPILERS))
def test_every_compiler_fits_and_ignores_target_column(name):
    X = data()
    tr, te = X[:400], X[400:].copy()
    c = make_compiler(name, ADJ, 2, xgb_params={"n_estimators": 50}).fit(tr)
    p1 = c.predict(te)
    te[:, 2] = np.nan
    p2 = c.predict(te)
    assert p1.shape == (200,) and np.allclose(p1, p2)


def test_empty_graph_gives_mean_predictor():
    X = data()
    c = make_compiler("M1_parents", np.zeros((6, 6), int), 2).fit(X)
    assert np.allclose(c.predict(X), X[:, 2].mean())


def test_parents_beat_wrong_features():
    X = data(n=1500)
    tr, te = X[:1000], X[1000:]
    r2 = lambda c: 1 - np.mean((c.fit(tr).predict(te) - te[:, 2]) ** 2) / te[:, 2].var()
    wrong = np.zeros((6, 6), int)
    wrong[5, 2] = 1
    assert r2(make_compiler("M1_parents", ADJ, 2)) > r2(make_compiler("M1_parents", wrong, 2)) + 0.3
