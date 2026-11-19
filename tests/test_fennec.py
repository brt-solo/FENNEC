"""Fast tests for fennec_search (run: pytest)."""
import numpy as np
import pandas as pd
import pytest

import fennec_search
from fennec_search.graph import crossover, full_mask, is_dag, mutate, random_dag, to_dag
from fennec_search.score import CVScore
from fennec_search.search import MapElites, hill_climb, near_optimal


def chain_data(n=500, seed=0):
    """A -> B -> C with nonlinear mechanisms, so the direction is identifiable."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = np.tanh(2 * a) + 0.3 * rng.normal(size=n)
    c = b ** 2 + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_version_matches_pyproject():
    assert fennec_search.__version__ == "0.1.0"


def test_graph_operators_keep_acyclicity_and_mask():
    rng = np.random.default_rng(1)
    d = 8
    mask = full_mask(d)
    mask[:, 0] = False  # nothing may point into variable 0
    for _ in range(200):
        a = random_dag(d, 0.3, rng, mask)
        b = mutate(a, 0.2, rng, mask)
        c = crossover(a, b, rng, mask)
        for g in (a, b, c):
            assert is_dag(g)
            assert not g[:, 0].any()


def test_to_dag_removes_cycles():
    cyc = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
    assert not is_dag(cyc)
    assert is_dag(to_dag(cyc, np.random.default_rng(0)))


def test_score_prefers_true_graph():
    X = chain_data().to_numpy()
    S = CVScore(X, n_jobs=1)
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8)
    empty = np.zeros((3, 3), dtype=np.int8)
    extra = true.copy()
    extra[0, 2] = 1  # A -> C is redundant given B
    s_true, s_empty, s_extra = S.score_many([true, empty, extra])
    assert s_true > s_empty
    assert s_true > s_extra


def test_penalty_scales_with_sample_size():
    X = chain_data(n=400).to_numpy()
    assert CVScore(X, n_jobs=1).penalty == pytest.approx(4 * np.log(400) / (2 * 400))
    assert CVScore(X, n_jobs=1, penalty_mult=1.0).penalty == pytest.approx(np.log(400) / (2 * 400))


def test_hill_climb_recovers_nonlinear_chain():
    S = CVScore(chain_data().to_numpy(), n_jobs=1)
    g, _ = hill_climb(S)
    assert {(int(i), int(j)) for i, j in zip(*np.nonzero(g))} == {(0, 1), (1, 2)}


def test_map_elites_population():
    S = CVScore(chain_data(n=300).to_numpy(), n_jobs=1)
    best, _ = hill_climb(S)
    me = MapElites(S, seed=0).run(n_generations=3, init_population=20, batch_size=10, seeds=[best])
    pop = near_optimal(me.pool, 3, band=0.05, max_graphs=10)
    assert 1 <= len(pop) <= 10
    assert all(is_dag(g) for g, _ in pop)
    assert np.array_equal(pop[0][0], me.best()[0])


def test_fennec_search_end_to_end():
    df = chain_data(n=300)
    fs = fennec_search.FennecSearch(n_jobs=1, me_generations=3, me_init=20, me_batch=10, max_graphs=5).fit(df)
    assert is_dag(fs.best_graph_)
    assert 1 <= len(fs.population_) <= 5
    eff = fs.effects(df, [("A", 1.0, -1.0)], n=500)
    assert set(eff.outcome) == {"B", "C"}
    assert eff.loc[eff.outcome == "B", "mean"].iloc[0] > 0.5  # tanh(2) - tanh(-2) is about 1.9
