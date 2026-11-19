"""FENNEC-Search: Fitness-guided Evolution of Niched Networks for Ensemble Causal discovery.

Typical use:
    from fennec_search import FennecSearch
    fs = FennecSearch().fit(df)                     # learn best graph + near-optimal population
    effects = fs.effects(df, [("X3", 1.0, -1.0)])   # E[Y|do(X3=1)] - E[Y|do(X3=-1)] for every Y,
                                                     # averaged over the population, with intervals
"""
import numpy as np

from .adjust import backdoor_effect
from .causal_model import XGB_BIG, build_causal_model, effects, infer_types
from .ensemble import GraphEnsemble
from .graph import full_mask, to_dag
from .score import CVScore
from .search import MapElites, hill_climb, near_optimal, noise_band

try:  # single source of truth: the version in pyproject.toml
    from importlib.metadata import PackageNotFoundError, version as _version
    __version__ = _version("fennec-search")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.1.0"


class FennecSearch:
    """Learn a population of causal graphs and answer effect queries across it.

    fit(): hill climbing for the best graph, MAP-Elites seeded with it for the population, then
    the near-optimal set = all evaluated graphs within the score's noise band of the best, where
    the band is the bootstrap variability of score differences (population_), or only the
    cross-validation-split variability (population_folds_, narrower).
    """

    def __init__(self, model="xgb", penalty_mult=4.0, mask=None, n_jobs=-1, seed=0, me_generations=60,
                 me_init=300, me_batch=50, max_graphs=50):
        self.model, self.penalty_mult, self.mask = model, penalty_mult, mask
        self.n_jobs, self.seed = n_jobs, seed
        self.me = dict(n_generations=me_generations, init_population=me_init, batch_size=me_batch)
        self.max_graphs = max_graphs

    def fit(self, df):
        X = np.asarray(df, float)
        self.columns = list(df.columns)
        self.n = len(X)
        self.score_ = CVScore(X, model=self.model, penalty_mult=self.penalty_mult, n_jobs=self.n_jobs)
        self.best_graph_, self.best_score_ = hill_climb(self.score_, self.mask)
        me = MapElites(self.score_, self.mask, seed=self.seed).run(seeds=[self.best_graph_], **self.me)
        self.pool_ = me.pool
        cands = [np.frombuffer(k, dtype=np.int8).reshape(X.shape[1], -1) for k, _ in
                 sorted(me.pool.items(), key=lambda kv: -kv[1])[1:200]]
        kw = dict(model=self.model, n_jobs=self.n_jobs, penalty_mult=self.penalty_mult, seed=self.seed)
        self.band_folds_ = noise_band(X, self.best_graph_, cands[::10], kind="folds", **kw)
        self.band_ = noise_band(X, self.best_graph_, cands[::10], kind="bootstrap", **kw)
        self.population_ = near_optimal(me.pool, X.shape[1], self.band_, self.max_graphs)
        self.population_folds_ = near_optimal(me.pool, X.shape[1], self.band_folds_, self.max_graphs)
        return self

    def ensemble(self, weighting="uniform"):
        graphs, scores = zip(*self.population_)
        return GraphEnsemble(graphs, scores, n_obs=self.n, weighting=weighting)

    def effects(self, df, queries, weighting="uniform", n_jobs=1, **kw):
        return self.ensemble(weighting).query(df, queries, n_jobs=n_jobs, **kw)
