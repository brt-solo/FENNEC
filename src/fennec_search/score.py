"""The FENNEC score: a decomposable, cross-validated likelihood with tree-ensemble regression.

    local(i, Pa) = -1/2 * log( CV-MSE of predicting X_i from X_Pa with XGBoost )   (standardized data)
    score(G)     = sum_i local(i, Pa_G(i)) - penalty * |E|,   penalty = 4 * log(n) / (2n)

The 4x BIC penalty was chosen on an exploratory synthetic benchmark and confirmed on fresh
graphs (benchmarks/README.md; seeds 0-4 exploratory, 5-9 confirmatory): the plain BIC penalty lets XGBoost's small spurious CV gains add
false edges. Each (node, parent set) fit is cached, so any search can call the score freely.
Discrete variables are scored as numeric codes (standardized), like continuous ones.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

XGB_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0, n_jobs=1)


def _cv_mse_xgb(X, y, folds, params):
    err = 0.0
    for tr, va in folds:
        pred = np.full(len(va), y[tr].mean()) if X.shape[1] == 0 else XGBRegressor(**params).fit(X[tr], y[tr]).predict(X[va])
        err += ((y[va] - pred) ** 2).sum()
    return err / len(y)


def _cv_mse_linear(X, y, folds):
    err = 0.0
    for tr, va in folds:
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        err += ((y[va] - np.column_stack([np.ones(len(va)), X[va]]) @ coef) ** 2).sum()
    return err / len(y)


class CVScore:
    def __init__(self, data, model="xgb", n_splits=5, penalty_mult=4.0, penalty=None, random_state=42, n_jobs=-1,
                 xgb_params=None):
        v = np.asarray(data, dtype=float)
        sd = v.std(0)
        sd[sd == 0] = 1.0
        self.values = (v - v.mean(0)) / sd
        self.n, self.d = self.values.shape
        self.model = model
        self.penalty = penalty_mult * np.log(self.n) / (2 * self.n) if penalty is None else penalty
        self.folds = list(KFold(n_splits, shuffle=True, random_state=random_state).split(self.values))
        self.n_jobs = n_jobs
        self.xgb_params = dict(XGB_PARAMS, **(xgb_params or {}))
        self.cache = {}
        self.n_fits = 0

    def _compute(self, keys):
        keys = [k for k in dict.fromkeys(keys) if k not in self.cache]
        if not keys:
            return
        V = self.values
        if self.model == "linear":
            mses = [_cv_mse_linear(V[:, list(pa)], V[:, i], self.folds) for i, pa in keys]
        elif self.n_jobs == 1 or len(keys) == 1:
            mses = [_cv_mse_xgb(V[:, list(pa)], V[:, i], self.folds, self.xgb_params) for i, pa in keys]
        else:
            mses = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_cv_mse_xgb)(V[:, list(pa)], V[:, i], self.folds, self.xgb_params) for i, pa in keys)
        for k, mse in zip(keys, mses):
            self.cache[k] = -0.5 * np.log(max(mse, 1e-12))
        self.n_fits += len(keys) * len(self.folds)

    @staticmethod
    def _keys(adj):
        return [(i, tuple(int(p) for p in np.flatnonzero(adj[:, i]))) for i in range(adj.shape[0])]

    def local(self, i, parents):
        k = (i, tuple(sorted(int(p) for p in parents)))
        self._compute([k])
        return self.cache[k]

    def score_many(self, adjs):
        adjs = [np.asarray(a) for a in adjs]
        self._compute([k for a in adjs for k in self._keys(a)])
        return [sum(self.cache[k] for k in self._keys(a)) - self.penalty * int(a.sum()) for a in adjs]

    def score(self, adj):
        return self.score_many([adj])[0]
