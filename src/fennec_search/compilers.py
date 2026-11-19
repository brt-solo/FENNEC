"""Graph -> predictor compilers: turn a causal graph hypothesis into a fitted predictor of one outcome.

Every compiler has the same interface:

    c = make_compiler("M1_blanket", adj, outcome=t, seed=0)
    c.fit(X_train)            # X: (n x d) array of ALL variables; column t is the target
    yhat = c.predict(X_test)  # column t of X_test is ignored (may be NaN)

adj[i, j] = 1 means i -> j. If adj[i, j] = adj[j, i] = 1 the edge is undirected (CPDAG output from
PC/GES); undirected neighbours of the outcome are treated as parents, and as neighbours elsewhere.

Compilers (PROJECT_CONTEXT.md section 4):
    M0_all              XGBoost on all other variables (ignores the graph; the baseline)
    M1_parents          XGBoost on the outcome's parents only
    M1_blanket          XGBoost on the outcome's Markov blanket (parents, children, children's other parents)
    M2_constrained      XGBoost on all other variables with interaction_constraints: features may only
                        interact inside a clique of the moral graph around the outcome (its parent set;
                        each child with its other parents) or along a graph edge. Optional monotone
                        constraints from edge signs (signs={feature: +1/-1}).
    M3_chain            causal chain: one model per parent p of the outcome, predicting p from ITS parents;
                        the outcome model is trained on out-of-fold predictions of its parents (roots enter
                        as observed). Tree analogue of a neural causal model.
    M3_chain_stacked    as M3_chain but the outcome model sees the observed parents AND their predictions.

All compilers share the same XGBoost settings (XGB_DEFAULT unless xgb_params is given), so differences
come from the graph, not from tuning. Nothing here looks at test data.
"""
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

XGB_DEFAULT = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, min_child_weight=1.0,
                   reg_lambda=1.0, tree_method="hist", n_jobs=1)


def _split(adj):
    a = (np.asarray(adj) != 0)
    np.fill_diagonal(a, False)
    return a & ~a.T, a & a.T  # directed, undirected


def parents(adj, t):
    directed, undirected = _split(adj)
    return sorted(set(np.flatnonzero(directed[:, t])) | set(np.flatnonzero(undirected[:, t])))


def children(adj, t):
    directed, _ = _split(adj)
    return sorted(np.flatnonzero(directed[t]).tolist())


def markov_blanket(adj, t):
    directed, _ = _split(adj)
    ch = children(adj, t)
    co = set(np.flatnonzero(directed[:, ch].any(1))) if ch else set()
    return sorted((set(parents(adj, t)) | set(ch) | co) - {t})


class _Mean:
    def fit(self, X, y):
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


class Compiler:
    name = "base"

    def __init__(self, adj, outcome, xgb_params=None, seed=0):
        self.adj = np.asarray(adj) if adj is not None else None
        self.t = int(outcome)
        self.params = {**XGB_DEFAULT, **(xgb_params or {}), "random_state": seed}
        self.seed = seed

    def _xgb(self, **extra):
        return XGBRegressor(**self.params, **extra)

    def features(self, d):
        raise NotImplementedError

    def fit(self, X):
        X = np.asarray(X, float)
        self.feats_ = [int(f) for f in self.features(X.shape[1])]
        y = X[:, self.t]
        self.model_ = (self._make_model() if self.feats_ else _Mean()).fit(X[:, self.feats_], y)
        return self

    def _make_model(self):
        return self._xgb()

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float)[:, self.feats_])


class AllFeatures(Compiler):
    name = "M0_all"

    def features(self, d):
        return [j for j in range(d) if j != self.t]


class Parents(Compiler):
    name = "M1_parents"

    def features(self, d):
        return parents(self.adj, self.t)


class Blanket(Compiler):
    name = "M1_blanket"

    def features(self, d):
        return markov_blanket(self.adj, self.t)


class Constrained(Compiler):
    name = "M2_constrained"

    def __init__(self, adj, outcome, xgb_params=None, seed=0, signs=None):
        super().__init__(adj, outcome, xgb_params, seed)
        self.signs = signs or {}

    def features(self, d):
        return [j for j in range(d) if j != self.t]

    def interaction_groups(self):
        """Feature-index groups (original variable indices) allowed to interact."""
        directed, undirected = _split(self.adj)
        t = self.t
        groups = []
        pa = parents(self.adj, t)
        if pa:
            groups.append(pa)
        for c in children(self.adj, t):
            groups.append(sorted({c} | (set(parents(self.adj, c)) - {t})))
        a = directed | undirected
        for i, j in zip(*np.nonzero(np.triu(a | a.T))):
            if t not in (i, j):
                groups.append([int(i), int(j)])
        covered = {f for g in groups for f in g}
        groups += [[f] for f in self.features(len(self.adj)) if f not in covered]
        uniq = {tuple(sorted(g)) for g in groups}
        return [list(g) for g in sorted(uniq)]

    def _make_model(self):
        pos = {f: k for k, f in enumerate(self.feats_)}
        ic = [[pos[f] for f in g] for g in self.interaction_groups()]
        extra = {"interaction_constraints": str(ic)}  # string form: xgboost 3 maps int lists via feature names
        if self.signs:
            extra["monotone_constraints"] = "(" + ",".join(str(int(self.signs.get(f, 0))) for f in self.feats_) + ")"
        return self._xgb(**extra)


class CausalChain(Compiler):
    name = "M3_chain"
    stacked = False

    def __init__(self, adj, outcome, xgb_params=None, seed=0, n_folds=5):
        super().__init__(adj, outcome, xgb_params, seed)
        self.n_folds = n_folds

    def fit(self, X):
        X = np.asarray(X, float)
        y = X[:, self.t]
        self.pa_ = parents(self.adj, self.t)
        # parents' own parents, excluding the outcome (an undirected edge could otherwise point back)
        self.papa_ = {p: [q for q in parents(self.adj, p) if q != self.t] for p in self.pa_}
        self.sub_ = {}
        oof = {}
        for p in self.pa_:
            q = self.papa_[p]
            if not q:
                continue
            o = np.zeros(len(X))
            for tr, te in KFold(self.n_folds, shuffle=True, random_state=self.seed).split(X):
                o[te] = self._xgb().fit(X[np.ix_(tr, q)], X[tr, p]).predict(X[np.ix_(te, q)])
            oof[p] = o
            self.sub_[p] = self._xgb().fit(X[:, q], X[:, p])
        Z = self._design(X, oof)
        self.model_ = (self._xgb() if Z.shape[1] else _Mean()).fit(Z, y)
        return self

    def _design(self, X, pred):
        cols = []
        for p in self.pa_:
            if p in pred:
                cols.append(pred[p])
                if self.stacked:
                    cols.append(X[:, p])
            else:  # a root parent: no model for it, enters as observed
                cols.append(X[:, p])
        return np.column_stack(cols) if cols else np.zeros((len(X), 0))

    def predict(self, X):
        X = np.asarray(X, float)
        pred = {p: m.predict(X[:, self.papa_[p]]) for p, m in self.sub_.items()}
        return self.model_.predict(self._design(X, pred))


class CausalChainStacked(CausalChain):
    name = "M3_chain_stacked"
    stacked = True


COMPILERS = {c.name: c for c in (AllFeatures, Parents, Blanket, Constrained, CausalChain, CausalChainStacked)}


def make_compiler(name, adj, outcome, **kw):
    return COMPILERS[name](adj, outcome, **kw)
