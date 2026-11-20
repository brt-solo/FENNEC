"""Outcome-local FENNEC search: choose the parent set of one target with the FENNEC score.

When the target is measured after every candidate (forecasting), its Markov blanket is its parent
set, so only the target's local term of the score matters:
    local(y, Pa) = -1/2 log CV-MSE(y | Pa)  -  penalty * |Pa|,   penalty = penalty_mult * log(n) / (2n)
Greedy forward selection (add the candidate that raises the score most, while it rises), then
backward elimination. With many candidates, screen first (screen=k keeps the union of the top k by
absolute correlation and the top k by XGBoost gain importance) so the search stays cheap.

    pa = select_parents(X, y, penalty_mult=4.0, screen=40)
"""
import numpy as np
from xgboost import XGBRegressor

from .score import CVScore


def screen_candidates(X, y, k, seed=0):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    corr = np.abs(Xs.T @ ((y - y.mean()) / (y.std() + 1e-12))) / len(y)
    imp = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, n_jobs=1,
                       random_state=seed, importance_type="gain").fit(X, y).feature_importances_
    return sorted(set(np.argsort(-corr)[:k].tolist()) | set(np.argsort(-imp)[:k].tolist()))


def select_parents(X, y, candidates=None, penalty_mult=4.0, screen=40, max_parents=30, n_jobs=1, seed=0,
                   score=None, return_score=False, extra_penalty=None, init=None, model="linear", xgb_params=None,
                   backward=True):
    """Indices (into X's columns) of the selected parents of y.

    score=          reuse a CVScore cache across settings (built on np.column_stack([X, y]))
    extra_penalty=  {feature: added penalty} on top of the BIC-type penalty (e.g. tier-wise EBIC)
    init=           start the forward search from this set (tier-sequential search)
    model=          "linear" (default; suited to many small effects) or "xgb"
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    p = X.shape[1]
    if candidates is None:
        candidates = screen_candidates(X, y, screen, seed) if screen and screen < p else list(range(p))
    S = score or CVScore(np.column_stack([X, y]), model=model, penalty_mult=penalty_mult, n_jobs=n_jobs,
                         xgb_params=xgb_params)
    pen = penalty_mult * np.log(S.n) / (2 * S.n)
    extra = extra_penalty or {}
    t = p  # target column in S
    loc = lambda pa: S.cache[(t, tuple(sorted(pa)))] - pen * len(pa) - sum(extra.get(c, 0.0) for c in pa)

    def eval_many(sets):
        S._compute([(t, tuple(sorted(s))) for s in sets])
        return [loc(s) for s in sets]

    cur = sorted(int(c) for c in (init or []))
    best = eval_many([cur])[0]
    while len(cur) < max_parents:  # forward
        rest = [c for c in candidates if c not in cur]
        if not rest:
            break
        vals = eval_many([cur + [c] for c in rest])
        k = int(np.argmax(vals))
        if vals[k] <= best:
            break
        cur, best = cur + [rest[k]], vals[k]
    improved = backward
    while improved and cur:  # backward
        vals = eval_many([[c for c in cur if c != r] for r in cur])
        k = int(np.argmax(vals))
        improved = vals[k] > best
        if improved:
            best = vals[k]
            cur = [c for c in cur if c != cur[k]]
    out = sorted(int(c) for c in cur)
    return (out, best) if return_score else out


def tier_ebic_penalty(tiers, n, candidates, gamma=0.5):
    """Extended-BIC term per feature, sized by its tier: gamma * log(#candidates in its tier) / n.
    (EBIC, Chen & Chen 2008, on this score's per-sample scale.) Big, noisy tiers pay more per feature."""
    tiers = np.asarray(tiers)
    size = {t: max(2, int(np.sum(tiers[list(candidates)] == t))) for t in set(tiers[list(candidates)].tolist())}
    return {int(c): gamma * np.log(size[tiers[c]]) / n for c in candidates}


def select_parents_tiered(X, y, tiers, candidates, score, penalty_mult=1.0, **kw):
    """Tier-sequential search: forward selection within tier 0, then tier 1 given what was kept, ...;
    one backward pass at the end over the whole set."""
    tiers = np.asarray(tiers)
    cur = []
    for t in sorted(set(tiers[list(candidates)].tolist())):
        tc = [c for c in candidates if tiers[c] == t]
        cur = select_parents(X, y, candidates=tc, penalty_mult=penalty_mult, score=score, init=cur,
                             backward=False, **kw)
    return select_parents(X, y, candidates=list(cur), penalty_mult=penalty_mult, score=score, init=cur, **kw)


def stability_parents(X, y, B=30, frac=0.5, penalty_mult=1.0, screen=60, seed=0, **kw):
    """Selection frequency of each feature over B subsamples (size frac * n) of the outcome-local
    FENNEC search: the population-frequency version of the parent set (stability-selection style).
    Returns an array of frequencies (length p)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    freq = np.zeros(p)
    for b in range(B):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        pa = select_parents(X[idx], y[idx], penalty_mult=penalty_mult, screen=screen, seed=seed + b, **kw)
        freq[pa] += 1
    return freq / B
