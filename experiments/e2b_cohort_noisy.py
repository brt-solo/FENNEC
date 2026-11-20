"""E2b: noisy CHILD-like data where using every feature HURTS. Population-frequency selection,
tier priors, hybrid predictors and heavily regularized boosting.

Data: benchmarks.generate_cohort.CohortSystem with weak true proxies (latent_y_weight = 0.25) and
p >= n, calibrated so that a model on the true causes beats the same model on all features
(elastic net by 0.03-0.06 R^2, XGBoost by about 0.05).

Feature sets (all chosen on TRAINING rows only; settings fixed in configs/e2b.json in advance):
    all, oracle (true causes), lasso (LassoCV nonzero)
    fennec          outcome-local FENNEC search, linear CV score, penalty_mult x BIC
    fennec_freq<t>  population frequency: the same search on B half-subsamples; keep features
                    selected in >= t of them (stability-selection style)
    fennec_tebic    the search with a tier-wise extended-BIC penalty (big tiers pay more per feature)
    fennec_tseq     tier-sequential search (genetics -> prenatal -> early life -> microbiome, then prune)
    fennec_xgbreg   the search with a heavily regularized XGBoost CV score
Predictors for each set: enet (ElasticNetCV), xgb (default settings), xgb_reg (depth 2, lr 0.02,
min_child_weight 20, L2 10, column subsampling, early stopping on an internal 20% split of TRAIN).
Hybrid predictors (use all features, guided by the population frequencies f_j):
    hybrid -> enet      elastic net on all features with column j scaled by s_min + f_j (s_max - s_min):
                        frequently selected features are penalized less, never-selected ones more
    hybrid -> xgb_reg   regularized XGBoost on all features with feature_weights = floor + f_j and
                        colsample_bynode = 0.3 (splits are drawn mostly from frequently selected features)
Test: n_test rows for each regime (iid, proxy_break, cause_shift). Metric: test R^2.

    python -m experiments.e2b_cohort_noisy [--pilot] [--shard k --nshards K --jobs J]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV, LassoCV
from xgboost import XGBRegressor

from benchmarks.generate_cohort import CohortSystem
from fennec_search.compilers import XGB_DEFAULT
from fennec_search.local import (screen_candidates, select_parents, select_parents_tiered, stability_parents,
                                 tier_ebic_penalty)
from fennec_search.score import CVScore

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


class _Mean:
    def fit(self, X, y):
        self.m = float(y.mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


class _Scaled:
    """Wrap a model so columns are multiplied by fixed scales before fitting/predicting."""
    def __init__(self, model, scale):
        self.model, self.scale = model, scale

    def fit(self, X, y):
        self.model.fit(X * self.scale, y)
        return self

    def predict(self, X):
        return self.model.predict(X * self.scale)


def enet(seed):
    return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, n_jobs=1, random_state=seed, max_iter=5000)


class RegXGB:
    """Heavily regularized boosting with early stopping on an internal split of the training rows."""
    def __init__(self, cfg, seed, feature_weights=None, colsample_bynode=None):
        c = dict(cfg)
        self.valid_frac = c.pop("valid_frac")
        if colsample_bynode is not None:
            c["colsample_bynode"] = colsample_bynode
        self.seed = seed
        if feature_weights is not None:
            c["feature_weights"] = feature_weights
        self.model = XGBRegressor(**c, tree_method="hist", n_jobs=1, random_state=seed)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        nv = int(self.valid_frac * len(y))
        va, tr = idx[:nv], idx[nv:]
        self.model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)


def make(kind, seed, cfg):
    if kind == "enet":
        return enet(seed)
    if kind == "xgb":
        return XGBRegressor(**{**XGB_DEFAULT, "random_state": seed})
    return RegXGB(cfg["xgb_reg"], seed)


def r2(yhat, y):
    return 1 - float(np.mean((yhat - y) ** 2)) / float(y.var())


def run_system(ds, cfg):
    p, n, sr2, seed = ds["p"], ds["n"], ds["signal_r2"], ds["seed"]
    sysm = CohortSystem(p=p, n_causes=cfg["n_causes"], signal_r2=sr2, seed=seed, latent_y_weight=cfg["latent_y_weight"])
    rng = np.random.default_rng(1_000_003 * seed + 7 * p + n + int(100 * sr2))
    X, y = sysm.sample(n, rng)
    tests = {reg: sysm.sample(cfg["n_test"], rng, reg) for reg in cfg["regimes"]}
    true_pa, prox = set(sysm.y_parents.tolist()), set(sysm.proxies())
    pm = cfg["penalty_mult"]
    secs = {}

    sets = {"all": list(range(p)), "oracle": sorted(true_pa)}
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sets["lasso"] = np.flatnonzero(LassoCV(cv=5, n_jobs=1, random_state=seed, max_iter=5000).fit(X, y).coef_).tolist()
    secs["lasso"] = time.time() - t0

    t0 = time.time()
    cands = screen_candidates(X, y, cfg["screen"], seed)
    S = CVScore(np.column_stack([X, y]), model="linear", n_jobs=1)
    sets["fennec"] = select_parents(X, y, candidates=cands, penalty_mult=pm, score=S)
    secs["fennec"] = time.time() - t0
    t0 = time.time()
    extra = tier_ebic_penalty(sysm.tier, n, range(p), cfg["tier_ebic_gamma"])  # tier sizes over ALL features
    sets["fennec_tebic"] = select_parents(X, y, candidates=cands, penalty_mult=pm, score=S, extra_penalty=extra)
    sets["fennec_tseq"] = select_parents_tiered(X, y, sysm.tier, cands, S, penalty_mult=pm)
    secs["fennec_tier"] = time.time() - t0
    t0 = time.time()
    Sx = CVScore(np.column_stack([X, y]), model="xgb", n_jobs=1, xgb_params=cfg["xgb_reg_score"])
    sets["fennec_xgbreg"] = select_parents(X, y, candidates=cands, penalty_mult=pm, score=Sx, model="xgb")
    secs["fennec_xgbreg"] = time.time() - t0
    t0 = time.time()
    st = cfg["stability"]
    freq = stability_parents(X, y, B=st["B"], frac=st["frac"], penalty_mult=pm, screen=cfg["screen"], seed=seed)
    for th in st["thresholds"]:
        sets[f"fennec_freq{th:g}"] = np.flatnonzero(freq >= th).tolist()
    secs["fennec_freq"] = time.time() - t0

    rows = []

    def score_rows(name, kind, model, cols, info):
        rows.append({**info, "set": name, "predictor": kind,
                     **{f"r2_{reg}": r2(model.predict(Xt[:, cols]), yt) for reg, (Xt, yt) in tests.items()}})

    for sname, cols in sets.items():
        cols = [int(c) for c in cols]
        sel = set(cols)
        info = {"n_selected": len(cols), "true_hit": len(sel & true_pa), "proxies": len(sel & prox)}
        for kind in ("enet", "xgb", "xgb_reg"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = (make(kind, seed, cfg) if cols else _Mean()).fit(X[:, cols], y)
            score_rows(sname, kind, m, cols, info)

    h = cfg["hybrid"]
    allc = list(range(p))
    info = {"n_selected": p, "true_hit": len(true_pa), "proxies": len(prox)}
    scale = h["enet_scale_min"] + freq * (h["enet_scale_max"] - h["enet_scale_min"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score_rows("hybrid", "enet", _Scaled(enet(seed), scale).fit(X, y), allc, info)
    fw = h["xgb_weight_floor"] + freq
    score_rows("hybrid", "xgb_reg", RegXGB(cfg["xgb_reg"], seed, fw, h["xgb_colsample_bynode"]).fit(X, y), allc, info)

    top = np.argsort(-freq)[:30]
    return {**ds, "n_proxies": len(prox), "seconds": secs, "rows": rows,
            "freq_true": float(freq[sorted(true_pa)].mean()), "freq_proxy": float(freq[sorted(prox)].mean()) if prox else None,
            "freq_other": float(np.delete(freq, sorted(true_pa | prox)).mean()),
            "top30_true": int(len(set(top.tolist()) & true_pa))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e2b.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.pilot:
        cfg.update(p=[500], n=[800], signal_r2=[0.3], seeds=[0, 1], n_test=2000)
        cfg["stability"]["B"] = 10
    RAW.mkdir(parents=True, exist_ok=True)
    tag = "_pilot" if a.pilot else (f"_shard{a.shard}" if a.nshards > 1 else "")
    out = RAW / f"{cfg['name']}{tag}.jsonl"
    systems = [{"p": p, "n": n, "signal_r2": s, "seed": k}
               for p, n, s, k in product(cfg["p"], cfg["n"], cfg["signal_r2"], cfg["seeds"])][a.shard::a.nshards]
    key = lambda r: (r["p"], r["n"], r["signal_r2"], r["seed"])
    done = {key(json.loads(l)) for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in systems if key(s) not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(systems)} systems, {len(todo)} to do -> {out}", flush=True)
    gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(delayed(run_system)(s, cfg) for s in todo)
    for k, rec in enumerate(gen, 1):
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  [{k}/{len(todo)}] {key(rec)} done ({sum(rec['seconds'].values()):.0f}s selection)", flush=True)


if __name__ == "__main__":
    main()
