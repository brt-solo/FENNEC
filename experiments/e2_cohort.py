"""E2: CHILD-like data with many features and small effects. Does a causal (parent) selection help?

Systems: benchmarks.generate_cohort.CohortSystem (time-ordered tiers, unmeasured shared causes that
create proxies, 30% binary, n_causes weak direct causes, outcome measured last so its Markov blanket
is its parent set). Grid over p (features), n (training rows), signal_r2; 10 seeds.

Feature sets (chosen on TRAINING rows only):
    all             every feature
    oracle          the true direct causes
    lasso           nonzero LassoCV coefficients (predictive feature selection, not causal)
    fennec_lin<m>   FENNEC outcome-local search (fennec_search.local), linear CV score, m x BIC penalty
    fennec_xgb<m>   the same with the XGBoost CV score (FENNEC's default score)
Each set is given to two predictors with fixed settings: XGBoost (compilers.XGB_DEFAULT) and
ElasticNetCV (its internal CV uses training rows only). Test sets: n_test rows per regime
(iid, proxy_break, cause_shift; see generate_cohort). Metric: test R^2.

    python -m experiments.e2_cohort [--pilot] [--shard k --nshards K --jobs J]
Penalties are fixed in the config in advance; none is chosen on test data.
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
from fennec_search.local import screen_candidates, select_parents
from fennec_search.score import CVScore

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


class _Mean:
    def fit(self, X, y):
        self.m = float(y.mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def fit_predictor(kind, X, y, seed, xgb):
    if X.shape[1] == 0:
        return _Mean().fit(X, y)
    if kind == "xgb":
        return XGBRegressor(**{**XGB_DEFAULT, **xgb, "random_state": seed}).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, n_jobs=1, random_state=seed, max_iter=5000).fit(X, y)


def r2(yhat, y):
    return 1 - float(np.mean((yhat - y) ** 2)) / float(y.var())


def run_system(ds, cfg):
    p, n, sr2, seed = ds["p"], ds["n"], ds["signal_r2"], ds["seed"]
    sysm = CohortSystem(p=p, n_causes=cfg["n_causes"], signal_r2=sr2, seed=seed)
    rng = np.random.default_rng(1_000_003 * seed + 7 * p + n)
    X, y = sysm.sample(n, rng)
    tests = {reg: sysm.sample(cfg["n_test"], rng, reg) for reg in cfg["regimes"]}
    true_pa, prox = set(sysm.y_parents.tolist()), set(sysm.proxies())

    sets, secs = {"all": list(range(p)), "oracle": sorted(true_pa)}, {}
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sets["lasso"] = np.flatnonzero(LassoCV(cv=5, n_jobs=1, random_state=seed, max_iter=5000).fit(X, y).coef_).tolist()
    secs["lasso"] = time.time() - t0
    cands = screen_candidates(X, y, cfg["screen"], seed) if cfg["screen"] < p else list(range(p))
    for model, pens in (("linear", cfg["fennec_linear_penalties"]), ("xgb", cfg["fennec_xgb_penalties"])):
        t0 = time.time()
        S = CVScore(np.column_stack([X, y]), model=model, n_jobs=1)
        tag = "lin" if model == "linear" else "xgb"
        for m in pens:
            sets[f"fennec_{tag}{m:g}"] = select_parents(X, y, candidates=cands, penalty_mult=m, score=S)
        secs[f"fennec_{tag}"] = time.time() - t0

    rows = []
    for sname, cols in sets.items():
        cols = [int(c) for c in cols]
        sel = set(cols)
        info = {"set": sname, "n_selected": len(cols), "true_hit": len(sel & true_pa), "proxies": len(sel & prox),
                "precision": len(sel & true_pa) / len(sel) if sel else float("nan"), "recall": len(sel & true_pa) / len(true_pa)}
        for kind in ("xgb", "enet"):
            model = fit_predictor(kind, X[:, cols], y, seed, cfg["xgb"])
            rows.append({**info, "predictor": kind,
                         **{f"r2_{reg}": r2(model.predict(Xt[:, cols]), yt) for reg, (Xt, yt) in tests.items()}})
    return {**ds, "n_proxies": len(prox), "seconds": secs, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e2.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.pilot:
        cfg.update(p=[200], n=[800], signal_r2=[0.3], seeds=[0, 1], n_test=2000)
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
