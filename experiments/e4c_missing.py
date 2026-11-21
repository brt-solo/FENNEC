"""E4c: a sub-study block missing for most participants. Can the causal graph fill it in?

CHILD's microbiome / HMO / methylation data exist only for sub-samples, and those sub-samples were
enriched for cases. Systems: CohortSystem (E2b regime: all features hurt). The block = tier 3
(microbiome-like, 30% of features, about 4-5 of the 15 causes of Y). In training, the block is
observed for obs_frac of rows, either
    mcar           completely at random, or
    case_enriched  P(observed) = sigmoid(c - case_strength * y): low-y (worse lung function) rows are
                   more likely to have the block, as in a case-enriched sub-study.
Test sets (n_test rows, same system) differ only in which rows have the block:
    same    the training mechanism again       mcar  30% at random
    none    nobody has the block                full  everybody has it
Methods (all fitted on training rows only; the outcome is never used to impute):
    pattern_*     pattern submodels: one model on rows with the block (all features), one on all rows
                  without block features; each test row uses the model matching what it has
    xgb_nan       regularized XGBoost on all features, NaN for missing (learns its own default
                  directions, so it can exploit WHO is missing)
    meanind_enet  mean imputation + a "block missing" indicator -> elastic net
    mice_enet     iterative chained-equation imputation (MICE-style, all features) -> elastic net
    mice_C_enet   the same imputed data restricted to the FENNEC causal set C
    scm_*         SCM imputation: for each block member of C, FENNEC finds its parents among the
                  always-observed (earlier-tier) features on rows where it is observed; missing values
                  are predicted from those parents; then the model on C uses all training rows
    oracle_scm    true causes, block causes imputed from their TRUE always-observed parents
    full_*        references trained and tested with no missing data
C = union of FENNEC population-frequency selections on complete rows (all features) and on all rows
(always-observed features only), threshold 0.3.
Metric: test R^2 over all test rows; also imputation accuracy (correlation with the true value) for
the block members of C on test rows, SCM vs MICE.

    python -m experiments.e4c_missing [--pilot] [--shard k --nshards K --jobs J]
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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

from benchmarks.generate_cohort import CohortSystem
from experiments.e2b_cohort_noisy import RegXGB, _Mean, enet, r2
from fennec_search.local import select_parents, stability_parents

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
XGB_REG = json.loads((ROOT / "experiments" / "configs" / "e2b.json").read_text())["xgb_reg"]


def obs_mask(y, mech, rng, frac, strength):
    n = len(y)
    if mech == "none":
        return np.zeros(n, bool)
    if mech == "full":
        return np.ones(n, bool)
    if mech == "mcar":
        return rng.random(n) < frac
    lo, hi = -20.0, 20.0  # case_enriched: solve for the intercept giving the target fraction
    for _ in range(60):
        c = (lo + hi) / 2
        if (1 / (1 + np.exp(-(c - strength * y)))).mean() < frac:
            lo = c
        else:
            hi = c
    return rng.random(n) < 1 / (1 + np.exp(-(c - strength * y)))


class Pattern:
    def __init__(self, make, block):
        self.make, self.block = make, block

    def fit(self, X, y, obs):
        nb = [j for j in range(X.shape[1]) if j not in set(self.block)]
        self.nb = nb
        self.full = self.make().fit(X[obs], y[obs]) if obs.sum() > 20 else None
        self.part = self.make().fit(X[:, nb], y)
        return self

    def predict(self, X, obs):
        out = self.part.predict(X[:, self.nb])
        if self.full is not None and obs.any():
            out[obs] = self.full.predict(X[obs])
        return out


def run_system(ds, cfg):
    p, n, mech, seed = ds["p"], ds["n"], ds["mechanism"], ds["seed"]
    sysm = CohortSystem(p=p, n_causes=cfg["n_causes"], signal_r2=cfg["signal_r2"], seed=seed,
                        latent_y_weight=cfg["latent_y_weight"])
    rng = np.random.default_rng(3_000_017 * seed + 7 * p + n + (0 if mech == "mcar" else 1))
    block = [int(j) for j in np.flatnonzero(sysm.tier == 3)]
    bset, nb = set(block), [j for j in range(p) if sysm.tier[j] != 3]
    Xfull, y = sysm.sample(n, rng)
    obs = obs_mask(y, mech, rng, cfg["obs_frac"], cfg["case_strength"])
    X = Xfull.copy()
    X[np.ix_(~obs, block)] = np.nan
    tests = {}
    for reg in cfg["test_regimes"]:
        Xt_full, yt = sysm.sample(cfg["n_test"], rng)
        o = obs_mask(yt, mech if reg == "same" else reg, rng, cfg["obs_frac"], cfg["case_strength"])
        Xt = Xt_full.copy()
        Xt[np.ix_(~o, block)] = np.nan
        tests[reg] = (Xt, Xt_full, yt, o)
    true_pa = set(sysm.y_parents.tolist())
    pm, st = cfg["penalty_mult"], cfg["stability"]
    secs = {}

    # causal set
    t0 = time.time()
    f_full = stability_parents(Xfull[obs], y[obs], B=st["B"], frac=st["frac"], penalty_mult=pm,
                               screen=cfg["screen"], seed=seed)
    f_nb = stability_parents(Xfull[:, nb], y, B=st["B"], frac=st["frac"], penalty_mult=pm,
                             screen=cfg["screen"], seed=seed)
    C = sorted(set(np.flatnonzero(f_full >= st["threshold"]).tolist()) |
               {nb[k] for k in np.flatnonzero(f_nb >= st["threshold"])})
    C = [int(c) for c in C]
    CB = [c for c in C if c in bset]
    secs["selection"] = time.time() - t0

    # SCM imputation models for block members of C (parents among always-observed features)
    t0 = time.time()
    imp_models = {}
    for b in CB:
        pa_local = select_parents(Xfull[np.ix_(obs, nb)], Xfull[obs, b], penalty_mult=pm, screen=cfg["screen"], seed=seed)
        pa = [nb[k] for k in pa_local]
        imp_models[b] = (pa, LinearRegression().fit(Xfull[np.ix_(obs, pa)], Xfull[obs, b]) if pa else None,
                         float(Xfull[obs, b].mean()))
    secs["scm_imputation"] = time.time() - t0

    def scm_fill(Xm):
        Z = Xm[:, C].copy()
        for k, c in enumerate(C):
            if c in imp_models:
                miss = np.isnan(Z[:, k])
                if miss.any():
                    pa, m, mu = imp_models[c]
                    Z[miss, k] = m.predict(Xm[np.ix_(miss, pa)]) if m is not None else mu
        return Z

    # oracle SCM imputation: true causes; block causes from their true always-observed parents
    TP = sorted(true_pa)
    o_models = {}
    for b in [c for c in TP if c in bset]:
        pa = [int(q) for q in sysm.parents[b] if int(q) not in bset]
        o_models[b] = (pa, LinearRegression().fit(Xfull[np.ix_(obs, pa)], Xfull[obs, b]) if pa else None,
                       float(Xfull[obs, b].mean()))

    def oracle_fill(Xm):
        Z = Xm[:, TP].copy()
        for k, c in enumerate(TP):
            if c in o_models:
                miss = np.isnan(Z[:, k])
                if miss.any():
                    pa, m, mu = o_models[c]
                    Z[miss, k] = m.predict(Xm[np.ix_(miss, pa)]) if m is not None else mu
        return Z

    # MICE
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mice = IterativeImputer(max_iter=cfg["mice"]["max_iter"], n_nearest_features=cfg["mice"]["n_nearest_features"],
                                random_state=seed).fit(X)
    secs["mice"] = time.time() - t0
    mu_tr = np.nanmean(X, 0)

    def meanind(Xm):
        Z = np.where(np.isnan(Xm), mu_tr, Xm)
        return np.column_stack([Z, np.isnan(Xm[:, block[0]]).astype(float)])

    rows = []

    def add(name, fit_pred):
        rec = {"method": name}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = fit_pred()
            for reg, (Xt, Xt_full, yt, o) in tests.items():
                yhat = pred(Xt, Xt_full, o)
                rec[f"r2_{reg}"] = r2(yhat, yt)
                if reg == "mcar":  # split by whether the row has the block
                    rec["r2_mcar_with_block"] = r2(yhat[o], yt[o]) if o.sum() > 10 else None
                    rec["r2_mcar_without_block"] = r2(yhat[~o], yt[~o])
        rows.append(rec)

    for kind, make in (("enet", lambda: enet(seed)), ("xgb_reg", lambda: RegXGB(XGB_REG, seed))):
        add(f"pattern_{kind}", lambda make=make: (lambda m: (lambda Xt, Xf, o: m.predict(Xt, o)))(
            Pattern(make, block).fit(X, y, obs)))
        add(f"scm_{kind}", lambda make=make: (lambda m: (lambda Xt, Xf, o: m.predict(scm_fill(Xt))))(
            make().fit(scm_fill(X), y) if C else _Mean().fit(X, y)))
        add(f"full_all_{kind}", lambda make=make: (lambda m: (lambda Xt, Xf, o: m.predict(Xf)))(make().fit(Xfull, y)))
        add(f"full_C_{kind}", lambda make=make: (lambda m: (lambda Xt, Xf, o: m.predict(Xf[:, C])))(
            make().fit(Xfull[:, C], y) if C else _Mean().fit(X, y)))
    add("xgb_nan", lambda: (lambda m: (lambda Xt, Xf, o: m.predict(Xt)))(RegXGB(XGB_REG, seed).fit(X, y)))
    add("meanind_enet", lambda: (lambda m: (lambda Xt, Xf, o: m.predict(meanind(Xt))))(enet(seed).fit(meanind(X), y)))
    Xi = mice.transform(X)
    add("mice_enet", lambda: (lambda m: (lambda Xt, Xf, o: m.predict(mice.transform(Xt))))(enet(seed).fit(Xi, y)))
    add("mice_C_enet", lambda: (lambda m: (lambda Xt, Xf, o: m.predict(mice.transform(Xt)[:, C])))(
        enet(seed).fit(Xi[:, C], y) if C else _Mean().fit(X, y)))
    add("oracle_scm_enet", lambda: (lambda m: (lambda Xt, Xf, o: m.predict(oracle_fill(Xt))))(enet(seed).fit(oracle_fill(X), y)))

    # imputation accuracy on the "none" test set (every block value imputed)
    Xt, Xt_full, yt, o = tests["none"]
    imp_acc = {}
    if CB:
        S, M = scm_fill(Xt), mice.transform(Xt)
        for k, c in enumerate(C):
            if c in bset:
                imp_acc[str(c)] = {"scm": float(np.corrcoef(S[:, k], Xt_full[:, c])[0, 1]) if S[:, k].std() > 0 else 0.0,
                                   "mice": float(np.corrcoef(M[:, c], Xt_full[:, c])[0, 1]) if M[:, c].std() > 0 else 0.0,
                                   "n_parents": len(imp_models[c][0]), "true_cause": c in true_pa}
    return {**ds, "seconds": secs, "n_obs_train": int(obs.sum()), "y_mean_observed": float(y[obs].mean()),
            "n_C": len(C), "n_C_block": len(CB), "C_true": len(set(C) & true_pa),
            "n_true_block_causes": len([c for c in TP if c in bset]), "imputation": imp_acc, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e4c.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.pilot:
        cfg.update(p=[200], n=[800], train_mechanisms=["case_enriched"], seeds=[0], n_test=2000)
        cfg["stability"]["B"] = 10
    RAW.mkdir(parents=True, exist_ok=True)
    tag = "_pilot" if a.pilot else (f"_shard{a.shard}" if a.nshards > 1 else "")
    out = RAW / f"{cfg['name']}{tag}.jsonl"
    systems = [{"p": p, "n": n, "mechanism": m, "seed": k}
               for p, n, m, k in product(cfg["p"], cfg["n"], cfg["train_mechanisms"], cfg["seeds"])][a.shard::a.nshards]
    key = lambda r: (r["p"], r["n"], r["mechanism"], r["seed"])
    done = {key(json.loads(l)) for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in systems if key(s) not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(systems)} systems, {len(todo)} to do -> {out}", flush=True)
    gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(delayed(run_system)(s, cfg) for s in todo)
    for k, rec in enumerate(gen, 1):
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  [{k}/{len(todo)}] {key(rec)} done ({rec['seconds']})", flush=True)


if __name__ == "__main__":
    main()
