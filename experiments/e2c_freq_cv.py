"""E2c: FENNEC population frequency with the number of kept features chosen by inner cross-validation.

Same systems and training/test draws as E2b (same seeds). On the training rows:
  1. population frequency f_j: outcome-local FENNEC search (linear score, 1x BIC) on B half-subsamples
  2. rank features by f_j (only f_j > 0 are eligible)
  3. for each k in k_grid, inner 5-fold CV error of the predictor on the top-k features; keep the best k
     (separately for elastic net and regularized XGBoost). Test rows are never used for any choice.
Caveat: the frequencies come from all training rows, so the inner CV is mildly optimistic about the
chosen k; this cannot leak test information.
Reported alongside: all features, true causes, fixed thresholds 0.1 / 0.2 (and E2b has 0.3 / 0.5 / 0.7).

    python -m experiments.e2c_freq_cv [--pilot] [--shard k --nshards K --jobs J]
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
from sklearn.model_selection import KFold

from benchmarks.generate_cohort import CohortSystem
from experiments.e2b_cohort_noisy import RAW, ROOT, RegXGB, _Mean, enet, r2
from fennec_search.local import stability_parents


def model(kind, seed, cfg):
    return enet(seed) if kind == "enet" else RegXGB(cfg["xgb_reg"], seed)


def choose_k(X, y, order, kind, seed, cfg, ccfg):
    ks = [k for k in ccfg["k_grid"] if k <= len(order)] or [len(order)]
    folds = list(KFold(ccfg["inner_folds"], shuffle=True, random_state=seed).split(X))
    err = {}
    for k in ks:
        cols = order[:k]
        e = 0.0
        for tr, va in folds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = model(kind, seed, cfg).fit(X[np.ix_(tr, cols)], y[tr])
            e += float(np.sum((m.predict(X[np.ix_(va, cols)]) - y[va]) ** 2))
        err[k] = e
    return min(err, key=err.get)


def run_system(ds, cfg, ccfg):
    p, n, sr2, seed = ds["p"], ds["n"], ds["signal_r2"], ds["seed"]
    sysm = CohortSystem(p=p, n_causes=cfg["n_causes"], signal_r2=sr2, seed=seed, latent_y_weight=cfg["latent_y_weight"])
    rng = np.random.default_rng(1_000_003 * seed + 7 * p + n + int(100 * sr2))  # identical to E2b
    X, y = sysm.sample(n, rng)
    tests = {reg: sysm.sample(cfg["n_test"], rng, reg) for reg in cfg["regimes"]}
    true_pa = set(sysm.y_parents.tolist())
    t0 = time.time()
    st = cfg["stability"]
    freq = stability_parents(X, y, B=st["B"], frac=st["frac"], penalty_mult=cfg["penalty_mult"], screen=cfg["screen"], seed=seed)
    order = [int(j) for j in np.argsort(-freq, kind="stable") if freq[j] > 0]
    secs = time.time() - t0

    rows = []

    def add(name, kind, cols, extra=None):
        cols = [int(c) for c in cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = (model(kind, seed, cfg) if cols else _Mean()).fit(X[:, cols], y)
        rows.append({"set": name, "predictor": kind, "n_selected": len(cols), "true_hit": len(set(cols) & true_pa),
                     **(extra or {}), **{f"r2_{reg}": r2(m.predict(Xt[:, cols]), yt) for reg, (Xt, yt) in tests.items()}})

    for kind in ("enet", "xgb_reg"):
        add("all", kind, range(p))
        add("oracle", kind, sorted(true_pa))
        for th in ccfg["fixed_thresholds"]:
            add(f"fennec_freq{th:g}", kind, np.flatnonzero(freq >= th))
        k = choose_k(X, y, order, kind, seed, cfg, ccfg) if order else 0
        add("fennec_freq_cv", kind, order[:k], {"k": k})
    return {**ds, "seconds": secs, "n_positive_freq": len(order),
            "true_rank_mean": float(np.mean([order.index(j) if j in order else len(order) for j in true_pa])),
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e2c.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    ccfg = json.loads(a.config.read_text())
    cfg = json.loads((a.config.parent / ccfg["base_config"]).read_text())
    if a.pilot:
        cfg.update(p=[500], n=[800], signal_r2=[0.3], seeds=[0], n_test=2000)
        cfg["stability"]["B"] = 10
    RAW.mkdir(parents=True, exist_ok=True)
    tag = "_pilot" if a.pilot else (f"_shard{a.shard}" if a.nshards > 1 else "")
    out = RAW / f"{ccfg['name']}{tag}.jsonl"
    systems = [{"p": p, "n": n, "signal_r2": s, "seed": k}
               for p, n, s, k in product(cfg["p"], cfg["n"], cfg["signal_r2"], cfg["seeds"])][a.shard::a.nshards]
    key = lambda r: (r["p"], r["n"], r["signal_r2"], r["seed"])
    done = {key(json.loads(l)) for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in systems if key(s) not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(systems)} systems, {len(todo)} to do -> {out}", flush=True)
    gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(delayed(run_system)(s, cfg, ccfg) for s in todo)
    for k, rec in enumerate(gen, 1):
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  [{k}/{len(todo)}] {key(rec)} done", flush=True)


if __name__ == "__main__":
    main()
