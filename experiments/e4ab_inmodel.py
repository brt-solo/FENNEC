"""E4a + E4b: using the causal graph INSIDE the predictor, beyond picking features.

Systems: benchmarks.generate_cohort.CohortSystem (CHILD-like: 4 time-ordered tiers, 15 weak causes,
6 unmeasured factors driving correlated feature blocks, two of which also cause Y). latent_y_weight
controls how much Y depends on those unmeasured causes: 0.25 = their measured "proxies" carry little
signal (E2b regime, all features hurt); 1.0 = proxies carry real signal (E2 regime).

Shared step (training rows only): FENNEC population frequency f_j (outcome-local search, linear
score, 1x BIC, 30 half-subsamples); causal set C = {j : f_j >= 0.3}.

E4a, latent factors. Proxies are noisy measurements of an unmeasured cause U of Y. One proxy says
little about U; the average of many says a lot (this is why factor models work). Selection throws
proxies away; using all of them separately means estimating hundreds of tiny coefficients.
A factor model estimates the latent nodes of the graph (a measurement model) and gives a few
denoised features. Designs:
    causal            C
    causal+factors    C plus the first n_factors principal components of the training features
    causal+selfac     C plus the factors FENNEC's local search adds given C (forward, from C)
    all, all+factors  every feature (with / without the factors)
    oracle+latent     true causes plus the TRUE latent values (unobservable ceiling)
Recorded: how well the factors recover Y's latent causes (R^2 of each true latent on the factors, test).

E4b, graph-distance penalties. Instead of keep / drop, every feature stays in the model and its
penalty depends on its place in the learned graph relative to Y:
    distance 1 = C (parents), distance 2 = parents of members of C (from a local search per parent),
    distance 3 = everything else.
A feature with scale s enters the elastic net as s * x_j, so its L1 penalty factor is 1/s (L2: 1/s^2).
    uniform           ordinary elastic net (all s = 1)
    adaptive          s_j = f_j + floor (adaptive-lasso-like weights from the population frequency)
    dist_soft         s = 1 / 0.5 / 0.2 for distance 1 / 2 / 3
    dist_strong       s = 1 / 0.3 / 0.05
    oracle_dist       dist_strong with the TRUE parents and grandparents
Each also as regularized XGBoost with feature_weights = s and colsample_bynode = 0.3 (splits are drawn
mostly from graph-near features). Recorded: nonzero elastic-net coefficients by true class.

    python -m experiments.e4ab_inmodel [--pilot] [--shard k --nshards K --jobs J]
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from benchmarks.generate_cohort import CohortSystem
from experiments.e2b_cohort_noisy import RegXGB, _Mean, _Scaled, enet, r2
from fennec_search.local import screen_candidates, select_parents, stability_parents
from fennec_search.score import CVScore

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
XGB_REG = json.loads((ROOT / "experiments" / "configs" / "e2b.json").read_text())["xgb_reg"]


def local_parents_of(X, target, candidates_pool, seed, pm, screen):
    """FENNEC local search for the parents of feature `target` among the other features."""
    others = np.array([j for j in range(X.shape[1]) if j != target])
    Xo = X[:, others]
    pa = select_parents(Xo, X[:, target], penalty_mult=pm, screen=screen, seed=seed)
    return [int(others[k]) for k in pa]


def true_grandparents(sysm, parents):
    gp = set()
    for j in parents:
        gp |= set(int(q) for q in sysm.parents[j])
    return gp - set(parents)


def classes(sysm):
    true_pa = set(sysm.y_parents.tolist())
    prox = set(sysm.proxies())
    gp = true_grandparents(sysm, true_pa) - prox
    return true_pa, prox, gp


def run_system(ds, cfg):
    p, n, lw, seed = ds["p"], ds["n"], ds["latent_y_weight"], ds["seed"]
    sysm = CohortSystem(p=p, n_causes=cfg["n_causes"], signal_r2=cfg["signal_r2"], seed=seed, latent_y_weight=lw)
    rng = np.random.default_rng(2_000_003 * seed + 7 * p + n + int(100 * lw))
    X, y, Ltr = sysm.sample(n, rng, return_latent=True)
    tests = {}
    for reg in cfg["regimes"]:
        Xt, yt, Lt = sysm.sample(cfg["n_test"], rng, reg, return_latent=True)
        tests[reg] = (Xt, yt, Lt)
    true_pa, prox, true_gp = classes(sysm)
    pm, st = cfg["penalty_mult"], cfg["stability"]
    secs = {}

    t0 = time.time()
    freq = stability_parents(X, y, B=st["B"], frac=st["frac"], penalty_mult=pm, screen=cfg["screen"], seed=seed)
    C = [int(j) for j in np.flatnonzero(freq >= st["threshold"])]
    secs["frequency"] = time.time() - t0

    rows = []

    def evaluate(name, predictor, fit_model, design, info=None):
        """design(X, L, which) -> matrix; which = 'train' or a regime name."""
        Ztr = design(X, Ltr, "train")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = (fit_model() if Ztr.shape[1] else _Mean()).fit(Ztr, y)
        rec = {"exp": name.split(":")[0], "design": name.split(":")[1], "predictor": predictor, **(info or {})}
        for reg, (Xt, yt, Lt) in tests.items():
            rec[f"r2_{reg}"] = r2(m.predict(design(Xt, Lt, reg)), yt)
        rows.append(rec)
        return m

    # ------------------------------------------------------------------ E4a: latent factors
    t0 = time.time()
    pca = PCA(n_components=cfg["n_factors"], random_state=seed).fit(X)
    F = lambda Xm: pca.transform(Xm)
    # factor recovery of Y's latent causes (on the iid test set)
    Xt, yt, Lt = tests["iid"]
    Ft = F(Xt)
    recov = [float(LinearRegression().fit(Ft, Lt[:, k]).score(Ft, Lt[:, k])) for k in range(Lt.shape[1])]
    # FENNEC adds factors to C: forward search from C over the factor columns only
    XF = np.column_stack([X, F(X)])
    fac_idx = list(range(p, p + cfg["n_factors"]))
    S = CVScore(np.column_stack([XF, y]), model="linear", n_jobs=1)
    sel = select_parents(XF, y, candidates=fac_idx, penalty_mult=pm, score=S, init=C, backward=False)
    selfac = [k - p for k in sel if k >= p]
    secs["factors"] = time.time() - t0

    designs_a = {
        "causal": lambda Xm, L, w: Xm[:, C],
        "causal+factors": lambda Xm, L, w: np.column_stack([Xm[:, C], F(Xm)]),
        "causal+selfac": lambda Xm, L, w: np.column_stack([Xm[:, C], F(Xm)[:, selfac]]),
        "all": lambda Xm, L, w: Xm,
        "all+factors": lambda Xm, L, w: np.column_stack([Xm, F(Xm)]),
        "oracle": lambda Xm, L, w: Xm[:, sorted(true_pa)],
        "oracle+latent": lambda Xm, L, w: np.column_stack([Xm[:, sorted(true_pa)], L]),
    }
    for dname, des in designs_a.items():
        evaluate(f"E4a:{dname}", "enet", lambda: enet(seed), des)
        if dname not in ("all+factors",):
            evaluate(f"E4a:{dname}", "xgb_reg", lambda: RegXGB(XGB_REG, seed), des)

    # ------------------------------------------------------------------ E4b: graph-distance penalties
    t0 = time.time()
    gp = set()
    for j in C:
        gp |= set(local_parents_of(X, j, None, seed, pm, cfg["screen"]))
    gp -= set(C)
    dist = np.full(p, 3)
    dist[list(gp)] = 2
    dist[C] = 1
    odist = np.full(p, 3)
    odist[list(true_gp)] = 2
    odist[sorted(true_pa)] = 1
    secs["grandparents"] = time.time() - t0

    scales = {"uniform": np.ones(p), "adaptive": freq + cfg["adaptive_floor"]}
    for name, sc in cfg["distance_scales"].items():
        scales[name] = np.array([sc[str(k)] for k in dist])
    sc = cfg["distance_scales"]["dist_strong"]
    scales["oracle_dist"] = np.array([sc[str(k)] for k in odist])

    cls = np.array(["other"] * p, dtype=object)
    cls[list(true_gp)] = "true grandparent"
    cls[list(prox)] = "proxy"
    cls[sorted(true_pa)] = "true parent"
    for name, s in scales.items():
        m = evaluate(f"E4b:{name}", "enet", lambda s=s: _Scaled(enet(seed), s), lambda Xm, L, w: Xm,
                     {"n_dist1": int((dist == 1).sum()), "n_dist2": int((dist == 2).sum())})
        coef = m.model.coef_ * s  # effective coefficient on the original feature
        nz = np.abs(coef) > 1e-10
        rows[-1].update({f"nz_{c.replace(' ', '_')}": int((nz & (cls == c)).sum()) for c in
                         ["true parent", "true grandparent", "proxy", "other"]})
        rows[-1]["abs_coef_true_parent"] = float(np.abs(coef[cls == "true parent"]).sum())
        rows[-1]["abs_coef_nonparent"] = float(np.abs(coef[cls != "true parent"]).sum())
        evaluate(f"E4b:{name}", "xgb_reg",
                 lambda s=s: RegXGB(XGB_REG, seed, feature_weights=s, colsample_bynode=cfg["xgb_colsample_bynode"]),
                 lambda Xm, L, w: Xm)
    # hard selection reference (E2b best)
    evaluate("E4b:hard_select_C", "enet", lambda: enet(seed), lambda Xm, L, w: Xm[:, C])
    evaluate("E4b:hard_select_C", "xgb_reg", lambda: RegXGB(XGB_REG, seed), lambda Xm, L, w: Xm[:, C])

    return {**ds, "seconds": secs, "n_C": len(C), "C_true": len(set(C) & true_pa), "C_proxy": len(set(C) & prox),
            "n_gp": len(gp), "gp_true_gp": len(gp & true_gp), "gp_true_parent": len(gp & true_pa),
            "gp_proxy": len(gp & prox), "n_true_gp": len(true_gp), "n_proxies": len(prox),
            "factor_recovery": recov, "selected_factors": selfac, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e4ab.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.pilot:
        cfg.update(p=[200], n=[800], latent_y_weight=[1.0], seeds=[0], n_test=2000)
        cfg["stability"]["B"] = 10
    RAW.mkdir(parents=True, exist_ok=True)
    tag = "_pilot" if a.pilot else (f"_shard{a.shard}" if a.nshards > 1 else "")
    out = RAW / f"{cfg['name']}{tag}.jsonl"
    systems = [{"p": p, "n": n, "latent_y_weight": w, "seed": k}
               for p, n, w, k in product(cfg["p"], cfg["n"], cfg["latent_y_weight"], cfg["seeds"])][a.shard::a.nshards]
    key = lambda r: (r["p"], r["n"], r["latent_y_weight"], r["seed"])
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
