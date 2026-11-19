"""Structure-learning methods for the synthetic benchmark, all with the same interface:

    method(X: np.ndarray (n x d), standardized) -> adjacency (d x d) of 0/1,
        adj[i, j] = 1 means i -> j; an undirected edge (CPDAG output) is adj[i, j] = adj[j, i] = 1;
        skeleton-only methods return symmetric matrices.

FENNEC-Search (the fennec_search package: cross-validated XGBoost parent score + hill climbing):
    fennec            4x BIC edge penalty (the default; chosen on seeds 0-4 and Sachs, confirmed on seeds 5-9)
    fennec_bic        plain BIC edge penalty (1x)
    fennec_screened   1x penalty, candidate parents restricted to the lasso neighbourhood (OR rule)
    linear_score      the same search with a cross-validated linear score (1x)
CAM, BiDAG, the generalized score (gsf) and the pruning step of score_sm / nogam / grandag call R
(set FENNEC_RSCRIPT to your Rscript; R packages CAM, BiDAG, bnlearn). SCORE and NoGAM are our own
implementations from the papers; GraN-DAG uses gcastle's defaults.
Baselines:
    pc                PC, Fisher-z, alpha 0.05 (causal-learn) -> CPDAG
    ges               GES with BIC (causal-learn) -> CPDAG
    nbsel             neighbourhood selection, LassoCV per node, AND rule -> skeleton
    mmhc              MMPC (Fisher-z, alpha 0.05, conditioning sets <= 3) + hill climbing with the
                      linear score restricted to the MMPC skeleton (Tsamardinos et al. 2006)
    lingam            DirectLiNGAM (causal-learn) -> DAG
    r2_sortnregress   artefact check: R^2-sortability baseline (should be poor on a sound benchmark)
    notears_mlp       NOTEARS-MLP (Zheng et al. 2020): MLP per node, augmented-Lagrangian acyclicity,
                      L-BFGS-B, lambda1 = lambda2 = 0.01, 10 hidden units, threshold 0.3
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import LassoCV

from fennec_search.score import CVScore
from fennec_search.search import hill_climb


def _full_mask(d):
    return ~np.eye(d, dtype=bool)


def _cl_graph(G):
    """causal-learn GeneralGraph -> our convention."""
    g = G.graph
    d = len(g)
    a = np.zeros((d, d), int)
    for i in range(d):
        for j in range(d):
            if g[j, i] == 1 and g[i, j] == -1:
                a[i, j] = 1
            elif g[i, j] == -1 and g[j, i] == -1:
                a[i, j] = a[j, i] = 1
            elif g[i, j] == 1 and g[j, i] == 1:
                a[i, j] = a[j, i] = 1
    return a


# ----------------------------------------------------------------------------------- ours
def fennec(X, n_jobs=-1):
    """FENNEC-Search best graph: XGBoost parent score with 4x the BIC edge penalty."""
    return hill_climb(CVScore(np.asarray(X, float), model="xgb", n_jobs=n_jobs, penalty_mult=4.0), _full_mask(X.shape[1]))[0]


def fennec_bic(X, n_jobs=-1):
    return hill_climb(CVScore(np.asarray(X, float), model="xgb", n_jobs=n_jobs, penalty_mult=1.0), _full_mask(X.shape[1]))[0]


def fennec_pen2(X, n_jobs=-1):
    """Penalty audit (E0): 2x BIC."""
    return hill_climb(CVScore(np.asarray(X, float), model="xgb", n_jobs=n_jobs, penalty_mult=2.0), _full_mask(X.shape[1]))[0]


def fennec_pen8(X, n_jobs=-1):
    """Penalty audit (E0): 8x BIC."""
    return hill_climb(CVScore(np.asarray(X, float), model="xgb", n_jobs=n_jobs, penalty_mult=8.0), _full_mask(X.shape[1]))[0]


def fennec_screened(X, n_jobs=-1):
    nb = _lasso_neighbours(X)
    mask = (nb | nb.T) & _full_mask(X.shape[1])
    return hill_climb(CVScore(np.asarray(X, float), model="xgb", n_jobs=n_jobs, penalty_mult=1.0), mask)[0]


def linear_score(X):
    return hill_climb(CVScore(np.asarray(X, float), model="linear", penalty_mult=1.0), _full_mask(X.shape[1]))[0]


# ------------------------------------------------------------------------------ baselines
def pc(X):
    from causallearn.search.ConstraintBased.PC import pc as _pc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _cl_graph(_pc(X, 0.05, "fisherz", show_progress=False).G)


def ges(X):
    from causallearn.search.ScoreBased.GES import ges as _ges
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _cl_graph(_ges(X, score_func="local_score_BIC")["G"])


def _lasso_neighbours(X):
    d = X.shape[1]
    sel = np.zeros((d, d), bool)
    for j in range(d):
        others = [i for i in range(d) if i != j]
        coef = LassoCV(cv=5, random_state=0, n_jobs=1).fit(X[:, others], X[:, j]).coef_
        for i, c in zip(others, coef):
            sel[i, j] = abs(c) > 1e-6  # i selected as a predictor of j
    return sel


def nbsel(X):
    sel = _lasso_neighbours(X)
    return (sel & sel.T).astype(int)


def _pcor_pvalue(C, n, i, j, S):
    idx = [i, j] + list(S)
    P = np.linalg.pinv(C[np.ix_(idx, idx)])
    r = np.clip(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]), -0.999999, 0.999999)
    z = np.sqrt(max(n - len(S) - 3, 1)) * np.arctanh(r)
    return 2 * stats.norm.sf(abs(z)), abs(z)


def _mmpc(C, n, t, alpha=0.05, kmax=3):
    d = len(C)
    cpc = []
    while True:
        best, best_assoc = None, -1.0
        for x in range(d):
            if x == t or x in cpc:
                continue
            min_assoc, min_p = np.inf, 0.0
            for k in range(min(kmax, len(cpc)) + 1):
                for S in itertools.combinations(cpc, k):
                    p, z = _pcor_pvalue(C, n, x, t, S)
                    if z < min_assoc:
                        min_assoc, min_p = z, p
            if min_p < alpha and min_assoc > best_assoc:
                best, best_assoc = x, min_assoc
        if best is None:
            break
        cpc.append(best)
    for x in list(cpc):  # backward phase
        rest = [c for c in cpc if c != x]
        if any(_pcor_pvalue(C, n, x, t, S)[0] >= alpha
               for k in range(min(kmax, len(rest)) + 1) for S in itertools.combinations(rest, k)):
            cpc.remove(x)
    return set(cpc)


def mmhc(X):
    n, d = X.shape
    C = np.corrcoef(X, rowvar=False)
    pcs = [_mmpc(C, n, t) for t in range(d)]
    skel = np.zeros((d, d), bool)
    for t in range(d):
        for x in pcs[t]:
            if t in pcs[x]:  # symmetry (AND) correction
                skel[t, x] = skel[x, t] = True
    S = CVScore(np.asarray(X, float), model="linear", penalty_mult=1.0)
    return hill_climb(S, skel)[0]


def lingam(X):
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    m = DirectLiNGAM()
    m.fit(X)
    B = m.adjacency_matrix_  # B[i, j]: effect of j on i
    return (np.abs(B.T) > 1e-8).astype(int)


def r2_sortnregress(X):
    """Artefact check (Reisach et al. 2023): order variables by R^2 from all others (ascending),
    then lasso-regress each on its predecessors. Doing well here means the simulation leaks the
    causal order through predictability, not that the data identify it."""
    from sklearn.linear_model import LassoLarsIC, LinearRegression
    n, d = X.shape
    r2 = [LinearRegression().fit(np.delete(X, j, 1), X[:, j]).score(np.delete(X, j, 1), X[:, j]) for j in range(d)]
    order = np.argsort(r2)
    adj = np.zeros((d, d), int)
    for k in range(1, d):
        j, prev = order[k], order[:k]
        coef = LassoLarsIC(criterion="bic").fit(X[:, prev], X[:, j]).coef_
        adj[prev[np.abs(coef) > 1e-8], j] = 1
    return adj


def var_sortnregress(X):
    """Artefact check (Reisach et al. 2021): order variables by marginal variance (ascending), then
    lasso-regress each on its predecessors. On standardized data this is at chance by construction."""
    from sklearn.linear_model import LassoLarsIC
    n, d = X.shape
    order = np.argsort(X.var(0) + 1e-12 * np.random.default_rng(0).random(d))  # random tie-break
    adj = np.zeros((d, d), int)
    for k in range(1, d):
        j, prev = order[k], order[:k]
        coef = LassoLarsIC(criterion="bic").fit(X[:, prev], X[:, j]).coef_
        adj[prev[np.abs(coef) > 1e-8], j] = 1
    return adj


# --------------------------------------------------------------------------- NOTEARS-MLP
def notears_mlp(X, hidden=10, lambda1=0.01, lambda2=0.01, max_iter=100, h_tol=1e-8, rho_max=1e16,
                w_threshold=0.3, seed=0):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    n, d = X.shape
    Xt = torch.tensor(X, dtype=torch.double)
    m1 = hidden
    # parameters: W1 split into positive/negative parts (bounded >= 0, so L1 is linear), b1, W2, b2
    shapes = {"w1p": (d * m1, d), "w1n": (d * m1, d), "b1": (d * m1,), "w2": (d, m1), "b2": (d,)}
    sizes = {k: int(np.prod(s)) for k, s in shapes.items()}
    rng = np.random.default_rng(seed)
    init = {"w1p": np.zeros(shapes["w1p"]), "w1n": np.zeros(shapes["w1n"]), "b1": np.zeros(d * m1),
            "w2": rng.normal(0, 1 / np.sqrt(m1), (d, m1)), "b2": np.zeros(d)}
    flat0 = np.concatenate([init[k].ravel() for k in shapes])
    diag = np.zeros(shapes["w1p"], bool)  # weight from input j into output j's hidden units
    for j in range(d):
        diag[j * m1:(j + 1) * m1, j] = True
    bounds = []
    for k in shapes:
        if k in ("w1p", "w1n"):
            bounds += [(0, 0) if dg else (0, None) for dg in diag.ravel()]
        else:
            bounds += [(None, None)] * sizes[k]

    def unpack(theta):
        out, o = {}, 0
        for k in shapes:
            out[k] = theta[o:o + sizes[k]].view(shapes[k])
            o += sizes[k]
        return out

    def adjacency(p):
        W1 = (p["w1p"] - p["w1n"]).view(d, m1, d)  # [output j, hidden m, input i]
        return torch.sqrt((W1 ** 2).sum(1)).T     # A[i, j]

    def h_of(p):
        W1 = (p["w1p"] - p["w1n"]).view(d, m1, d)
        A2 = (W1 ** 2).sum(1).T
        return torch.trace(torch.matrix_exp(A2)) - d

    def forward(p):
        H = torch.sigmoid(Xt @ (p["w1p"] - p["w1n"]).T + p["b1"]).view(n, d, m1)
        return (H * p["w2"]).sum(-1) + p["b2"]

    rho, alpha, h = 1.0, 0.0, np.inf
    theta = flat0
    for _ in range(max_iter):
        while rho < rho_max:
            def fun(th):
                t = torch.tensor(th, dtype=torch.double, requires_grad=True)
                p = unpack(t)
                loss = 0.5 / n * ((Xt - forward(p)) ** 2).sum()
                hv = h_of(p)
                obj = (loss + 0.5 * rho * hv ** 2 + alpha * hv
                       + 0.5 * lambda2 * sum((p[k] ** 2).sum() for k in ("w1p", "w1n", "w2"))
                       + lambda1 * (p["w1p"].sum() + p["w1n"].sum()))
                obj.backward()
                return obj.item(), t.grad.numpy().copy()

            res = optimize.minimize(fun, theta, jac=True, method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": 200})
            new_theta = res.x
            with torch.no_grad():
                h_new = h_of(unpack(torch.tensor(new_theta))).item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        theta, h = new_theta, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    with torch.no_grad():
        A = adjacency(unpack(torch.tensor(theta))).numpy()
    return (A > w_threshold).astype(int)


RSCRIPT = os.environ.get("FENNEC_RSCRIPT", "Rscript")


def _r_method(name, X, timeout=1800, dag=None):
    import subprocess
    import tempfile
    from pathlib import Path
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout, fdag = Path(tmp) / "x.csv", Path(tmp) / "a.csv", Path(tmp) / "g.csv"
        pd.DataFrame(X).to_csv(fin, index=False)
        extra = []
        if dag is not None:
            np.savetxt(fdag, np.asarray(dag, int), fmt="%d", delimiter=",")
            extra = [str(fdag)]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run([RSCRIPT, str(here / "r" / "methods.R"), name, str(fin), str(fout)] + extra, check=True,
                       capture_output=True, timeout=timeout, env=env)
        return np.loadtxt(fout, delimiter=",").astype(int).reshape(X.shape[1], X.shape[1])


def bidag_samples(X, n_samples=50, timeout=3600):
    """Posterior DAG samples from BiDAG partition MCMC (BGe score); list of (d x d) arrays."""
    import subprocess
    import tempfile
    from pathlib import Path
    here = Path(__file__).resolve().parent
    d = X.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "x.csv", Path(tmp) / "s.csv"
        pd.DataFrame(X).to_csv(fin, index=False)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run([RSCRIPT, str(here / "r" / "bidag.R"), str(fin), str(fout), str(n_samples)], check=True,
                       capture_output=True, timeout=timeout, env=env)
        S = np.loadtxt(fout, delimiter=",", ndmin=2)
        return [row.reshape(d, d).astype(int) for row in S]


def bidag(X):
    """BiDAG posterior summary graph: edges with posterior probability > 0.5."""
    S = bidag_samples(X)
    return (np.mean(S, axis=0) > 0.5).astype(int)


def gsf(X):
    """GES with Huang et al. (2018)'s generalized score: cross-validated likelihood with kernel
    (RKHS) regression, 10-fold, lambda 0.01 (causal-learn defaults). Cost grows ~ n^3."""
    from causallearn.search.ScoreBased.GES import ges as _ges
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _cl_graph(_ges(np.asarray(X, float), score_func="local_score_CV_general")["G"])


def cam(X):
    """CAM (Buehlmann et al. 2014) via its R package: additive nonlinear SEM, greedy search, pruning.
    CAM's spline bases are singular on variables with few distinct values, so discrete columns
    (< 10 distinct values) get small Gaussian jitter (SD 0.05 on the standardized scale) first;
    continuous columns are untouched."""
    X = np.array(X, float)
    rng = np.random.default_rng(0)
    for j in range(X.shape[1]):
        if len(np.unique(X[:, j])) < 10:
            X[:, j] = X[:, j] + rng.normal(0, 0.05, len(X))
    return _r_method("cam", X)


def mmhc_bnlearn(X):
    """bnlearn's reference MMHC (cross-check for our Python MMHC)."""
    return _r_method("mmhc_bnlearn", X)


def _jitter_discrete(X):
    """Small jitter on columns with < 10 distinct values (CAM's spline bases are singular on them; as in cam())."""
    X = np.array(X, float)
    rng = np.random.default_rng(0)
    for j in range(X.shape[1]):
        if len(np.unique(X[:, j])) < 10:
            X[:, j] = X[:, j] + rng.normal(0, 0.05, len(X))
    return X


def cam_prune(X, dag):
    """CAM's significance pruning of a DAG (R package CAM), the post-processing of SCORE, NoGAM and GraN-DAG."""
    return _r_method("prune", X, dag=dag)


def _order_to_dag(order):
    d = len(order)
    A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a + 1, d):
            A[order[a], order[b]] = 1
    return A


def _stein(X, eta_G=1e-3, eta_H=1e-3):
    """Stein estimators of the score (gradient of log p) and the diagonal of its Hessian, RBF kernel with
    median bandwidth (Rolland et al. 2022, as in their reference code)."""
    import torch
    torch.set_num_threads(1)
    X = torch.as_tensor(np.asarray(X, float))
    n = X.shape[0]
    Xd = X.unsqueeze(1) - X
    D = torch.norm(Xd, dim=2)
    s = D.flatten().median()
    K = torch.exp(-D ** 2 / (2 * s ** 2)) / s
    I = torch.eye(n, dtype=X.dtype)
    nablaK = -torch.einsum("kij,ik->kj", Xd, K) / s ** 2
    G = torch.linalg.solve(K + eta_G * I, nablaK)
    nabla2K = torch.einsum("kij,ik->kj", -1 / s ** 2 + Xd ** 2 / s ** 4, K)
    H = -G ** 2 + torch.linalg.solve(K + eta_H * I, nabla2K)
    return G.numpy(), H.numpy()


def score_sm(X):
    """SCORE (Rolland et al. 2022), our implementation: repeatedly remove the leaf = the variable whose Stein
    Hessian diagonal has the smallest variance; then CAM pruning of the full DAG over that order."""
    X = _jitter_discrete(X)
    active, order, Z = list(range(X.shape[1])), [], X.copy()
    while len(active) > 1:
        _, H = _stein(Z)
        leaf = int(np.argmin(H.var(axis=0)))
        order.append(active.pop(leaf))
        Z = np.delete(Z, leaf, axis=1)
    order = (order + active)[::-1]
    return cam_prune(X, _order_to_dag(order))


def nogam(X):
    """NoGAM (Montagna et al. 2023), our implementation: the leaf is the variable whose Stein score entry is best
    predicted from its own regression residual (kernel ridge regressions, 5-fold cross-fitted); then CAM pruning."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import cross_val_predict
    X = _jitter_discrete(X)
    active, order, Z = list(range(X.shape[1])), [], X.copy()

    def krr(F):
        from scipy.spatial.distance import pdist
        med = np.median(pdist(F[:500])) if len(F) > 1 else 1.0
        return KernelRidge(kernel="rbf", alpha=0.01, gamma=1 / (2 * max(med, 1e-6) ** 2))
    while len(active) > 1:
        G, _ = _stein(Z)
        err = []
        for j in range(Z.shape[1]):
            others = np.delete(Z, j, axis=1)
            r = Z[:, j] - cross_val_predict(krr(others), others, Z[:, j], cv=5)
            pred = cross_val_predict(krr(r[:, None]), r[:, None], G[:, j], cv=5)
            err.append(np.mean((G[:, j] - pred) ** 2))
        leaf = int(np.argmin(err))
        order.append(active.pop(leaf))
        Z = np.delete(Z, leaf, axis=1)
    order = (order + active)[::-1]
    return cam_prune(X, _order_to_dag(order))


def grandag(X, iterations=10000):
    """GraN-DAG (Lachapelle et al. 2020) via gcastle (gcastle defaults: 2 hidden layers x 10 units, 10,000
    iterations), with preliminary neighbour selection and CAM pruning as in the original paper."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    import torch
    torch.set_num_threads(1)
    from castle.algorithms import GraNDAG
    Xj = _jitter_discrete(X)
    m = GraNDAG(input_dim=Xj.shape[1], iterations=iterations, use_pns=True, random_seed=0)
    m.learn(Xj)
    A = (np.asarray(m.causal_matrix) != 0).astype(int)
    return cam_prune(Xj, A) if A.any() else A


METHODS = {"fennec": fennec, "fennec_bic": fennec_bic, "fennec_pen2": fennec_pen2, "fennec_pen8": fennec_pen8, "fennec_screened": fennec_screened, "linear_score": linear_score,
           "pc": pc, "ges": ges, "nbsel": nbsel, "mmhc": mmhc, "lingam": lingam, "notears_mlp": notears_mlp,
           "r2_sortnregress": r2_sortnregress, "var_sortnregress": var_sortnregress, "cam": cam, "mmhc_bnlearn": mmhc_bnlearn, "gsf": gsf, "bidag": bidag,
           "score_sm": score_sm, "nogam": nogam, "grandag": grandag}
PARALLEL_SAFE = {"linear_score", "pc", "ges", "nbsel", "mmhc", "lingam", "notears_mlp", "r2_sortnregress", "var_sortnregress",
                 "cam", "mmhc_bnlearn", "gsf", "bidag", "score_sm", "nogam", "grandag"}  # single-threaded
