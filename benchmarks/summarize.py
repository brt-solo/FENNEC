"""Summarise benchmark results: skeleton F1 and oriented-edge recall per method, mechanism and sample size.

    python -m benchmarks.summarize                  # benchmarks/results/results.jsonl (signal:noise 4:1)
    python -m benchmarks.summarize --noise-sd 1     # benchmarks/results/results_noise1.jsonl (1:1)

The FENNEC penalty multiplier (4x BIC) was chosen on seeds 0-4 (plus Sachs), so the default report
uses only the held-out seeds 5-9. Pass --seeds all to pool seeds 0-9.
Writes results/tables/benchmark_<tag>.csv and results/figures/benchmark_<tag>.png.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# older result files used internal method names
LEGACY = {"main_xgb_pen4": "fennec", "main_xgb": "fennec_bic", "main_xgb_screened": "fennec_screened",
          "main_linear": "linear_score"}
LABEL = {"fennec": "FENNEC (XGBoost CV score, 4x BIC)", "fennec_bic": "FENNEC score, 1x BIC",
         "fennec_screened": "FENNEC score, screened", "linear_score": "linear CV score",
         "cam": "CAM", "bidag": "BiDAG (MCMC)", "nogam": "NoGAM", "score_sm": "SCORE", "grandag": "GraN-DAG",
         "notears_mlp": "NOTEARS-MLP", "ges": "GES (BIC)", "gsf": "GES + generalized score", "pc": "PC",
         "mmhc": "MMHC", "mmhc_bnlearn": "MMHC (bnlearn)", "lingam": "DirectLiNGAM", "nbsel": "neighbourhood selection",
         "r2_sortnregress": "R²-sortnregress"}
MECH = ["linear", "additive", "mlp", "mixed"]


def load(path, seeds):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    d = pd.DataFrame(rows)
    d = d[d.error.isna() & d.seed.isin(seeds)].copy()
    d["method"] = d.method.replace(LEGACY)
    d["orient_recall"] = d.dir_tp / d.true_edges.clip(lower=1)
    return d


def summarise(d):
    cols = [(m, n) for n in (200, 1000) for m in MECH]
    out = {}
    for metric in ("skel_f1", "orient_recall"):
        t = d.groupby(["method", "mechanism", "n"])[metric].mean().unstack(["mechanism", "n"])
        t = t.reindex(columns=pd.MultiIndex.from_tuples(cols, names=["mechanism", "n"]))
        t = t.loc[t.mean(axis=1).sort_values(ascending=False).index]
        t.index = [LABEL.get(i, i) for i in t.index]
        out[metric] = t
    return out


def plot(t, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(t) + 1.6))
    im = ax.imshow(t.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(t.shape[1]), [f"{m}\nn={n}" for m, n in t.columns], fontsize=8)
    ax.set_yticks(range(len(t)), t.index, fontsize=8)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-sd", type=float, default=None)
    ap.add_argument("--seeds", default="5-9", help="'5-9' (held out from penalty tuning, default) or 'all'")
    ap.add_argument("--results", type=Path, default=None)
    a = ap.parse_args()
    tag = "noise4to1" if a.noise_sd is None else f"noise{a.noise_sd:g}"
    path = a.results or HERE / "results" / ("results.jsonl" if a.noise_sd is None else f"results_noise{a.noise_sd:g}.jsonl")
    seeds = range(10) if a.seeds == "all" else range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1)
    d = load(path, seeds)
    tabs = summarise(d)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    for metric, t in tabs.items():
        t.to_csv(ROOT / "results" / "tables" / f"benchmark_{tag}_{metric}.csv")
        print(f"\n{metric} ({path.name}, seeds {a.seeds}, d = 10 and 20 pooled)\n", t.round(2).to_string())
    snr = "signal:noise 4:1" if a.noise_sd is None else f"noise sd {a.noise_sd:g} (signal:noise 1:1)"
    plot(tabs["skel_f1"], f"Skeleton F1, {snr}, seeds {a.seeds}", ROOT / "results" / "figures" / f"benchmark_{tag}.png")


if __name__ == "__main__":
    main()
