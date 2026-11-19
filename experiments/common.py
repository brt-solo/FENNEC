"""Shared plotting and table helpers for the experiment analyses (figures -> results/figures, tables -> results/tables)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
OUT = FIGURES
for d in (FIGURES, TABLES):
    d.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False})
MD = []


def save_fig(fig, name):
    fig.savefig(FIGURES / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("figure", FIGURES / name)


def table(df, name, title, note=""):
    df.to_csv(TABLES / f"{name}.csv")
    MD.append(f"## {name}: {title}\n\n{note}\n\n{df.to_markdown(floatfmt='.3f')}\n")
    print("table ", TABLES / f"{name}.csv")


def boot_ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    b = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])
