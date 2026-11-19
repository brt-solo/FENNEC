"""Graph comparison metrics for the structure benchmark."""
import numpy as np


def graph_metrics(pred, truth):
    """Compare a predicted graph to a ground-truth DAG.

    pred: (n x n) 0/1 matrix; an undirected edge (from a CPDAG) is pred[i,j] = pred[j,i] = 1.
    Returns skeleton precision/recall/F1, directed TP/precision/recall (undirected edges never
    count as correctly oriented), standard SHD (missing/extra skeleton edge = 1, wrong or
    missing orientation = 1) and, for fully directed predictions, the v1 "legacy" SHD (FP + FN
    over directed edges, so a reversal costs 2).
    """
    pred, truth = np.asarray(pred, int), np.asarray(truth, int)
    sk_p, sk_t = np.triu(pred | pred.T, 1), np.triu(truth | truth.T, 1)
    sk_tp = int((sk_p & sk_t).sum())
    directed = pred & (1 - pred.T)
    undirected_pairs = int(np.triu(pred & pred.T, 1).sum())
    d_tp = int((directed & truth).sum())
    n_pred_edges = int(sk_p.sum())
    shd = 0
    n = len(pred)
    for i in range(n):
        for j in range(i + 1, n):
            p, t = (pred[i, j], pred[j, i]), (truth[i, j], truth[j, i])
            if (p != (0, 0)) != (t != (0, 0)) or (p != (0, 0) and p != t):
                shd += 1
    out = {
        "n_edges": n_pred_edges, "n_undirected": undirected_pairs,
        "skel_tp": sk_tp,
        "skel_precision": sk_tp / n_pred_edges if n_pred_edges else 0.0,
        "skel_recall": sk_tp / int(sk_t.sum()),
        "dir_tp": d_tp,
        "dir_precision": d_tp / n_pred_edges if n_pred_edges else 0.0,
        "dir_recall": d_tp / int(truth.sum()),
        "shd": shd,
    }
    out["skel_f1"] = (2 * out["skel_precision"] * out["skel_recall"] / (out["skel_precision"] + out["skel_recall"])
                      if sk_tp else 0.0)
    out["legacy_shd"] = (int((directed & (1 - truth)).sum() + (truth & (1 - directed)).sum())
                         if undirected_pairs == 0 else None)
    return out
