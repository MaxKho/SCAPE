"""
Ablation studies for SCAPE-T.

This module provides three public functions:

1) submetric_ablation_dev(...)
   – Runs submetric ablations (α, β, δ, η → set to 0) on the development split
     (Dev_simple vs Dev_complex) and reports pairwise accuracy deltas.

2) submetric_ablation_abstractness(...)
   – Runs the same submetric ablations on the Abstractness_Test set, evaluated
     as Class 4 vs Classes 1&2 with tie=0.5.

3) layer_ablation_dev(...)
   – Performs per-layer ablations on the development split (removes a single
     encoder layer at a time via `layer_abl`) and reports accuracy drops.

All functions default to using the packaged datasets (from `scape.datasets`)
and the default SCAPE-T parameters. They do **not** rebuild datasets.

Returns:
- submetric_ablation_dev(params) -> (baseline_acc: float, results_df: pd.DataFrame)
- submetric_ablation_abstractness(params) -> (baseline_acc: float, results_df: pd.DataFrame)
- layer_ablation_dev(params) -> (layer_df: pd.DataFrame, drops: List[float])

Notes:
- Pairwise accuracy is computed the same way as in `eval.py`.
- `scape.datasets` are lazily imported inside functions to avoid any import loops.
"""

# ---- imports ----
from __future__ import annotations
from typing import Dict, List, Tuple
import math, gc
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib import patches
from scape.scape_t.core import scape_t, get_model, get_tokeniser, device as scape_t_device
from scape.scape_t.resources import DEFAULT_PARAMS

# ---- local helpers ----
def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _pairwise_accuracy(score_dict: Dict[str, float],
                       simple_sents: List[str],
                       complex_sents: List[str]) -> float:
    """Pairwise accuracy: P(score_complex > score_simple) with 0.5 for ties."""
    simple_scores  = [score_dict[s] for s in simple_sents  if s in score_dict]
    complex_scores = [score_dict[s] for s in complex_sents if s in score_dict]
    total = len(simple_scores) * len(complex_scores)
    if total == 0:
        return float("nan")
    correct = 0.0
    for cs in complex_scores:
        for ss in simple_scores:
            if cs > ss:   correct += 1.0
            elif cs == ss: correct += 0.5
    return correct / total

def _acc_4_vs_12_from_lists(scores: List[float], cats: List[int]) -> float:
    """Class 4 vs Classes 1&2, tie = 0.5 (used for Abstractness_Test)."""
    arr  = np.asarray(scores, dtype=float)
    cats = np.asarray(cats, dtype=int)
    mask4  = (cats == 4) & ~np.isnan(arr)
    mask12 = ((cats == 1) | (cats == 2)) & ~np.isnan(arr)
    s4  = arr[mask4]
    s12 = arr[mask12]
    if s4.size == 0 or s12.size == 0:
        return float("nan")
    diff   = s4[:, None] - s12[None, :]
    wins   = (diff > 0).sum()
    ties   = (diff == 0).sum()
    total  = diff.size
    return (wins + 0.5 * ties) / total

# ---- submetric ablation on dev set ----
def submetric_ablation_dev(params: Dict | None = None) -> Tuple[float, pd.DataFrame]:
    """
    Run submetric ablations (α, β, δ, η → 0) on the Dev split.

    Returns:
        baseline_acc: float
        df: DataFrame with columns ["Ablation", "Pairwise Acc.", "Δ from Baseline"]
    """
    if params is None:
        params = DEFAULT_PARAMS

    # Lazy import packaged datasets to avoid any import loops.
    from scape import datasets 
    Dev_simple  = datasets.Dev_simple
    Dev_complex = datasets.Dev_complex

    # Build a dev corpus mapping (text -> class 0/1) for scoring
    dev_corpus = {s: 0 for s in Dev_simple}
    dev_corpus.update({s: 1 for s in Dev_complex})

    # Baseline
    scores_base = scape_t(dev_corpus, p=params)  # dict: text -> score
    baseline_acc = _pairwise_accuracy(scores_base, Dev_simple, Dev_complex) * 100

    ablations = {
        "alpha": "No head redundancy (α=0)",
        "beta":  "No CLS focus (β=0)",
        "delta": "No self-focus penalty (δ=0)",
        "eta":   "Uniform layer weights (η=0)",
    }

    rows = []
    for k, label in ablations.items():
        q = {**params, k: 0.0}
        d = scape_t(dev_corpus, p=q)
        acc = _pairwise_accuracy(d, Dev_simple, Dev_complex) * 100
        rows.append((label, acc, acc - baseline_acc))

    df = (pd.DataFrame(rows, columns=["Ablation", "Pairwise Acc.", "Δ from Baseline"])
            .sort_values("Pairwise Acc.", ascending=False, na_position="last")
            .reset_index(drop=True)
            .round(2))
    return baseline_acc, df

# ---- submetric ablation on abstractness set (4 vs 1&2) ----
def submetric_ablation_abstractness(params: Dict | None = None) -> Tuple[float, pd.DataFrame]:
    """
    Run submetric ablations (α, β, δ, η → 0) on Abstractness_Test, evaluated as
    Class 4 vs Classes 1&2 with tie=0.5.

    Returns:
        baseline_acc: float
        df: DataFrame with columns
            ["Ablation", "Pairwise Acc. (4 vs 1&2)", "Δ from Baseline"]
    """
    if params is None:
        params = DEFAULT_PARAMS

    from scape import datasets 

    Abstractness_Test = datasets.Abstractness_Test  
    items = list(Abstractness_Test.keys())
    cats  = [Abstractness_Test[s] for s in items]

    # Baseline scores on the abstractness dictionary (keys are scored)
    scores_map = scape_t(Abstractness_Test, p=params)
    base_scores = [scores_map.get(s, float("nan")) for s in items]
    baseline_acc = _acc_4_vs_12_from_lists(base_scores, cats) * 100

    ablations = {
        "alpha": "No head redundancy (α=0)",
        "beta":  "No CLS focus (β=0)",
        "delta": "No self-focus penalty (δ=0)",
        "eta":   "Uniform layer weights (η=0)",
    }

    rows = []
    for k, label in ablations.items():
        q = {**params, k: 0.0}
        s_map = scape_t(Abstractness_Test, p=q)
        vals  = [s_map.get(s, float("nan")) for s in items]
        acc_q = _acc_4_vs_12_from_lists(vals, cats) * 100
        rows.append((label, acc_q, acc_q - baseline_acc))

    df = (pd.DataFrame(rows,
                       columns=["Ablation", "Pairwise Acc. (4 vs 1&2)", "Δ from Baseline"])
            .sort_values("Pairwise Acc. (4 vs 1&2)", ascending=False, na_position="last")
            .reset_index(drop=True)
            .round(2))
    return baseline_acc, df

# ---- per-layer ablation on Dev set ----

# Visualisation helper
def draw_layer_boxes(deltas, fname="layer_importance_boxes.png"):
    vals = np.array(
        [0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in deltas],
        dtype=float
    )
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if math.isclose(vmin, vmax, rel_tol=1e-9, abs_tol=1e-12):
        vmin, vmax = 0.0, 1.0
        vals = np.zeros_like(vals)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis_r

    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(vals) + 2), 1.7))
    ax.set_xlim(0, len(vals))
    ax.set_ylim(0, 1)
    ax.axis('off')

    for i, v in enumerate(vals):
        rect = patches.Rectangle((i, 0), 1, 1,
                                 facecolor=cmap(norm(v)),
                                 edgecolor='black',
                                 linewidth=1.0)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, f"L{i}\n{v:.3f}",
                ha='center', va='center', fontsize=9,
                color=('black' if norm(v) < 0.7 else 'white'))

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.invert_yaxis()
    cb.set_label('Δ accuracy when layer removed', rotation=270, labelpad=12)

    plt.tight_layout(rect=[0.02, 0.02, 0.90, 0.98])
    plt.savefig(fname, dpi=180)
    plt.show()
    print(f"[saved] {fname}")

@torch.no_grad()
def layer_ablation_dev(params: Dict | None = None) -> Tuple[pd.DataFrame, List[float]]:
    """
    Remove one encoder layer at a time (via `layer_abl`) and measure pairwise accuracy
    drops on the Dev split. Also returns the list of drops per layer (baseline − ablated).

    Returns:
        layer_df: DataFrame with columns ["Ablation", "Pairwise Acc.", "Δ from Baseline"]
        drops:    List[float] (one value per layer, >= 0 means removing the layer hurts)
    """
    if params is None:
        params = DEFAULT_PARAMS

    from scape import datasets  # lazy import

    Dev_simple  = datasets.Dev_simple
    Dev_complex = datasets.Dev_complex
    dev_corpus  = {s: 0 for s in Dev_simple}; dev_corpus.update({s: 1 for s in Dev_complex})

    # Baseline accuracy
    scores_base = scape_t(dev_corpus, p=params)
    baseline_acc = _pairwise_accuracy(scores_base, Dev_simple, Dev_complex) * 100

    # Find number of layers using a tiny probe
    model = get_model()
    tokenizer = get_tokeniser()
    probe = tokenizer(["hello world"], return_tensors="pt").to(scape_t_device)
    L = len(model(**probe).attentions)
    del probe
    _cleanup_cuda()

    rows, deltas = [], []
    for li in range(L):
        s_li = scape_t(dev_corpus, p=params, layer_abl=li)
        acc_li = _pairwise_accuracy(s_li, Dev_simple, Dev_complex) * 100
        delta = acc_li - baseline_acc
        deltas.append(delta)
        rows.append((f"Layer {li} removed", acc_li, acc_li - baseline_acc))

    draw_layer_boxes(deltas)

    layer_df = pd.DataFrame(rows, columns=["Ablation", "Pairwise Acc.", "Δ from Baseline"]).reset_index(drop=True).round(2)
    return layer_df, deltas
