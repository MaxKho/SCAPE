"""
Evolutionary hyper-parameter search for SCAPE-T.

Public API:
- param_bounds               : dict of search ranges (aligned with DEFAULT_PARAMS)
- mutate(parent, scale=0.3)  : single Gaussian mutation step
- evolutionary_search(corpus, simple_sents, complex_sents, ...)
                             : run the search on any corpus + paired dev lists
- search_on_dev(... )        : convenience wrapper that runs on packaged Dev split
"""

from __future__ import annotations
import math, random
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from scape.scape_t.core import scape_t 
from scape.scape_t.resources import DEFAULT_PARAMS

# ---- bounds ----
param_bounds: Dict[str, Tuple[float, float]] = {
    "alpha": (0.0, 1.0),
    "beta":  (0.0, 5.0),
    "gamma": (0.0, 1.0),
    "delta": (0.0, 20.0),
    "zeta":  (0.0, 1.0),
    "kappa": (0.0, 2.0),
    "eta":   (0.0, 2.0),
    "tau":   (0.0, 1.0),
    "mu":    (1.0, 12.0),
    "sigma": (0.5, 4.0),
}

# ---- metrics ----
def _pairwise_accuracy(score_dict: Dict[str, float],
                       simple_sents: List[str],
                       complex_sents: List[str]) -> float:
    """P(score_complex > score_simple), ties=0.5."""
    simple_scores  = [score_dict[s] for s in simple_sents  if s in score_dict]
    complex_scores = [score_dict[s] for s in complex_sents if s in score_dict]
    total = len(simple_scores) * len(complex_scores)
    if total == 0:
        return float("nan")
    wins = 0.0
    for cs in complex_scores:
        for ss in simple_scores:
            if cs > ss: wins += 1.0
            elif cs == ss: wins += 0.5
    return wins / total

# ---- evolutionary algorithm (EA) primitives ----
def mutate(parent: Dict[str, float], scale: float = 0.3,
           rng: random.Random | None = None) -> Dict[str, float]:
    """Gaussian mutation per dimension, clamped to bounds."""
    r = rng or random
    child = dict(parent)
    for k, (lo, hi) in param_bounds.items():
        span = hi - lo
        proposal = parent.get(k, DEFAULT_PARAMS.get(k, (lo+hi)/2))
        proposal = proposal + r.gauss(0.0, scale * span)
        child[k] = max(lo, min(hi, proposal))
    return child

def evolutionary_search(corpus: Dict[str, int],
                        simple_sents: List[str],
                        complex_sents: List[str],
                        n_iter: int = 50,
                        n_offspring: int = 5,
                        scale: float = 0.3,
                        seed: int | None = None,
                        verbose: bool = True) -> Tuple[Dict, List[Dict]]:
    """
    Run a (μ=1, λ=n_offspring) hill-climbing EA over SCAPE-T params.

    Args:
        corpus:        dict {text -> 0/1 or any label}; keys are what gets scored
        simple_sents:  list of 'simple' items (for pairwise metric)
        complex_sents: list of 'complex' items
        n_iter:        number of parent generations
        n_offspring:   offspring per generation
        scale:         mutation scale as fraction of each bound span
        seed:          optional RNG seed
        verbose:       print improvements

    Returns:
        best:    {"params": dict, "acc": float}
        history: list of best snapshots per generation
    """
    rng = random.Random(seed)

    # initialise parent uniformly within bounds
    parent = {k: rng.uniform(lo, hi) for k, (lo, hi) in param_bounds.items()}

    parent_scores = scape_t(corpus, p=parent)
    parent_acc = _pairwise_accuracy(parent_scores, simple_sents, complex_sents)
    best = {"params": dict(parent), "acc": parent_acc}
    history = [dict(best)]

    for _ in tqdm(range(n_iter), desc="Evolutionary search", disable=not verbose):
        improved = False
        for _ in range(n_offspring):
            child = mutate(parent, scale=scale, rng=rng)
            scores = scape_t(corpus, p=child)
            acc = _pairwise_accuracy(scores, simple_sents, complex_sents)
            if acc > parent_acc:
                parent, parent_acc = child, acc
                best = {"params": dict(child), "acc": acc}
                history.append(dict(best))
                improved = True
                if verbose:
                    print(f" New best ACC={acc:.4f}, params={child}")
                break
        if not improved:
            history.append(dict(best))

    return best, history

# ---- convenience: run on packaged Dev split ----
def search_on_dev(n_iter: int = 50,
                  n_offspring: int = 5,
                  scale: float = 0.3,
                  seed: int | None = None,
                  verbose: bool = True) -> Tuple[Dict, List[Dict]]:
    """
    Convenience wrapper that uses packaged Dev_simple/Dev_complex.
    """
    # lazy import to avoid package init cycles
    from scape import datasets
    Dev_simple  = datasets.Dev_simple
    Dev_complex = datasets.Dev_complex

    dev_corpus = {s: 0 for s in Dev_simple}
    dev_corpus.update({s: 1 for s in Dev_complex})

    return evolutionary_search(
        corpus=dev_corpus,
        simple_sents=Dev_simple,
        complex_sents=Dev_complex,
        n_iter=n_iter,
        n_offspring=n_offspring,
        scale=scale,
        seed=seed,
        verbose=verbose)
