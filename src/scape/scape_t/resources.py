"""
resources.py
------------

Centralised text normalisation resources and default hyperparameters for SCAPE-T.

This module loads external JSON data files (e.g. abbreviation expansions) and 
exposes them alongside common preprocessing constants. It is designed to work 
both during development (loading from `src/data/` or a top-level `data/` 
directory, or current working directory) and after installation (using packaged 
data if included).

Exposed constants:
- ABBREVIATIONS : list of (pattern, replacement) regex pairs, loaded from abbreviations.json
- DELIMITERS    : set of sentence boundary characters
- STOPWORDS     : set of semantically uninformative function words to ignore in analysis
- PUNCT         : translation table for stripping punctuation (excluding apostrophes)
- DEFAULT_PARAMS: dict of default SCAPE-T hyperparameters

Search order for JSON data:
1. SCAPE_DATA_DIR environment variable (explicit override)
2. Packaged data inside scape.scape_t/data (if installed with package data)
3. Project-root/src/data (when working in src/ layout)
4. Project-root/data (legacy layout without src/)
5. Current working directory and cwd/data (for Colab or ad-hoc runs)

Usage example:
    from scape.scape_t.resources import ABBREVIATIONS, DEFAULT_PARAMS
    print(ABBREVIATIONS[:3])
    print(DEFAULT_PARAMS["alpha"])
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import json, os, string
from pathlib import Path

try:
    from importlib.resources import files
    _PKG_DATA = files("scape.scape_t").joinpath("data")
    _HAS_PKG_DATA = _PKG_DATA.is_dir()
except Exception:
    _PKG_DATA = None
    _HAS_PKG_DATA = False

def _candidate_data_dirs() -> list[Path]:
    """
    Search order:
    - packaged data: scape.scape_t/data (when installed)
    - project-root/src/data (development in src/ layout)
    - project-root/data (legacy layout)
    - current working directory (Colab / ad-hoc)
    """
    cands = []
    if _HAS_PKG_DATA:
        cands.append(_PKG_DATA)

    # project-root/src/data and project-root/data
    try:
        here = Path(__file__).resolve()
        root = here.parents[3] if len(here.parents) >= 4 else here.parents[-1]
        cands.append(root / "src" / "data")
        cands.append(root / "data")
    except Exception:
        pass

    # cwd
    cands.append(Path(os.getcwd()))
    cands.append(Path(os.getcwd()) / "data")

    # de-dup while preserving order
    out, seen = [], set()
    for p in cands:
        if p is None:
            continue
        q = p.resolve()
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out

def _load_json(filename: str):
    for d in _candidate_data_dirs():
        path = d / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Could not find '{filename}'. Looked in: "
        + ", ".join(str(d) for d in _candidate_data_dirs()))


ABBREVIATIONS = [(pat, repl) for pat, repl in _load_json("abbreviations.json")]
DELIMITERS = {".", "!", "?"}
STOPWORDS = {"a", "an", "the", "that", "and", "very", "will", "just"}
PUNCT = str.maketrans("", "", string.punctuation.replace("'", ""))

DEFAULT_PARAMS = {"alpha": 0.35, "beta": 3.25, "gamma": 0.4, "delta": 10.0,
                  "zeta": 0.55, "kappa": 0.1, "eta": 1.0, "tau": 0.9,
                  "mu": 6.75, "sigma": 1.3}
