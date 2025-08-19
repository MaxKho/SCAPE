"""
resources.py
------------

Centralised text normalisation and default hyperparameters for SCAPE-T.

This module defines built‑in resources (e.g., abbreviation expansions) and
exposes them alongside common preprocessing constants.

Exposed constants:
- ABBREVIATIONS : list of (pattern, replacement) regex pairs, loaded from Abbreviations.json
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

import string

ABBREVIATIONS = [["&c\\s*\\.","and so forth "],
                 ["&c\\s*\\.\\s*([A-Z])", "and so forth . \\1"],
                 ["([.,;:?!])[.,;:?!]+", "\\1"],
                 ["Dr\\.", "Doctor "],
                 ["Jr\\.", "Junior "],
                 ["Mr\\.", "Mr "],
                 ["Mrs\\.", "Mrs "],
                 ["Ms\\.", "Ms "],
                 ["Prof\\.", "Professor "],
                 ["Sr\\.", "Senior "],
                 ["St\\.", "Saint "],
                 ["\\b([ivxIVX]+)\\.", "\\1 "],
                 ["\\bP\\.", "Page "],
                 ["\\bVol\\.", "volume "],
                 [ "\\bcf\\.", "cf "],
                 ["\\bch\\.", "chapter "],
                 ["\\bchap\\.", "chapter "],
                 ["\\be\\.\\s*g\\.", "for example "],
                 ["\\bed\\.", "edition "],
                 ["\\bedd\\.", "editions "],
                 ["\\bi\\.\\s*e\\.", "that is, "],
                 ["\\bl\\.", "line "],
                 ["\\bp\\.", "page "],
                 ["\\bpp\\.", "pages "],
                 ["\\bpref\\.", "preface "],
                 ["\\bviz\\.", "namely "],
                 ["\\bvol\\.", "volume "],
                 ["etc\\s*\\.", "and so forth"],
                 ["etc\\s*\\.\\s*([A-Z])", "and so forth . \\1"]]
DELIMITERS = {".", "!", "?"}
STOPWORDS = {"a", "an", "the", "that", "and", "very", "will", "just"}
PUNCT = str.maketrans("", "", string.punctuation.replace("'", ""))
DEFAULT_PARAMS = {"alpha": 0.35, "beta": 3.25, "gamma": 0.4, "delta": 10.0,
                  "zeta": 0.55, "kappa": 0.1, "eta": 1.0, "tau": 0.9,
                  "mu": 6.75, "sigma": 1.3}
