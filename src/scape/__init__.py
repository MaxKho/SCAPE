"""
scape: Semantic Complexity & Analysis Package

Subpackages / modules exposed lazily:
- scape.scape_t     → available as attribute: scape.scape_t
- scape.datasets    → available as attribute: scape.datasets
Utilities re-exported lazily:
- bert_token_length → scape.bert_token_length  (from scape.dataset_builder)
- build_results_table, ranking → scape.build_results_table / scape.ranking
                                (re-exported from scape.scape_t)
"""

from importlib import import_module
from typing import Any

__all__ = [
    "scape_t",
    "datasets",
    "bert_token_length",
    "build_results_table",
    "ranking",
]

def __getattr__(name: str) -> Any:
    # submodules
    if name == "scape_t":
        return import_module("scape.scape_t")
    if name == "datasets":
        return import_module("scape.datasets")

    # utilities
    if name == "bert_token_length":
        return getattr(import_module("scape.dataset_builder"), "bert_token_length")
    if name in {"build_results_table", "ranking"}:
        # re-export from scape.scape_t (which re-exports them from its eval.py)
        return getattr(import_module("scape.scape_t"), name)

    raise AttributeError(name)

def __dir__():
    return sorted(list(globals().keys()) + __all__)
