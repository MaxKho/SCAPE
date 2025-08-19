"""
scape: Semantic Complexity & Analysis Package

Subpackages:
- scape.scape_t   : SCAPE-T (encoder-only Transformer variant)
- scape.datasets  : Bundled datasets (importable variables)

Convenience re-exports:
- dataset builders and helpers from scape.dataset_builder
"""

from . import scape_t, datasets

# Re-export dataset builder helpers at top level
from .dataset_builder import (
    get_dev_splits,
    get_test_splits,
    get_gau_split,
    get_all_datasets,
    bert_token_length,
)

__all__ = [
    "scape_t",
    "datasets",
    "get_dev_splits",
    "get_test_splits",
    "get_gau_split",
    "get_all_datasets",
    "bert_token_length",
]
