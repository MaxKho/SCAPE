"""
SCAPE-T: Semantic Complexity from Attention Patterns in Encoders - encoder-only Transformer variant
"""

from .core import scapet, pairwise_accuracy
from .datasets import (
    get_all_datasets, get_dev_splits, get_test_splits,
    get_gau_split, get_abstractness_dict
)
from .eval import build_results_table
from .ablations import (
    run_submetric_ablation_dev, run_layer_ablation_dev,
    run_submetric_ablation_4v12, draw_layer_boxes
)
from .hyperopt import evolutionary_search
from .resources import ABBREVIATIONS, DELIMITERS, STOPWORDS, PUNCT, DEFAULT_PARAMS

__all__ = [
    "scapet", "pairwise_accuracy",
    "get_all_datasets", "get_dev_splits", "get_test_splits", "get_gau_split", "get_abstractness_dict",
    "build_results_table",
    "run_submetric_ablation_dev", "run_layer_ablation_dev", "run_submetric_ablation_4v12", "draw_layer_boxes",
    "evolutionary_search",
    "ABBREVIATIONS", "DELIMITERS", "STOPWORDS", "PUNCT", "DEFAULT_PARAMS"
]
