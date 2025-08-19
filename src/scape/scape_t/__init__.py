"""
SCAPE-T: Semantic Complexity from Attention Patterns in Encoders
(encoder-only Transformer variant)
"""

from .core import get_model, get_tokeniser, device, scape_t
from .resources import ABBREVIATIONS, DELIMITERS, STOPWORDS, PUNCT, DEFAULT_PARAMS
from .eval import build_results_table, ranking
from .ablations import submetric_ablation_dev, submetric_ablation_abstractness, layer_ablation_dev
from .tuning import param_bounds, mutate, evolutionary_search, search_on_dev

__all__ = [
    "get_model",
    "get_tokeniser",
    "device",
    "scape_t",
    "ABBREVIATIONS",
    "DELIMITERS",
    "STOPWORDS",
    "PUNCT",
    "DEFAULT_PARAMS",
    "build_results_table",
    "ranking",
    "submetric_ablation_dev",
    "submetric_ablation_abstractness",
    "layer_ablation_dev",
    "param_bounds", 
    "mutate", 
    "evolutionary_search", 
    "search_on_dev"
]
