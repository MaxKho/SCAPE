"""
SCAPE-T: Semantic Complexity from Attention Patterns in Encoders - encoder-only Transformer variant
"""

from .core import get_model, get_tokeniser, device, scape_t
from .resources import ABBREVIATIONS, DELIMITERS, STOPWORDS, PUNCT, DEFAULT_PARAMS, ABSTRACTNESS
from .dataset_builder import get_dev_splits, get_test_splits, get_gau_split, get_all_datasets, bert_token_length

__all__ = ["get_model", "get_tokeniser", "device", "scape_t",
           "ABBREVIATIONS", "DELIMITERS", "STOPWORDS", "PUNCT", "DEFAULT_PARAMS", "ABSTRACTNESS",
           "get_dev_splits, get_test_splits, get_gau_split, get_all_datasets, bert_token_length"]
