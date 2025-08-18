"""
SCAPE-T: Semantic Complexity from Attention Patterns in Encoders - encoder-only Transformer variant
"""

from .core import get_model, get_tokeniser, device, scape_t
from .resources import ABBREVIATIONS, DELIMITERS, STOPWORDS, PUNCT, DEFAULT_PARAMS

__all__ = [
    "scape_t", "get_model", "get_tokeniser", "device",
    "ABBREVIATIONS", "DELIMITERS", "STOPWORDS", "PUNCT", "DEFAULT_PARAMS", "ABSTRACTNESS"
]
