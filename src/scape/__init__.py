"""
scape: Semantic Complexity & Analysis Package

Subpackages:
- scape.scape_t: SCAPE-T (Encoder-only Transformer variant of SCAPE)
"""

# re-export the scape_t subpackage as a namespace
from . import scape_t

__all__ = ["scape_t"]
