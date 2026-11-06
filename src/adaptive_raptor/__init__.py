"""Adaptive-RAPTOR: dynamic chunking and query-aware context optimization on RAPTOR.

Importing this package is lightweight: it never imports torch, sentence-transformers,
or the upstream ``raptor`` package. Heavy modules are imported lazily where needed.
"""

__version__ = "0.1.0"
