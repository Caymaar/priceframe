"""
PriceFrame - A modern, efficient price data management library.

PriceFrame provides a canonical way to store, manipulate, and analyze financial price data
using PyArrow, Polars, and pandas. It supports OHLCV data with a flexible feature system
for technical indicators.

Main components:
    - PriceFrame: Core data structure for price data
    - FeatureSpec: Declarative feature specification
    - Feature registry: Extensible system for technical indicators
"""

from .core import PriceFrame, FeatureSpec
from .features import list_registered_features, feature

__all__ = ["PriceFrame", "FeatureSpec", "list_registered_features", "feature"]
