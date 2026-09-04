"""
models/__init__.py
==================
Makes the models/ directory a Python package.

All four model classes are lazy-imported here — the heavy dependencies
(Prophet, scikit-learn) are only resolved when the class is actually
instantiated, so importing this package never raises ImportError if
a library is missing.

Usage:
    from models.forecasting    import ProphetForecaster
    from models.clustering     import KMeansClusterer
    from models.classifier     import FaultClassifier
    from models.route_optimizer import DijkstraRouter
"""

# Expose top-level names via lazy try/except so the package
# remains importable even when optional heavy deps are absent.
try:
    from models.forecasting import ProphetForecaster
except ImportError:
    ProphetForecaster = None  # type: ignore

try:
    from models.clustering import KMeansClusterer
except ImportError:
    KMeansClusterer = None  # type: ignore

try:
    from models.classifier import FaultClassifier
except ImportError:
    FaultClassifier = None  # type: ignore

try:
    from models.route_optimizer import DijkstraRouter
except ImportError:
    DijkstraRouter = None  # type: ignore

__all__ = [
    "ProphetForecaster",
    "KMeansClusterer",
    "FaultClassifier",
    "DijkstraRouter",
]
