"""
data/__init__.py
================
Makes the data/ directory a Python package so all submodules can be
imported with the canonical prefix:

    from data.data_extractor  import DataExtractor
    from data.data_cleaner    import DataCleaner
    from data.feature_engineer import FeatureEngineer
    from data.synthetic_data  import SyntheticDataGenerator

Public re-exports for convenience (allows `from data import DataExtractor`):
"""

from data.data_extractor   import DataExtractor
from data.data_cleaner     import DataCleaner
from data.feature_engineer import FeatureEngineer
from data.synthetic_data   import SyntheticDataGenerator

__all__ = [
    "DataExtractor",
    "DataCleaner",
    "FeatureEngineer",
    "SyntheticDataGenerator",
]
