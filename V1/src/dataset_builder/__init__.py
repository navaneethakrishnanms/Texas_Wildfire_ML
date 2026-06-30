"""
dataset_builder
===============
Production pipeline for building the Texas Wildfire ML training dataset.

Public API
----------
    from src.dataset_builder import (
        build_dataset, RasterSampler, generate_negative_samples,
        engineer_temporal_features, FEATURE_COLUMNS, TEMPORAL_FEATURE_COLUMNS
    )

Modules
-------
build_dataset.py
    End-to-end pipeline orchestrator. Entry-point for the dataset build.
    Includes engineer_temporal_features() which derives 8 cyclically-encoded
    seasonal features (month, day_of_year, season_code, sin_month, cos_month,
    sin_doy, cos_doy, is_peak_fire_season) from acq_date to capture Texas
    seasonal wildfire risk variance that annual raster composites cannot express.

sample_rasters.py
    RasterSampler class: windowed rasterio reads, auto-tile discovery,
    geospatially correct CRS reprojection, physical unit scaling.

generate_negatives.py
    Vectorised scipy KDTree-based negative sample generator with
    configurable spatial exclusion buffer.
"""

from src.dataset_builder.build_dataset import (
    build_dataset,
    engineer_temporal_features,
    TEMPORAL_FEATURE_COLUMNS,
    PEAK_FIRE_MONTHS,
    SEASON_MAP,
)
from src.dataset_builder.sample_rasters import RasterSampler, FEATURE_COLUMNS
from src.dataset_builder.generate_negatives import generate_negative_samples

__all__ = [
    "build_dataset",
    "engineer_temporal_features",
    "RasterSampler",
    "generate_negative_samples",
    "FEATURE_COLUMNS",
    "TEMPORAL_FEATURE_COLUMNS",
    "PEAK_FIRE_MONTHS",
    "SEASON_MAP",
]

