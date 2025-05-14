# Feature Engineering Scripts

This module is dedicated to scripts that generate features for machine learning models from the processed data (usually from `data/interim/`).

**Key Scripts:**
- `1_prepare_base_features.py`: Generates temporal, lag, weather, holiday, and event features.
- `2_add_poi_features_batched.py`: Adds Point of Interest (POI) features in a batched manner.
- `gtfs_features.py`: Contains logic for engineering features from GTFS data.

The `deprecated/` subdirectory contains older or superseded feature engineering scripts that are kept for reference but are no longer part of the main pipeline. 