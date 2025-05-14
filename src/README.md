# Source Code (src)

This directory contains all the Python source code for the project, organized into functional modules.

- `data_ingestion/`: Scripts for collecting and fetching data from various sources.
- `data_processing/`: Scripts for cleaning, transforming, and consolidating raw data.
- `features/`: Scripts dedicated to feature engineering.
  - `features/deprecated/`: Older or superseded feature engineering scripts.
- `data_validation/`: Scripts for data quality checks and validation.
- `models/`: Scripts for training, evaluating, and making predictions with machine learning models.
- `utils/`: Utility functions and classes shared across multiple modules.
- `visualization/`: Scripts for generating visualizations (if not primarily done in notebooks).

Each subdirectory typically contains an `__init__.py` file to be treated as a Python package.
Refer to the `README.md` in each subdirectory for module-specific details. 