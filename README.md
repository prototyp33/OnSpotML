# OnSpotML: Barcelona Parking Occupancy Prediction

OnSpotML is a project focused on predicting parking occupancy in Barcelona. It encompasses the entire MLOps lifecycle, including data collection from various open sources, feature engineering, model training, and potentially real-time prediction (though the latter part's implementation details are not fully specified here).

## Features

*   **Data Collection:**
    *   Collects parking availability data (e.g., from Barcelona's Open Data portal).
    *   Fetches weather forecasts and historical data.
    *   Downloads public transport information (GTFS feeds from TMB).
    *   Gathers data on local events and holidays.
*   **Feature Engineering:**
    *   Generates temporal features (time of day, day of week, holidays).
    *   Integrates weather conditions.
    *   Calculates spatial features (e.g., proximity to points of interest, DBSCAN clustering of locations).
    *   Incorporates GTFS data (e.g., nearest public transport stops, stop density).
*   **Modeling:**
    *   Supports training baseline models (e.g., LightGBM).
    *   (Potentially supports other model types and evaluation).
*   **Data Integration:** Consolidates all collected and engineered features into a master dataset for model training.

## Project Structure

A brief overview of key directories:

*   `.github/workflows/`: Contains CI/CD workflows (e.g., running tests automatically).
*   `config/`: For configuration files (though API keys are managed via `pass.json` and `.env`).
*   `data/`: Stores raw, processed, and interim datasets.
    *   `data/raw/`: Original, unmodified data collected from sources.
    *   `data/processed/`: Cleaned and processed data ready for feature engineering or modeling.
*   `docs/`: Project documentation, architectural diagrams, or detailed explanations.
*   `models/`: Saved trained machine learning models.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA), experimentation, and visualization.
*   `reports/`: Generated reports, figures, or model evaluation summaries.
*   `src/`: All source code for the project.
    *   `src/data_ingestion/`: Scripts for collecting data from various sources.
    *   `src/data_processing/`: Scripts for cleaning and pre-processing raw data.
    *   `src/features/`: Scripts for feature engineering (e.g., `build_features.py`).
    *   `src/modeling/`: Scripts for training, evaluating, and (potentially) serving models.
    *   `src/utils/`: Utility functions shared across the project.
    *   `src/visualization/`: Scripts for generating visualizations.
*   `tests/`: Automated tests (unit, integration).
    *   `tests/sample_data/`: Sample data used by the tests.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/yourusername/OnSpotML.git # Replace with actual URL
    cd OnSpotML
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

**IMPORTANT:** This project requires API keys and tokens to access various data sources. Please configure them as follows:

1.  **`pass.json` (for TMB Transport Credentials):**
    *   Create a file named `pass.json` in the project root.
    *   Use the following structure, replacing placeholders with your actual TMB App ID and Key from the [TMB Developer Portal](https://www.tmb.cat/en/dades-obertes):
        ```json
        {
          "github": {
            "app_key": "d9905ec9b70bc6aca11e39be3cd0d856", // Example, not for use
            "app_id": "df5c473f" // Example, not for use
          },
          "local": {
            "app_key": "YOUR_VALID_TMB_KEY",
            "app_id": "YOUR_VALID_TMB_ID"
          }
        }
        ```

2.  **`.env` (for WeatherAPI Key & BCN Open Data Token):**
    *   Create a file named `.env` in the project root.
    *   Add your WeatherAPI key from [WeatherAPI.com](https://www.weatherapi.com/).
    *   Add your Open Data BCN Access Token if required for specific datasets (obtain from the portal after registration).
        ```dotenv
        # .env file
        WEATHER_API_KEY=YOUR_VALID_WEATHERAPI_KEY
        BCN_OD_TOKEN=YOUR_BCN_OPEN_DATA_ACCESS_TOKEN
        ```
The scripts typically validate these configurations on startup.

## Usage

The project involves several key steps. Here's how to run the main scripts for each stage:

1.  **Data Collection:**
    *   To collect all relevant data (parking, weather, events, transport):
        ```bash
        python src/data_ingestion/barcelona_data_collector.py
        ```
    *   (Note: You might need to run specific collectors individually if `barcelona_data_collector.py` doesn't orchestrate all of them or if you want to refresh specific data.)

2.  **Feature Engineering:**
    *   To process raw data and generate the master feature table:
        ```bash
        python src/features/build_features.py
        ```
    *   This script will use data from `data/processed/` (like `parking_predictions_with_pois.parquet`) and raw sources (weather, events, GTFS) to create `data/processed/features_master_table.parquet`.

3.  **Model Training:**
    *   To train a baseline model (example):
        ```bash
        python src/modeling/train_baseline.py
        ```
    *   (Refer to specific scripts in `src/modeling/` for other models or training routines.)

4.  **(Prediction - if applicable):**
    *   (Instructions for running prediction scripts would go here if they exist, e.g., `python src/predict.py --input-data <path_to_new_data>`)

Please refer to individual scripts or their corresponding READMEs (if any in `src/` subdirectories) for more detailed options.

## Running Tests

To run the automated tests for the project:

```bash
pytest tests/
```
Or, to include coverage reporting (as configured in CI):
```bash
pytest --cov=src --cov-report=html tests/
```
(The CI uses `pytest --cov=./ --cov-report=xml`. For local viewing, `html` is often more convenient. Ensure `pytest-cov` is installed: `pip install pytest-cov`)

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to report bugs, propose features, and submit pull requests.

## License

This project is licensed under the MIT License. See the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details, and consider adding a `LICENSE` file to the repository root with the full license text.
