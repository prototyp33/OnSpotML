# Barcelona Data Collection Module

This module provides functionality to collect and integrate various types of data from Barcelona's open data sources and APIs.

## Features

- Collects parking data from Barcelona's open data portal
- Fetches weather data from Meteocat API
- Downloads transport data from TMB (Barcelona's public transport)
- Gathers event data from Barcelona's open data portal
- Integrates collected data into a unified dataset

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

**IMPORTANT:** This script requires API keys for TMB (Transport) and WeatherAPI.com (Weather) to function correctly. Configuration is handled via two files:

1.  **`pass.json` (for TMB Credentials):**
    *   Create a file named `pass.json` in the project root.
    *   Use the following structure, replacing the placeholders in the `"local"` section with your actual TMB App ID and Key obtained from the [TMB Developer Portal](https://www.tmb.cat/en/dades-obertes):
    ```json
    {
      "github": {
        "app_key": "d9905ec9b70bc6aca11e39be3cd0d856",
        "app_id": "df5c473f"
      },
      "local": {
        "app_key": "YOUR_VALID_TMB_KEY",
        "app_id": "YOUR_VALID_TMB_ID"
      }
    }
    ```

2.  **`.env` (for WeatherAPI Key & BCN Token):**
    *   Create a file named `.env` in the project root.
    *   Add your WeatherAPI key obtained from [WeatherAPI.com](https://www.weatherapi.com/).
    *   Add your Open Data BCN Access Token (if required for certain datasets, obtain from the portal after registration).
    ```dotenv
    # .env file
    WEATHER_API_KEY=YOUR_VALID_WEATHERAPI_KEY
    BCN_OD_TOKEN=YOUR_BCN_OPEN_DATA_ACCESS_TOKEN
    ```

The script will validate these configuration files on startup and will raise an error if they are missing or incorrectly formatted.

## Usage

To collect all data:

```