import os
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import zipfile
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from dotenv import load_dotenv
import numpy as np
from shapely.geometry import LineString

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_credentials():
    """Check for pass.json (TMB), BCN_OD_TOKEN (env), and WEATHER_API_KEY (env)."""
    pass_json_path = Path("pass.json")
    missing = []
    valid_pass_json = False
    
    # Check for pass.json and its structure
    if not pass_json_path.exists():
        missing.append("pass.json file (for TMB credentials)")
    else:
        try:
            with open(pass_json_path, 'r') as f:
                creds = json.load(f)
            if not isinstance(creds.get('local'), dict) or \
               not creds['local'].get('app_id') or \
               not creds['local'].get('app_key'):
                missing.append("Valid 'local' section with 'app_id' and 'app_key' in pass.json")
            else:
                valid_pass_json = True # Mark as valid if structure is okay
        except json.JSONDecodeError:
            missing.append("Valid JSON format in pass.json")
        except Exception as e:
             missing.append(f"Error reading pass.json: {e}")

    # Check for Weather API key in environment
    if not os.getenv('WEATHER_API_KEY'):
        missing.append("WEATHER_API_KEY environment variable (for WeatherAPI.com)")

    # Check for BCN Open Data Token in environment
    if not os.getenv('BCN_OD_TOKEN'):
        missing.append("BCN_OD_TOKEN environment variable (for Open Data BCN)")

    if missing:
        error_message = (
            f"Configuration errors found:\n" +
            "\n".join([f"  - {m}" for m in missing]) +
            "\nPlease check your configuration files (.env, pass.json)."
        )
        logger.error(error_message)
        raise EnvironmentError(error_message)
    else:
        logger.info("Configuration (pass.json, .env) successfully validated.")

class BarcelonaDataCollector:
    """Class for collecting and managing Barcelona data from various sources."""
    
    def __init__(self, base_dir: str = "data") -> None:
        """Initialize the data collector with base directory."""
        self.base_dir = Path(base_dir)
        self._setup_directories()
        self._load_config()
        self.meteo_cat_api_key: Optional[str] = os.getenv("METEO_CAT_API_KEY")
        self.tmb_app_id: Optional[str] = os.getenv("TMB_APP_ID")
        self.tmb_app_key: Optional[str] = os.getenv("TMB_APP_KEY")
        
    def _setup_directories(self) -> None:
        """Create necessary directories for data storage."""
        directories = ['parking', 'weather', 'transport', 'events', 'auxiliary', 'raw', 'processed']
        for dir_name in directories:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.base_dir / dir_name}")
    
    def _load_config(self):
        """Load TMB config from pass.json; Weather & BCN Token from env vars."""
        tmb_app_id = None
        tmb_app_key = None
        pass_json_path = Path("pass.json")

        # Load TMB keys from pass.json
        if pass_json_path.exists():
            try:
                with open(pass_json_path, 'r') as f:
                    creds = json.load(f)
                # Use 'local' section keys
                local_creds = creds.get('local', {})
                tmb_app_id = local_creds.get('app_id')
                tmb_app_key = local_creds.get('app_key')
                if not tmb_app_id or not tmb_app_key or tmb_app_id == 'your app id here' or tmb_app_key == 'your app key here':
                    logger.warning("TMB credentials in pass.json->local seem to be placeholders.")
                    # Decide if this should be an error or just a warning
                    # For now, allow script to continue, but API calls will likely fail
            except Exception as e:
                logger.error(f"Failed to load or parse pass.json: {e}")
        else:
             logger.error("pass.json not found. TMB API calls will fail.")

        # Load Weather key from environment
        weather_api_key = os.getenv('WEATHER_API_KEY')
        # Load BCN Open Data Token from environment
        bcn_od_token = os.getenv('BCN_OD_TOKEN')

        self.config = {
            'weather_api_key': weather_api_key,
            'tmb_app_id': tmb_app_id,      
            'tmb_app_key': tmb_app_key,     
            'bcn_od_token': bcn_od_token
        }
        logger.info(f"Loaded config - TMB App ID from pass.json: {bool(self.config['tmb_app_id'])}")
        logger.info(f"Loaded config - Weather API Key from .env: {bool(self.config['weather_api_key'])}")
        logger.info(f"Loaded config - BCN Open Data Token from .env: {bool(self.config['bcn_od_token'])}")
    
    def _validate_csv(self, filepath: Path, expected_columns: Optional[list] = None) -> bool:
        """Validate a downloaded CSV file."""
        if not filepath or not filepath.exists():
            logger.error(f"Validation failed: File not found - {filepath}")
            return False
        try:
            df = pd.read_csv(filepath, nrows=5) # Read only a few rows for validation
            if expected_columns and not all(col in df.columns for col in expected_columns):
                logger.error(f"Validation failed: Missing expected columns in {filepath}")
                missing = set(expected_columns) - set(df.columns)
                logger.error(f"Missing columns: {missing}")
                return False
            logger.info(f"Validation successful for {filepath}")
            return True
        except Exception as e:
            logger.error(f"Validation failed for {filepath}: {str(e)}")
            return False

    def download_file(
        self, 
        url: str, 
        filename: str, 
        folder: str, 
        is_zip: bool = False, 
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ) -> Optional[Path]:
        """Generic file downloader with error handling and optional ZIP extraction."""
        folder_path = self.base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        filepath = folder_path / filename

        logger.info(f"Attempting to download {filename} from {url} into {folder_path}")
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded {filename} to {filepath}")

            if is_zip or filename.lower().endswith(".zip"):
                logger.info(f"Extracting ZIP file: {filepath}")
                extract_to_path = folder_path / Path(filename).stem
                try:
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(extract_to_path)
                    logger.info(f"Successfully extracted {filename} to {extract_to_path}")
                    return extract_to_path
                except zipfile.BadZipFile:
                    logger.error(f"Error: Bad ZIP file for {filename}. Download may be incomplete or corrupted.")
                    return None
                except Exception as e:
                    logger.error(f"Error extracting ZIP {filename}: {str(e)}")
                    return None
            return filepath
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error downloading {filename}: {http_err} (Status code: {response.status_code if 'response' in locals() else 'N/A'})")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error downloading {filename}: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error downloading {filename}: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Generic error downloading {filename}: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred with {filename}: {str(e)}")
        return None

    def get_traffic_segment_geometries(self, local_csv_path: Optional[str] = None) -> Optional[Path]:
        """
        Fetches or loads traffic segment (tram) geometries.
        Prioritizes local_csv_path if provided, otherwise fetches from Barcelona's OpenData BCN CKAN API.
        Processes them into LineString objects and saves as a GeoPackage file.
        The data source is "Relació de trams de la via pública de la ciutat de Barcelona".
        Expected CRS of the source coordinates is EPSG:4326 (WGS84).

        Args:
            local_csv_path: Optional path to a local CSV file containing the tram segment data
                            in the "long format" (Tram, Tram_Components, Descripció, Longitud, Latitud).
        """
        logger.info("Starting collection/processing of traffic segment geometries.")
        
        output_filename = "trams_geometries.gpkg"
        output_path = self.base_dir / "processed" / output_filename
        df = None

        if local_csv_path:
            local_file = Path(local_csv_path)
            if local_file.exists() and local_file.is_file() and local_file.suffix.lower() == '.csv':
                logger.info(f"Attempting to load traffic segment geometries from local CSV: {local_csv_path}")
                try:
                    df = pd.read_csv(local_file)
                    # Basic validation of expected columns for local CSV
                    expected_cols = ["Tram", "Tram_Components", "Descripció", "Longitud", "Latitud"]
                    if not all(col in df.columns for col in expected_cols):
                        logger.error(f"Local CSV {local_csv_path} is missing one or more expected columns: {expected_cols}. Columns found: {df.columns.tolist()}")
                        df = None # Invalidate df to trigger API fallback or error out
                    else:
                        logger.info(f"Successfully loaded {len(df)} records from {local_csv_path}.")
                except Exception as e:
                    logger.error(f"Error reading local CSV {local_csv_path}: {e}")
                    df = None # Invalidate df
            else:
                logger.warning(f"Local CSV path provided but file not found, not a file, or not a CSV: {local_csv_path}. Will attempt API fallback.")

        if df is None: # If local loading failed or was not attempted
            logger.info("Attempting to fetch traffic segment geometries from API.")
            resource_id = "c97072a3-3619-4547-84dd-f1999d2a3fec"
            sql_query = f'SELECT * FROM "{resource_id}" ORDER BY "Tram", "Tram_Components"'
            params = {'sql': sql_query}
            api_url = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search_sql"
            
            try:
                logger.info(f"Querying API: {api_url} for resource_id: {resource_id}")
                response = requests.get(api_url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    logger.error(f"API call was not successful. Response: {data.get('error', 'Unknown API error')}")
                    return None
                    
                records = data.get("result", {}).get("records", [])
                if not records:
                    logger.warning("No records found in the API response for traffic segment geometries.")
                    return None
                
                logger.info(f"Successfully fetched {len(records)} records from the API.")
                df = pd.DataFrame(records)
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error fetching segment geometries from API: {http_err}")
                return None
            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"Connection error fetching segment geometries from API: {conn_err}")
                return None
            except requests.exceptions.Timeout as timeout_err:
                logger.error(f"Timeout error fetching segment geometries from API: {timeout_err}")
                return None
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Generic error fetching segment geometries from API: {req_err}")
                return None
            except json.JSONDecodeError:
                logger.error("Error decoding JSON response from API.")
                return None
            except KeyError as ke:
                logger.error(f"KeyError while processing API response (likely unexpected structure): {ke}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during API fetch for traffic segment geometries: {str(e)}")
                return None
        
        if df is None or df.empty:
            logger.error("No data available (either from local CSV or API) to process traffic segment geometries.")
            return None

        try:
            # Ensure correct data types, especially for coordinates and sorting keys
            # These names come from the API and the provided local CSV
            df["Tram"] = pd.to_numeric(df["Tram"], errors='coerce')
            df["Tram_Components"] = pd.to_numeric(df["Tram_Components"], errors='coerce')
            df["Longitud"] = pd.to_numeric(df["Longitud"], errors='coerce')
            df["Latitud"] = pd.to_numeric(df["Latitud"], errors='coerce')
            
            df.dropna(subset=["Tram", "Tram_Components", "Longitud", "Latitud"], inplace=True)
            
            if df.empty:
                logger.warning("DataFrame is empty after type conversion and NA drop. Cannot create geometries.")
                return None

            logger.info("Processing records into LineString geometries...")
            geometries = []
            descriptions_map = {}
            
            for tram_id, group in df.groupby("Tram"):
                group_sorted = group.sort_values("Tram_Components")
                coordinates = list(zip(group_sorted["Longitud"], group_sorted["Latitud"]))
                
                if len(coordinates) >= 2:
                    geometries.append({
                        "ID_TRAM": int(tram_id),
                        "geometry": LineString(coordinates)
                    })
                    descriptions_map[int(tram_id)] = group_sorted["Descripció"].iloc[0] if not group_sorted["Descripció"].empty else ""
                else:
                    logger.warning(f"Tram ID {tram_id} has less than 2 valid coordinate pairs. Skipping.")

            if not geometries:
                logger.warning("No valid geometries could be created from the fetched/loaded records.")
                return None

            gdf = gpd.GeoDataFrame(geometries, crs="EPSG:4326")
            gdf['description'] = gdf['ID_TRAM'].map(descriptions_map)
            gdf = gdf[['ID_TRAM', 'description', 'geometry']]

            logger.info(f"Successfully created GeoDataFrame with {len(gdf)} segment geometries.")
            (self.base_dir / "processed").mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_path, driver="GPKG", layer="trams_geometries", overwrite=True) # Added overwrite=True
            logger.info(f"Successfully saved traffic segment geometries to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during traffic segment geometry processing: {str(e)}")
            return None

    def get_parking_data(self) -> Dict[str, Optional[Path]]:
        """Collect parking-related data, focusing on the secured underground parking JSON."""
        logger.info("Starting parking data collection for underground parkings (BSM).")
        downloaded_files = {}
        
        # Updated source information for BSM underground parkings
        bsm_underground_parking_source = {
            'Aparcaments_securitzat.json': {
                "url": "https://opendata-ajuntament.barcelona.cat/data/dataset/68b29854-7c61-4126-9004-83ed792d675c/resource/1f8bbdb6-e9d8-43d8-80df-338bf933d1d2/download",
                "is_json": True # Mark as JSON, no CSV columns needed here
            }
        }

        filename_to_download = 'Aparcaments_securitzat.json'
        details = bsm_underground_parking_source.get(filename_to_download)

        if not details:
            logger.error(f"Configuration for {filename_to_download} not found in parking_sources.")
            return downloaded_files

        headers = None
        bcn_od_token = self.config.get('bcn_od_token')

        if details.get("is_json"): # This is the secured BSM parking data
            if bcn_od_token:
                logger.info(f"Using BCN_OD_TOKEN for {filename_to_download}")
                headers = {'Authorization': bcn_od_token}
            else:
                logger.warning(f"BCN_OD_TOKEN not found in config. Download of {filename_to_download} might fail if token is required.")
        
        downloaded_path = self.download_file(
            details["url"], 
            filename_to_download, 
            'parking', 
            headers=headers
        )
        
        if downloaded_path:
            logger.info(f"Successfully downloaded {filename_to_download} to {downloaded_path}")
            # For JSON, validation is different. For now, we assume download success implies validity.
            # Further parsing/validation can be added here or in integrate_data if needed.
            # Example: Check if it's valid JSON
            try:
                with open(downloaded_path, 'r') as f:
                    json.load(f)
                logger.info(f"Successfully validated {downloaded_path} as JSON.")
                downloaded_files[downloaded_path.name] = downloaded_path
            except json.JSONDecodeError:
                logger.error(f"Validation failed: {downloaded_path} is not valid JSON.")
            except Exception as e:
                logger.error(f"Error during JSON validation for {downloaded_path}: {e}")
        else:
            logger.warning(f"Download failed for {filename_to_download}.")
            
        return downloaded_files
    
    def get_weather_data(self, include_realtime: bool = False) -> Dict[str, Optional[Path]]:
        """Collect weather data from WeatherAPI.com."""
        logger.info("Starting weather data collection")
        downloaded_files = {}
        
        # Historical weather data
        hist_weather_url = "https://opendata-ajuntament.barcelona.cat/data/dataset/00904de2-8660-4c41-92e3-66e7c87265be/resource/c6c07ed8-d890-4b94-bcac-8fbd920e69ea/download"
        filepath = self.download_file(hist_weather_url, "historical_weather.csv", 'weather')
        if filepath:
            downloaded_files['historical_weather.csv'] = filepath
        
        # Real-time weather data from WeatherAPI.com
        if self.config['weather_api_key'] and include_realtime:
            try:
                # Barcelona coordinates
                lat, lon = 41.3851, 2.1734
                
                # Get current weather
                current_url = f"http://api.weatherapi.com/v1/current.json?key={self.config['weather_api_key']}&q={lat},{lon}"
                response = requests.get(current_url, timeout=30)
                response.raise_for_status()
                
                current_weather = response.json()
                
                # Get forecast
                forecast_url = f"http://api.weatherapi.com/v1/forecast.json?key={self.config['weather_api_key']}&q={lat},{lon}&days=3"
                response = requests.get(forecast_url, timeout=30)
                response.raise_for_status()
                
                forecast = response.json()
                
                # Combine current and forecast data
                weather_data = {
                    'current': current_weather['current'],
                    'forecast': forecast['forecast']
                }
                
                filepath = self.base_dir / 'weather' / 'realtime_weather.json'
                with open(filepath, 'w') as f:
                    json.dump(weather_data, f, indent=2)
                
                downloaded_files['realtime_weather.json'] = filepath
                logger.info("Successfully downloaded weather data from WeatherAPI.com")
                
            except Exception as e:
                logger.error(f"Error fetching weather data from WeatherAPI.com: {str(e)}")
        
        return downloaded_files
    
    def get_transport_data(self) -> Dict[str, Path]:
        """Collect transport data using appropriate auth methods per endpoint."""
        logger.info("Starting transport data collection")
        downloaded_files = {}
        
        app_id = self.config.get('tmb_app_id')
        app_key = self.config.get('tmb_app_key')

        if not app_id or not app_key:
            logger.error("TMB app_id or app_key missing in configuration (pass.json). Skipping TMB downloads.")
            return downloaded_files
            
        # --- GTFS Download (Static API - uses Query Parameters) --- 
        auth_params = f"?app_id={app_id}&app_key={app_key}"
        gtfs_base_url = "https://api.tmb.cat/v1/static/datasets/gtfs.zip"
        gtfs_url = gtfs_base_url + auth_params
        
        logger.info(f"Downloading GTFS from {gtfs_base_url} using query parameters.")
        filepath_gtfs = self.download_file(gtfs_url, "tmb_gtfs.zip", 'transport', headers=None)
        if filepath_gtfs:
             if filepath_gtfs.is_dir(): 
                 downloaded_files['tmb_gtfs_dir'] = filepath_gtfs 
             else: 
                 downloaded_files[filepath_gtfs.name] = filepath_gtfs

        # --- TMB iBus API (Real-time - uses Headers for Auth) ---
        # Example: Get info for a specific bus stop line
        # This URL might need adjustment based on specific needs (e.g., specific stop, line)
        ibus_base_url = "https://api.tmb.cat/v1/ibus/stops/2775" # Example stop
        ibus_url = ibus_base_url # No params in base_url for this one

        if app_id and app_key: # Ensure keys are present
            logger.info(f"Querying TMB iBus API at {ibus_base_url} using headers.")
            headers = {
                'Accept': 'application/json',
                'app_id': app_id,
                'app_key': app_key,
                'User-Agent': 'OnSpotMLDataCollector/1.0'
            }
            try:
                # TODO: Resolve SSL certificate verification issue properly for production.
                # Using verify=False for now to bypass local issuer certificate error.
                logger.warning("SSL verification is currently disabled for TMB iBus API. This is insecure and should be fixed.")
                response = requests.get(ibus_url, headers=headers, timeout=30, verify=False)
                response.raise_for_status()
                ibus_data = response.json()
                # Save the JSON response to a file
                ibus_path = self.base_dir / 'transport' / 'tmb_ibus_stop_2775.json'
                with open(ibus_path, 'w', encoding='utf-8') as f:
                    json.dump(ibus_data, f, ensure_ascii=False, indent=4)
                downloaded_files['tmb_ibus_stop_2775'] = ibus_path
                logger.info(f"Successfully downloaded and saved iBus data for stop 2775 to {ibus_path}")
            except requests.exceptions.SSLError as e:
                logger.error(f"SSL Error connecting to TMB iBus API {ibus_url}: {e}. Try updating certifi package or system certificates.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error connecting to TMB iBus API {ibus_url}: {e}")
        else:
            logger.warning("TMB App ID or App Key not found in config. Skipping TMB iBus API call.")

        return downloaded_files
    
    def get_events_data(self) -> Dict[str, Path]:
        """Collect event data, handling JSON response and validating actual fields."""
        logger.info("Starting events data collection")
        downloaded_files = {}
        
        primary_url = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=5a0d8ece-07f9-4c2c-ac4c-450b70b877f7&limit=1000" # Increased limit
        target_filename = "cultural_events.csv"
        fallback_url = "https://barcelonadadescultura.bcn.cat/api/1/activitats/dades"
        
        filepath = None
        try:
            # --- Primary Download Attempt --- 
            logger.info(f"Attempting to download events from: {primary_url}")
            response = requests.get(primary_url, timeout=45) # Increased timeout slightly
            response.raise_for_status()
            data = response.json()

            if not data.get('success'):
                 raise Exception(f"Events API request was not successful: {data.get('error', 'Unknown error')}")
            
            records = data.get('result', {}).get('records')
            if records is None: # Check specifically for None, empty list is valid
                 raise Exception("Events API response missing 'result' or 'records' field.")
            
            if not records:
                 logger.warning("No event records found in the primary API response.")
                 # Decide if we should proceed to fallback even if primary succeeds but is empty
                 # For now, save the empty file and let validation pass (if applicable)

            actual_columns = [field['id'] for field in data.get('result', {}).get('fields', [])] 
            if not actual_columns:
                 logger.warning("Could not determine actual columns from events API response fields.")
                 # Use a default basic check if fields are missing
                 actual_columns = list(records[0].keys()) if records else []
            
            logger.info(f"Found {len(records)} event records. Actual columns: {actual_columns}")
            
            # Save the extracted records to CSV
            df = pd.DataFrame(records)
            filepath = self.base_dir / 'events' / target_filename
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully saved {len(records)} events to {filepath}")

            # Validate based on actual columns found in the JSON response structure
            if not self._validate_csv(filepath, expected_columns=actual_columns):
                logger.warning(f"Primary events file {filepath} failed validation despite successful save.")
                # Reset filepath to trigger fallback if validation fails
                filepath = None 
            else:
                 downloaded_files[filepath.name] = filepath # Add to dict if valid

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading primary events data: {str(e)}")
            filepath = None # Ensure filepath is None to trigger fallback
        except Exception as e:
            logger.error(f"Error processing primary events data: {str(e)}")
            filepath = None # Ensure filepath is None to trigger fallback

        # --- Fallback Download Attempt (if primary failed or validation failed) --- 
        if filepath is None:
             logger.info(f"Attempting fallback events download from: {fallback_url}")
             # Use the download_file utility for fallback (handles retries)
             # Note: Fallback validation might need different columns
             fallback_filepath = self.download_file(fallback_url, target_filename, 'events', expected_columns=None) # No validation for now
             if fallback_filepath:
                 # Decide how to handle fallback data - maybe parse differently if it's also JSON?
                 # For now, just log and add to dict if downloaded.
                 logger.info(f"Fallback events file downloaded to {fallback_filepath}")
                 # We might need to re-parse/validate based on fallback format
                 downloaded_files[fallback_filepath.name] = fallback_filepath
             else:
                 logger.error(f"Failed to download events data from fallback source as well.")
            
        return downloaded_files
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning steps."""
        logger.info(f"Cleaning data... Initial shape: {df.shape}")
        # Example: Handle missing values (replace with appropriate strategy)
        # df.fillna(value=np.nan, inplace=True) 
        # Example: Convert data types
        # df['some_column'] = pd.to_numeric(df['some_column'], errors='coerce')
        # Example: Remove duplicates
        df.drop_duplicates(inplace=True)
        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df

    def integrate_data(self) -> Optional[Path]:
        """Integrate and clean collected data into a single dataset."""
        logger.info("Starting data integration and cleaning")
        
        try:
            # Define paths using the base directory
            parking_path = self.base_dir / 'parking' / 'trams_aparcament.csv'
            tariffs_path = self.base_dir / 'parking' / '2024_4T_TARIFES.csv'
            output_path = self.base_dir / 'integrated_parking_data.csv'

            # Check if necessary files exist
            if not parking_path.exists() or not tariffs_path.exists():
                logger.error("Cannot integrate data: Missing required parking or tariff files.")
                required_files_exist = False
                if not parking_path.exists(): logger.error(f"Missing: {parking_path}")
                if not tariffs_path.exists(): logger.error(f"Missing: {tariffs_path}")
                return None
            
            # Load parking data
            parking = pd.read_csv(parking_path)
            tariffs = pd.read_csv(tariffs_path)
            
            # Clean data before merging
            parking_cleaned = self._clean_data(parking.copy())
            tariffs_cleaned = self._clean_data(tariffs.copy())
            
            # Check column names AFTER loading
            logger.info(f"Parking columns: {parking_cleaned.columns.tolist()}")
            logger.info(f"Tariffs columns: {tariffs_cleaned.columns.tolist()}")
            
            # Ensure merge keys exist
            if 'ID_TARIFA' not in parking_cleaned.columns or 'ID_TARIFA' not in tariffs_cleaned.columns:
                 logger.error("Integration failed: Missing 'ID_TARIFA' merge key in one or both dataframes.")
                 return None

            # Merge parking segments with tariffs
            merged_data = pd.merge(
                parking_cleaned,
                tariffs_cleaned,
                on='ID_TARIFA', # Use common column name directly
                how='left'
            )
            
            # Clean the merged data
            merged_data_cleaned = self._clean_data(merged_data)
            
            # Save integrated dataset
            merged_data_cleaned.to_csv(output_path, index=False)
            logger.info(f"Integrated and cleaned dataset saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error during data integration: {str(e)}")
            return None # Return None on error

    def collect_all_data(self, include_realtime_weather: bool = False) -> Dict[str, Any]:
        """Collects all defined datasets for Barcelona."""
        logger.info("Starting all data collection processes...")
        collected_data_paths: Dict[str, Any] = {}

        collected_data_paths["parking"] = self.get_parking_data()
        collected_data_paths["weather"] = self.get_weather_data(include_realtime=include_realtime_weather)
        collected_data_paths["transport"] = self.get_transport_data()
        collected_data_paths["events"] = self.get_events_data()
        # Add the new geometry collection method here if desired to run with "all"
        # For now, it's a standalone callable method.
        # Example: collected_data_paths["traffic_segment_geometries"] = self.get_traffic_segment_geometries()

        logger.info("All data collection processes finished.")
        # Further processing to return paths of successfully downloaded files/main outputs
        summary_paths = {}
        for key, paths_dict in collected_data_paths.items():
            if isinstance(paths_dict, dict):
                summary_paths[key] = {name: str(p.resolve()) if p else None for name, p in paths_dict.items()}
            elif isinstance(paths_dict, Path):
                 summary_paths[key] = str(paths_dict.resolve())
            else:
                summary_paths[key] = paths_dict
        
        return summary_paths

def main():
    """Main execution function."""
    try:
        # Validate credentials before proceeding
        validate_credentials()
        
        collector = BarcelonaDataCollector()
        
        # Collect data from all sources
        logger.info("Starting Barcelona data collection...")
        
        logger.info("\n=== Downloading parking data ===")
        parking_files = collector.get_parking_data()
        
        logger.info("\n=== Downloading weather data ===")
        weather_files = collector.get_weather_data()
        
        logger.info("\n=== Downloading transport data ===")
        transport_files = collector.get_transport_data()
        
        logger.info("\n=== Downloading event data ===")
        event_files = collector.get_events_data()
        
        # Integrate data
        logger.info("\n=== Integrating data ===")
        integrated_data_path = collector.integrate_data()
        
        if integrated_data_path:
            logger.info("\nData collection and integration complete!")
            # Print summary
            print("\nSummary of downloaded files:")
            for category, files in [
                ("Parking", parking_files),
                ("Weather", weather_files),
                ("Transport", transport_files),
                ("Events", event_files)
            ]:
                print(f"\n{category}:")
                for filename, path in files.items():
                    print(f"  - {filename}: {path}")
            
            print(f"\nIntegrated dataset: {integrated_data_path}")
        else:
            logger.error("Data integration failed. See logs for details.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 