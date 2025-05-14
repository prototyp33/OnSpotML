import pandas as pd
import numpy as np
import os
import logging
import time
import holidays
import geopandas as gpd
from pyrosm import OSM
from shapely.geometry import Point, LineString, box
from shapely.ops import unary_union
import warnings

# --- Configuration ---
INPUT_DATA_PATH = "data/processed/parking_history_consolidated.parquet"
OUTPUT_FEATURE_PATH_TEST = "data/processed/features_master_table_historical_TEST.parquet" # Temp output for testing
OUTPUT_FEATURE_PATH_FULL = "data/processed/features_master_table_historical.parquet"

# External data paths
WEATHER_DATA_PATH = "data/weather/historical_weather.csv" # Corrected path
# HOLIDAY_DATA_PATH = "data/external/holidays_2022_2023.csv" # Removed - using holidays library
EVENT_DATA_PATH = "data/events/events.csv" # Corrected path
PARKING_SEGMENTS_PATH = "data/processed/trams_geometries.gpkg" # MODIFIED: New path to the GeoPackage
OSM_PBF_PATH = "data/raw/cataluna-latest.osm.pbf" # Added path to OSM PBF file

# POI feature generation parameters (can be refined)
# IMPORTANT: Tune POI_CHUNK_SIZE for full dataset runs.
# Smaller chunks: More overhead per chunk (BBOX calculation, merging small results).
# Larger chunks: Higher memory usage per chunk.
# Monitor memory and total processing time to find an optimal balance.
POI_CHUNK_SIZE = 50000 # Adjust based on memory and dataset size
POI_RADII = [100, 200, 500] # Example radii in meters

TARGET_COLUMN = 'actual_state'
TIMESTAMP_COLUMN = 'timestamp'
ID_COLUMN = 'ID_TRAMO'


# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s') # Set to DEBUG to see column logs
logger = logging.getLogger("build_features_historical")


# --- Feature Engineering Functions ---

def load_data(file_path):
    """Loads the consolidated parking history data."""
    logger.info(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def generate_temporal_features(df, timestamp_col=TIMESTAMP_COLUMN):
    """Generates temporal features from the timestamp column."""
    logger.info("Generating temporal features...")
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found in DataFrame.")
        return df # Or raise error

    try:
        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek # Monday=0, Sunday=6
        df['dayofyear'] = df[timestamp_col].dt.dayofyear
        df['month'] = df[timestamp_col].dt.month
        df['year'] = df[timestamp_col].dt.year
        df['weekofyear'] = df[timestamp_col].dt.isocalendar().week.astype(int)
        df['quarter'] = df[timestamp_col].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        # Cyclical features (sine/cosine transformations)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        logger.info("Successfully generated temporal features.")
        return df
    except Exception as e:
        logger.error(f"Error generating temporal features: {e}")
        return df # Or raise error

def generate_lag_features(df, id_col=ID_COLUMN, timestamp_col=TIMESTAMP_COLUMN, target_col=TARGET_COLUMN, lag_hours=None, fill_value=None, freq_minutes=5):
    """
    Generates lag features for the target variable, grouped by ID.
    Assumes data is sorted by ID and timestamp.
    Calculates shift periods based on lag_hours and data frequency.
    """
    if lag_hours is None:
        lag_hours = [1, 6, 12, 24, 48, 168] # Default lags in hours

    logger.info(f"Generating lag features for target '{target_col}' with lags (hours): {lag_hours}...")
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found for lag generation.")
        return df

    if freq_minutes <= 0:
        logger.error("freq_minutes must be positive for lag calculation.")
        return df

    # Sort data first: crucial for correct lag calculation within groups
    # Checking if already sorted can save time, but sorting ensures correctness
    logger.info("Sorting data by ID and timestamp for lag generation...")
    df = df.sort_values(by=[id_col, timestamp_col])

    periods_per_hour = 60 // freq_minutes
    if 60 % freq_minutes != 0:
         logger.warning(f"Data frequency {freq_minutes} mins does not divide an hour evenly. Lag calculations might be approximate.")

    for lag_h in lag_hours:
        periods_to_shift = lag_h * periods_per_hour
        lag_col_name = f'{target_col}_lag_{lag_h}h'
        logger.info(f"Creating lag: {lag_col_name} ({periods_to_shift} periods)")

        # The shift operation within groupby will correctly handle lags per ID_TRAMO
        df[lag_col_name] = df.groupby(id_col)[target_col].shift(periods_to_shift)

        if fill_value is not None:
            # Consider filling with a specific value or maybe forward/backward fill
            df[lag_col_name] = df[lag_col_name].fillna(fill_value)
            # logger.info(f"Filled NaN values in {lag_col_name} with {fill_value}")
        else:
            # Optional: Log how many NaNs were created by the shift
            nan_count = df[lag_col_name].isnull().sum()
            logger.info(f"{lag_col_name} created with {nan_count} NaN values (due to shift).")

    logger.info("Successfully generated lag features.")
    return df

def load_external_data(df, weather_path=WEATHER_DATA_PATH, event_path=EVENT_DATA_PATH, timestamp_col=TIMESTAMP_COLUMN):
    """Loads and merges external data sources (weather, holidays, events)."""
    logger.info("Loading and merging external data...")
    df_original_cols = df.columns.tolist()

    # --- Weather data ---
    if os.path.exists(weather_path):
        try:
            logger.info(f"Loading weather data from {weather_path}...")
            df_weather = pd.read_csv(weather_path)
            weather_timestamp_col = 'DATA_LECTURA' # Corrected column name
            if weather_timestamp_col not in df_weather.columns:
                 raise ValueError(f"Weather data must contain a timestamp column named '{weather_timestamp_col}'")

            logger.info(f"Loaded weather data: {df_weather.shape[0]} rows.")
            df_weather[weather_timestamp_col] = pd.to_datetime(df_weather[weather_timestamp_col])
            df_weather = df_weather.sort_values(by=weather_timestamp_col)

            # Ensure main df is sorted by timestamp for merge_asof
            df = df.sort_values(by=timestamp_col)

            logger.info("Merging weather data using merge_asof (nearest, 1h tolerance)...")
            # Identify weather feature columns (excluding timestamp)
            weather_features = [col for col in df_weather.columns if col != weather_timestamp_col]
            df = pd.merge_asof(df,
                               df_weather[[weather_timestamp_col] + weather_features],
                               left_on=timestamp_col,
                               right_on=weather_timestamp_col,
                               direction='nearest',
                               tolerance=pd.Timedelta('1hour'))

            # Handle NaNs introduced by merge_asof (e.g., edges of time range)
            logger.info(f"Forward/backward filling NaNs in weather features: {weather_features}")
            df[weather_features] = df[weather_features].ffill().bfill()
            logger.info("Weather data merged successfully.")

        except FileNotFoundError:
             logger.warning(f"Weather data file not found at {weather_path}. Skipping.")
        except Exception as e:
            logger.error(f"Error loading or processing weather data from {weather_path}: {e}")
    else:
        logger.warning(f"Weather data file not found at {weather_path}. Skipping.")

    # --- Holiday data (using holidays library) ---
    logger.info("Attempting to generate holiday features using 'holidays' library...")
    try:
        if timestamp_col not in df.columns:
            logger.error(f"Timestamp column '{timestamp_col}' needed for holiday generation is missing.")
            raise ValueError(f"Timestamp column '{timestamp_col}' missing.") # Raise error to be caught below

        years = df[timestamp_col].dt.year.unique()
        logger.info(f"Determined years for holidays: {years}")

        # Use subdiv='CT' for Catalonia - ensure library supports this or adjust
        # For holidays library, 'prov' is not a valid parameter for CountryHoliday.
        # For Spain, subdivisions (communities) are specified using 'subdiv'.
        logger.info(f"Initializing holidays.CountryHoliday('ES', subdiv='CT', years={years})...")
        es_holidays_obj = holidays.CountryHoliday('ES', subdiv='CT', years=list(years))
        logger.info(f"Holiday object created successfully.")

        # Get a set of actual holiday dates from the object for efficient lookup
        actual_holiday_dates = set(es_holidays_obj.keys())
        logger.debug(f"Actual holiday dates extracted for {years}: {sorted(list(actual_holiday_dates))[:20]}") # Log some

        # Create date column for checking
        # logger.info("Creating temporary date column for holiday check...")
        # df['date_only'] = df[timestamp_col].dt.date

        logger.info("Applying holiday check using the set of actual holiday dates with .isin()...")
        # df['is_holiday'] = df['date_only'].apply(lambda d: d in actual_holiday_dates).astype(int)
        df['is_holiday'] = df[timestamp_col].dt.date.isin(actual_holiday_dates).astype(int)
        
        num_holidays_found = df['is_holiday'].sum()
        logger.info(f"Successfully generated 'is_holiday' feature. Found {num_holidays_found} holiday records in the current dataset slice.")
        # df = df.drop(columns=['date_only'])
        # logger.info("Dropped temporary date column.")

    except ImportError:
        logger.error("ImportError: The 'holidays' library is not installed or accessible. Cannot generate holiday features. Please run: pip install holidays")
        # Optionally add a dummy column so downstream code doesn't break if it expects 'is_holiday'
        # df['is_holiday'] = 0
    except Exception as e:
        logger.error(f"An unexpected error occurred during holiday feature generation: {e}", exc_info=True)
        # Optionally add a dummy column
        # df['is_holiday'] = 0

    # --- Event data (using efficient apply) ---
    if os.path.exists(event_path):
        try:
            logger.info(f"Loading event data from {event_path}...")
            df_events = pd.read_csv(event_path)
            event_date_col = 'DataInici' # Corrected column name
            if event_date_col not in df_events.columns:
                 raise ValueError(f"Event data must contain a date column named '{event_date_col}'")

            logger.info(f"Loaded event data: {df_events.shape[0]} records.")
            df_events[event_date_col] = pd.to_datetime(df_events[event_date_col])

            # Create a set of unique event dates for efficient lookup
            event_dates_set = set(df_events[event_date_col].dt.date.unique())
            logger.info(f"Found {len(event_dates_set)} unique event dates.")

            # Create 'is_event_day' feature using efficient lookup
            logger.info("Generating 'is_event_day' feature...")
            df['date_only_temp'] = df[timestamp_col].dt.date # Temporary column
            df['is_event_day'] = df['date_only_temp'].apply(lambda date: 1 if date in event_dates_set else 0)
            df = df.drop(columns=['date_only_temp'])

            event_day_count = df['is_event_day'].sum()
            logger.info(f"Event data merged successfully ('is_event_day' feature created). Found {event_day_count} event day records.")

        except FileNotFoundError:
             logger.warning(f"Event data file not found at {event_path}. Skipping.")
        except Exception as e:
            logger.error(f"Error loading or processing event data from {event_path}: {e}")
    else:
        logger.warning(f"Event data file not found at {event_path}. Skipping.")

    # Log newly added columns
    new_cols = [col for col in df.columns if col not in df_original_cols]
    logger.info(f"External data processing added columns: {new_cols}")
    logger.info("External data loading/merging step completed.")
    return df

def generate_poi_features(df_history, parking_segments_path=PARKING_SEGMENTS_PATH,
                           id_col_history=ID_COLUMN, id_col_segments='ID_TRAM',
                           osm_pbf_path=OSM_PBF_PATH,
                           chunk_size=POI_CHUNK_SIZE, radii=None, target_crs="EPSG:25831"):
    """
    Generates POI features by joining historical data with parking segments and POIs.
    Handles large datasets through chunking.
    """
    logger.info("Starting POI feature generation process...")

    # Check for rtree library (spatial indexing for sjoin)
    try:
        import rtree
        logger.info(f"rtree library found. Spatial indexing for sjoin will be enabled. Version: {rtree.__version__}")
    except ImportError:
        logger.warning("rtree library not found. pip install rtree for significantly faster spatial joins. sjoin will still work but will be slower.")

    if radii is None:
        radii = POI_RADII

    # --- 1. Load and Prepare Parking Segments GeoDataFrame ---
    logger.info(f"Loading parking segments from: {parking_segments_path}")
    if not os.path.exists(parking_segments_path):
        logger.error(f"Parking segments file not found: {parking_segments_path}. Cannot generate POI features.")
        # Add empty POI columns to avoid downstream errors if main df needs them
        empty_poi_cols = {}
        if radii is None:
            radii = POI_RADII # Use default if not provided
        osm_tags_of_interest = {
            'amenity': ['restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'bank', 'atm', 'pharmacy', 'hospital', 'clinic', 'doctors', 'dentist', 'kindergarten', 'school', 'college', 'university', 'library', 'theatre', 'cinema', 'nightclub', 'community_centre', 'parking', 'fuel', 'bus_station', 'taxi', 'car_rental', 'police', 'post_office', 'townhall'],
            'shop': ['supermarket', 'convenience', 'department_store', 'mall', 'bakery', 'butcher', 'clothes', 'electronics', 'hardware', 'books', 'kiosk'],
            'leisure': ['park', 'playground', 'sports_centre', 'stadium', 'fitness_centre', 'park', 'pitch', 'track', 'swimming_pool'],
            'tourism': ['hotel', 'motel', 'guest_house', 'attraction', 'museum', 'viewpoint', 'information'],
            'historic': ['castle', 'monument', 'ruins', 'memorial'],
            'office': True, # Generic office tag
            'public_transport': ['station', 'platform', 'stop_position']
        }
        for tag_key, values in osm_tags_of_interest.items():
            if isinstance(values, list):
                for value in values:
                    for r in radii:
                        empty_poi_cols[f'poi_count_{tag_key}_{value}_{r}m'] = 0
            elif values is True: # For generic tags like 'office'
                 for r in radii:
                    empty_poi_cols[f'poi_count_{tag_key}_{r}m'] = 0
        for r in radii: # Overall POI density
            empty_poi_cols[f'poi_density_{r}m'] = 0
        
        df_history = df_history.assign(**empty_poi_cols)
        logger.info(f"Added {len(empty_poi_cols)} empty POI columns as fallback.")
        return df_history
    
    try:
        gdf_segments_raw = gpd.read_file(parking_segments_path) # MODIFIED: Now loads a GeoPackage
        logger.info(f"Successfully loaded parking segments: {gdf_segments_raw.shape[0]} rows, {gdf_segments_raw.shape[1]} columns.")

        # Ensure ID_TRAM is the ID column from the GeoPackage (as set by BarcelonaDataCollector)
        id_col_segments = 'ID_TRAM' # Standardized ID column name from the GeoPackage
        
        # Ensure id_col_segments is numeric
        if id_col_segments in gdf_segments_raw.columns:
            try:
                gdf_segments_raw[id_col_segments] = pd.to_numeric(gdf_segments_raw[id_col_segments])
                logger.info(f"Successfully converted '{id_col_segments}' in parking segments to numeric. Dtype: {gdf_segments_raw[id_col_segments].dtype}")
            except Exception as e:
                logger.error(f"Could not convert '{id_col_segments}' in parking segments to numeric: {e}. This might cause merge issues and errors in set_index.")
                # Fallback: Add empty POI columns as this is critical for merging
                # (Duplicating fallback logic for clarity here, though it's also below)
                empty_poi_cols = {}
                if radii is None: radii = POI_RADII 
                osm_tags_of_interest = {
                    'amenity': ['restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'bank', 'atm', 'pharmacy', 'hospital', 'clinic', 'doctors', 'dentist', 'kindergarten', 'school', 'college', 'university', 'library', 'theatre', 'cinema', 'nightclub', 'community_centre', 'parking', 'fuel', 'bus_station', 'taxi', 'car_rental', 'police', 'post_office', 'townhall'],
                    'shop': ['supermarket', 'convenience', 'department_store', 'mall', 'bakery', 'butcher', 'clothes', 'electronics', 'hardware', 'books', 'kiosk'],
                    'leisure': ['park', 'playground', 'sports_centre', 'stadium', 'fitness_centre', 'park', 'pitch', 'track', 'swimming_pool'],
                    'tourism': ['hotel', 'motel', 'guest_house', 'attraction', 'museum', 'viewpoint', 'information'],
                    'historic': ['castle', 'monument', 'ruins', 'memorial'],
                    'office': True,
                    'public_transport': ['station', 'platform', 'stop_position']
                }
                for tag_key, values in osm_tags_of_interest.items():
                    if isinstance(values, list):
                        for value in values:
                            for r_val in radii: empty_poi_cols[f'poi_count_{tag_key}_{value}_{r_val}m'] = 0
                    elif values is True:
                        for r_val in radii: empty_poi_cols[f'poi_count_{tag_key}_{r_val}m'] = 0
                for r_val in radii: empty_poi_cols[f'poi_density_{r_val}m'] = 0
                df_history = df_history.assign(**empty_poi_cols)
                logger.info(f"Added {len(empty_poi_cols)} empty POI columns as fallback due to ID conversion error in segments.")
                return df_history
        else:
            logger.error(f"Critical: '{id_col_segments}' column not found in {parking_segments_path} before numeric conversion. Check file content.")
            # Fallback logic (as above)
            empty_poi_cols = {}
            if radii is None: radii = POI_RADII
            osm_tags_of_interest = { # Copied for brevity, ensure this structure is maintained
                'amenity': ['restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'bank', 'atm', 'pharmacy', 'hospital', 'clinic', 'doctors', 'dentist', 'kindergarten', 'school', 'college', 'university', 'library', 'theatre', 'cinema', 'nightclub', 'community_centre', 'parking', 'fuel', 'bus_station', 'taxi', 'car_rental', 'police', 'post_office', 'townhall'],
                'shop': ['supermarket', 'convenience', 'department_store', 'mall', 'bakery', 'butcher', 'clothes', 'electronics', 'hardware', 'books', 'kiosk'],
                'leisure': ['park', 'playground', 'sports_centre', 'stadium', 'fitness_centre', 'park', 'pitch', 'track', 'swimming_pool'],
                'tourism': ['hotel', 'motel', 'guest_house', 'attraction', 'museum', 'viewpoint', 'information'],
                'historic': ['castle', 'monument', 'ruins', 'memorial'],
                'office': True,
                'public_transport': ['station', 'platform', 'stop_position']
            }
            for tag_key, values in osm_tags_of_interest.items():
                if isinstance(values, list):
                    for value in values:
                        for r_val in radii: empty_poi_cols[f'poi_count_{tag_key}_{value}_{r_val}m'] = 0
                elif values is True:
                    for r_val in radii: empty_poi_cols[f'poi_count_{tag_key}_{r_val}m'] = 0
            for r_val in radii: empty_poi_cols[f'poi_density_{r_val}m'] = 0
            df_history = df_history.assign(**empty_poi_cols)
            logger.info(f"Added {len(empty_poi_cols)} empty POI columns as fallback due to missing ID column in segments.")
            return df_history

        # The GeoPackage geometry is already LineString and should have CRS (EPSG:4326)
        # We just need to ensure it's set if gpd.read_file didn't pick it up, or to be explicit.
        if gdf_segments_raw.crs is None:
            logger.warning(f"CRS for {parking_segments_path} is None. Assuming EPSG:4326.")
            gdf_segments_raw = gdf_segments_raw.set_crs("EPSG:4326", allow_override=True)
        elif gdf_segments_raw.crs.to_epsg() != 4326:
            logger.warning(f"CRS for {parking_segments_path} is {gdf_segments_raw.crs.to_string()}. Will re-project to EPSG:4326 before projecting to target_crs.")
            gdf_segments_raw = gdf_segments_raw.to_crs("EPSG:4326")
        
        logger.info(f"CRS for segments confirmed/set to EPSG:4326. Contains {len(gdf_segments_raw)} segments.")

        # Deduplicate segments based on ID_TRAM, keeping the first occurrence
        initial_count = len(gdf_segments_raw)
        gdf_segments_raw = gdf_segments_raw.drop_duplicates(subset=[id_col_segments], keep='first')
        deduplicated_count = len(gdf_segments_raw)
        if initial_count > deduplicated_count:
            logger.info(f"Deduplicated parking segments by '{id_col_segments}': {initial_count} -> {deduplicated_count} unique segments.")

        # Project to the target CRS (e.g., EPSG:25831 for Barcelona area for accurate meter-based buffering)
        logger.info(f"Projecting segment geometries to target CRS: {target_crs}...")
        # Ensure we only take necessary columns for this projection step to avoid issues with other dtypes
        gdf_segments = gdf_segments_raw[[id_col_segments, 'geometry']].copy() 
        gdf_segments = gdf_segments.to_crs(target_crs)
        logger.info(f"Projection to {target_crs} successful.")

        # Create buffered geometries for segments
        logger.info(f"Pre-calculating buffers for radii: {radii}m...")
        for r in radii:
            buffer_col_name = f'buffer_{r}m'
            gdf_segments[buffer_col_name] = gdf_segments.geometry.buffer(r)
            logger.info(f"Created buffer column: {buffer_col_name}")

        # Keep only ID and buffer columns for merging later
        buffer_cols = [f'buffer_{r}m' for r in radii]
        gdf_segments_buffered = gdf_segments[[id_col_segments] + buffer_cols].copy()
        logger.info(f"Prepared buffered segments GeoDataFrame with columns: {gdf_segments_buffered.columns.tolist()}")

        # --- FIX for Row Inflation: Set index and drop duplicates BEFORE merge ---
        gdf_segments_buffered = gdf_segments_buffered.set_index(id_col_segments)
        if gdf_segments_buffered.index.has_duplicates:
            num_duplicates = gdf_segments_buffered.index.duplicated().sum()
            logger.warning(f"Duplicate {id_col_segments} found in parking segments index ({num_duplicates} duplicates). Keeping first occurrence for each {id_col_segments} before merge.")
            gdf_segments_buffered = gdf_segments_buffered[~gdf_segments_buffered.index.duplicated(keep='first')]
        logger.info(f"Using {len(gdf_segments_buffered)} unique segments for merge.")
        # --- END FIX ---

        # --- 3. Merge Segments (Buffers) with History (before chunking) ---
        # This step adds the buffer geometry columns needed for the spatial joins later.
        df_history_with_geoms = df_history.copy() # Keep original df_history intact
        # Preserve original index - important for joining POI features back later
        df_history_with_geoms['original_index'] = df_history_with_geoms.index

        # Now perform the merge
        logger.info(f"Merging historical data ({len(df_history_with_geoms)} rows) with buffered segments ({len(gdf_segments_buffered)} unique rows) on '{id_col_history}' == index ('{gdf_segments_buffered.index.name}')...")
        df_history_with_geoms = pd.merge(
            df_history_with_geoms,
            gdf_segments_buffered, # Has unique index now
            left_on=id_col_history,
            right_index=True,
            how='left',
            validate='many_to_one' # Validation should pass now
        )
        merge_nan_count = df_history_with_geoms[buffer_cols[0]].isnull().sum() # Check NaNs in first buffer col
        if merge_nan_count > 0:
            logger.warning(f"Merge resulted in {merge_nan_count} rows with no matching segment geometry (NaNs in buffer columns). Check if all {id_col_history} values exist in parking segments {id_col_segments}.")
        logger.info(f"Shape after merging with segments: {df_history_with_geoms.shape}")

    except Exception as e:
        logger.error(f"Failed to load or prepare parking segments: {e}", exc_info=True)
        return df_history # Return original df if segments fail

    # --- 4. Initialize OSM Reader and Preprocess POIs ---
    try:
        logger.info(f"Initializing OSM data reader for: {osm_pbf_path}")
        osm = OSM(osm_pbf_path)
        logger.info("OSM reader initialized.")
    except Exception as e:
        logger.error(f"Error initializing OSM reader for {osm_pbf_path}: {e}", exc_info=True)
        # If OSM fails to load, we can't generate POI features. Add empty columns and return.
        empty_poi_cols = {}
        # ... (Logic to create all possible POI column names with 0/False) ... 
        # This part needs to be robust to create all expected POI columns if OSM fails.
        # For simplicity in this edit, I'll assume the previous fallback logic for missing segments handles this if we return early.
        # A more robust solution would define all potential POI columns upfront.
        logger.warning("Returning DataFrame without POI features due to OSM loading failure.")
        return df_history # Or df_history with pre-defined empty POI columns

    logger.info("Defining POI categories for preprocessing...")
    poi_categories = { 
        'sustenance': {'amenity': ['restaurant', 'cafe', 'fast_food', 'pub', 'bar']},
        'shop': {'shop': True}, 
        'education': {'amenity': ['school', 'university', 'college', 'kindergarten']},
        'health': {'amenity': ['hospital', 'clinic', 'doctors', 'pharmacy']},
        'transport': {'public_transport': ['stop_position', 'platform', 'station'], 'highway': ['bus_stop']},
        'leisure': {'leisure': True},
        'tourism': {'tourism': ['hotel', 'hostel', 'guest_house', 'motel', 'attraction']},
        'parking': {'amenity': ['parking', 'parking_entrance', 'parking_space']},
        'finance': {'amenity': ['bank', 'atm']},
    }
    logger.info(f"POI categories defined: {list(poi_categories.keys())}")

    processed_pois_by_category = {}
    logger.info("--- Starting Upfront POI Preprocessing (Extraction, Projection, Cleaning) ---")
    for category, filter_dict in poi_categories.items():
        logger.info(f"Preprocessing POIs for category: {category} with filter: {filter_dict}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning) # Suppress pyrosm geometry warnings
                category_gdf_all = osm.get_pois(custom_filter=filter_dict)

            if category_gdf_all is None or category_gdf_all.empty:
                logger.info(f"No POIs found for category '{category}' in the entire PBF. Storing None.")
                processed_pois_by_category[category] = None
                continue
            
            logger.info(f"Extracted {len(category_gdf_all)} raw POIs for '{category}'.")

            # Project to target_crs
            if category_gdf_all.crs is None:
                logger.warning(f"Raw POIs for '{category}' have no CRS. Assuming EPSG:4326.")
                category_gdf_all = category_gdf_all.set_crs("EPSG:4326", allow_override=True)
            
            if category_gdf_all.crs.to_epsg() != target_crs.split(':')[1]:
                logger.info(f"Projecting raw POIs for {category} from {category_gdf_all.crs} to {target_crs}.")
                category_gdf_all = category_gdf_all.to_crs(target_crs)
            
            # Geometry Cleaning (adapted from chunk loop)
            logger.info(f"Cleaning geometries for '{category}' POIs...")
            non_point_mask = category_gdf_all.geometry.geom_type != 'Point'
            if 'geometry' in category_gdf_all.columns and non_point_mask.any():
                geoms_to_fix = category_gdf_all.loc[non_point_mask, 'geometry']
                if category_gdf_all.geometry.name != 'geometry':
                    category_gdf_all = category_gdf_all.set_geometry('geometry')
                fixed_geoms = geoms_to_fix.buffer(0)
                category_gdf_all.loc[non_point_mask, 'geometry'] = fixed_geoms
            
            initial_rows = len(category_gdf_all)
            category_gdf_all = category_gdf_all[~category_gdf_all.geometry.is_empty & category_gdf_all.geometry.is_valid]
            rows_dropped = initial_rows - len(category_gdf_all)
            if rows_dropped > 0:
                logger.warning(f"Dropped {rows_dropped} POIs for '{category}' due to empty/invalid geometry after cleaning.")

            if category_gdf_all.empty:
                logger.info(f"No valid POIs remaining for category '{category}' after cleaning. Storing None.")
                processed_pois_by_category[category] = None
            else:
                processed_pois_by_category[category] = category_gdf_all
                logger.info(f"Successfully preprocessed {len(category_gdf_all)} POIs for category '{category}'.")

        except ImportError as ie:
            logger.error(f"ImportError preprocessing category {category}: {ie}. Wildcard filters might require additional libraries.", exc_info=True)
            processed_pois_by_category[category] = None
        except Exception as e:
            logger.error(f"Error preprocessing category {category}: {e}", exc_info=True)
            processed_pois_by_category[category] = None
    logger.info("--- Finished Upfront POI Preprocessing ---")

    # --- 5. Chunked Processing ---
    all_chunk_results = []
    total_rows_processed = 0
    # Iterate over df_history_with_geoms, which contains buffer columns
    n_chunks = int(np.ceil(len(df_history_with_geoms) / chunk_size))
    logger.info(f"Starting chunked processing for {len(df_history_with_geoms)} rows in {n_chunks} chunks (size={chunk_size})...")

    for i, chunk_start in enumerate(range(0, len(df_history_with_geoms), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(df_history_with_geoms))
        # Slice df_history_with_geoms, use copy() to avoid SettingWithCopyWarning
        chunk_with_geoms = df_history_with_geoms.iloc[chunk_start:chunk_end].copy()
        logger.info(f"--- Processing Chunk {i+1}/{n_chunks} (rows {chunk_start}-{chunk_end}) ---")
        # Keep track of the original index within this chunk
        chunk_with_geoms = chunk_with_geoms.set_index('original_index', drop=False)

        chunk_start_time = time.time()

        # --- 5a. Process POIs for the Chunk ---
        logger.info("Starting POI processing for chunk...")

        # Initialize POI columns in the chunk
        poi_feature_columns = [] # Keep track of generated columns
        for category in poi_categories:
            for r in radii:
                count_col = f'poi_{category}_count_{r}m'
                present_col = f'poi_{category}_present_{r}m'
                chunk_with_geoms[count_col] = 0
                chunk_with_geoms[present_col] = False # Use Boolean
                poi_feature_columns.extend([count_col, present_col])

        # Drop rows where merge failed (no segment match -> buffer cols are NaN)
        buffer_cols_check = [f'buffer_{r}m' for r in radii]
        chunk_valid_geoms = chunk_with_geoms.dropna(subset=buffer_cols_check).copy()

        if chunk_valid_geoms.empty:
            logger.warning(f"Chunk {i+1} has no valid geometries after merge check. Skipping POI calculation, keeping initial zeros.")
            # Store only the initialized POI columns with the original index
            all_chunk_results.append(chunk_with_geoms[['original_index'] + poi_feature_columns].set_index('original_index'))
            continue # Skip to the next chunk

        # Determine the total bounding box for the valid buffers in this chunk
        all_chunk_buffers = []
        for r in radii:
             all_chunk_buffers.extend(chunk_valid_geoms[f'buffer_{r}m'].tolist())
        valid_geoms = [geom for geom in all_chunk_buffers if geom is not None and not geom.is_empty]

        if not valid_geoms:
             logger.warning(f"Chunk {i+1} has no valid buffer geometries after filtering Nones/Empties. Skipping POI calculation.")
             all_chunk_results.append(chunk_with_geoms[['original_index'] + poi_feature_columns].set_index('original_index'))
             continue

        # --- FIX: Calculate chunk_bbox here, before the category loop ---
        total_bounds_geom = unary_union(valid_geoms)
        chunk_bbox = total_bounds_geom.bounds # (minx, miny, maxx, maxy)
        logger.info(f"Chunk BBOX for POI query: {chunk_bbox}")
        # --- END FIX ---

        # --- POI Category Loop ---
        for category, filter_dict in poi_categories.items(): # filter_dict is not used here anymore
            logger.info(f"Processing category: {category} for chunk {i+1}")
            
            # Retrieve preprocessed POIs for this category
            gdf_pois_all_preprocessed = processed_pois_by_category.get(category)

            if gdf_pois_all_preprocessed is None or gdf_pois_all_preprocessed.empty:
                logger.info(f"No preprocessed POIs available for category '{category}'. Skipping for this chunk.")
                # Counts will remain 0, present flags False due to initialization earlier
                continue
            
            # Create a polygon from the chunk_bbox for clipping
            minx, miny, maxx, maxy = chunk_bbox
            chunk_bbox_poly = box(minx, miny, maxx, maxy)

            # Clip the preprocessed (already projected and cleaned) POIs to the chunk_bbox
            logger.info(f"Clipping {len(gdf_pois_all_preprocessed)} preprocessed POIs for '{category}' to chunk BBOX {chunk_bbox}")
            # Ensure crs matches if there was any doubt, though it should be target_crs from preprocessing
            if gdf_pois_all_preprocessed.crs is None or gdf_pois_all_preprocessed.crs.to_string().upper() != target_crs.upper():
                 logger.warning(f"CRS mismatch or missing for preprocessed '{category}' POIs ({gdf_pois_all_preprocessed.crs}). Re-projecting to {target_crs}")
                 gdf_pois_all_preprocessed = gdf_pois_all_preprocessed.to_crs(target_crs)
            
            clip_gdf = gpd.GeoDataFrame([{'geometry': chunk_bbox_poly}], crs=target_crs) # Ensure clip_gdf has CRS
            gdf_pois_chunk = gpd.clip(gdf_pois_all_preprocessed, clip_gdf)

            if gdf_pois_chunk is None or gdf_pois_chunk.empty:
                logger.info(f"No POIs remaining for category '{category}' after clipping to chunk bbox.")
                continue

            logger.info(f"Found {len(gdf_pois_chunk)} POIs (any type) for '{category}' after clipping to chunk.")
            geom_counts = gdf_pois_chunk.geom_type.value_counts()
            logger.debug(f"Geometry types for '{category}' in chunk: {geom_counts.to_dict()}")

            # Perform spatial join for each radius
            for r in radii:
                buffer_col = f'buffer_{r}m'
                count_col = f'poi_{category}_count_{r}m'
                present_col = f'poi_{category}_present_{r}m'

                temp_chunk_geoms_df = chunk_valid_geoms[[buffer_col]].copy()
                temp_chunk_geoms_df.rename(columns={buffer_col: 'geometry'}, inplace=True)
                chunk_buffers_gdf = gpd.GeoDataFrame(
                    temp_chunk_geoms_df,
                    geometry='geometry',
                    crs=target_crs
                )

                logger.info(f"Performing sjoin for '{category}' radius {r}m with {len(gdf_pois_chunk)} POIs and {len(chunk_buffers_gdf)} buffers...")
                try:
                    joined_pois = gpd.sjoin(chunk_buffers_gdf, gdf_pois_chunk, how='left', predicate='intersects')
                    counts = joined_pois.groupby(joined_pois.index)['index_right'].count()
                    chunk_with_geoms[count_col] = chunk_with_geoms[count_col].add(counts, fill_value=0).astype(int)
                    chunk_with_geoms[present_col] = chunk_with_geoms[present_col] | (counts > 0)

                    if not counts.empty:
                        logger.debug(f"Sample counts for {category} radius {r}m (Index: Count): {counts.head().to_dict()}")
                    else:
                        logger.debug(f"No intersections found for {category} radius {r}m.")
                except Exception as sjoin_err:
                    logger.error(f"Error during sjoin/aggregation for {category}, radius {r}m: {sjoin_err}", exc_info=True)
            
            # No need for the original try-except that caught ImportError for osm.get_pois here,
            # as that's handled in the upfront preprocessing.

        logger.info("POI processing for chunk completed.")

        # --- 5b. Store Chunk Results ---
        # Select only the original index and the POI feature columns we generated/updated
        chunk_result_to_store = chunk_with_geoms[['original_index'] + poi_feature_columns].set_index('original_index')
        all_chunk_results.append(chunk_result_to_store)

        total_rows_processed += len(chunk_with_geoms) # Use length of original chunk slice
        chunk_end_time = time.time()
        logger.info(f"--- Finished Chunk {i+1}/{n_chunks} in {chunk_end_time - chunk_start_time:.2f} seconds --- ({total_rows_processed}/{len(df_history_with_geoms)} rows processed)")

    # --- 6. Combine POI Results and Join back to Original DataFrame ---
    if not all_chunk_results:
        logger.warning("No results generated from chunk processing. Returning original dataframe without POI features.")
        # Optionally, add empty POI columns here if downstream code expects them
        # for category in poi_categories:
        #     for r in radii:
        #         df_history[f'poi_{category}_count_{r}m'] = 0
        #         df_history[f'poi_{category}_present_{r}m'] = False
        return df_history

    logger.info("Concatenating POI results from all chunks...")
    df_poi_features = pd.concat(all_chunk_results)
    logger.info(f"Concatenated POI features shape: {df_poi_features.shape}")

    # Add Log transform and fix types for POI features
    logger.info("Applying log transformation (np.log1p) and type casting to POI features...")
    log_transform_cols = []
    for category in poi_categories:
        for r in radii:
            count_col = f'poi_{category}_count_{r}m'
            present_col = f'poi_{category}_present_{r}m'
            log_col_name = count_col.replace('_count_', '_log1p_count_')

            if count_col in df_poi_features.columns:
                 df_poi_features[count_col] = pd.to_numeric(df_poi_features[count_col], errors='coerce').fillna(0).astype(int)
                 df_poi_features[log_col_name] = np.log1p(df_poi_features[count_col])
                 log_transform_cols.append(log_col_name)
            else:
                 logger.warning(f"Count column {count_col} not found in concatenated results.")

            if present_col in df_poi_features.columns:
                 df_poi_features[present_col] = df_poi_features[present_col].fillna(False).astype(bool) # Use bool type
            else:
                 logger.warning(f"Present column {present_col} not found in concatenated results.")

    logger.info(f"Log transformation applied, creating {len(log_transform_cols)} log count columns. Presence flags cast to bool.")

    # Join POI features back to the original df_history
    logger.info(f"Joining POI features ({len(df_poi_features)} rows) back to original data ({len(df_history)} rows)...")
    # Ensure df_history index is suitable for joining (should be unique integer index if reset in main)
    df_final = df_history.join(df_poi_features, how='left')

    # Verify row count after join
    if len(df_final) != len(df_history):
        logger.error(f"Row count mismatch after joining POI features! Expected {len(df_history)}, got {len(df_final)}. Check index alignment.")
        # Decide handling - maybe return df_history or raise error
        return df_history # Safest fallback
    else:
        logger.info("Join successful, row count matches original.")

    # Fill any NaNs in POI columns introduced by the join (shouldn't happen with left join if indices align, but good practice)
    poi_cols_final = [col for col in df_final.columns if col.startswith('poi_')]
    for col in poi_cols_final:
        if '_present_' in col:
             df_final[col] = df_final[col].fillna(False)
        elif '_count_' in col: # Includes log1p counts
             df_final[col] = df_final[col].fillna(0)

    logger.info("POI feature generation and merging process completed.")
    return df_final


# --- Main Orchestration ---

def main(test_run=True, test_rows=100000): # Added test_run flag and row count
    logger.info("Starting historical feature engineering pipeline...")
    start_time = time.time()

    # 1. Load data
    df = load_data(INPUT_DATA_PATH)
    if df is None:
        logger.error("Halting pipeline due to data loading failure.")
        return

    # Ensure ID_COLUMN is numeric for reliable merges
    if ID_COLUMN in df.columns:
        try:
            df[ID_COLUMN] = pd.to_numeric(df[ID_COLUMN])
            logger.info(f"Successfully converted '{ID_COLUMN}' in main data to numeric. Dtype: {df[ID_COLUMN].dtype}")
        except Exception as e:
            logger.error(f"Could not convert '{ID_COLUMN}' in main data to numeric: {e}. This might cause merge issues. Halting pipeline.")
            return
    else:
        logger.error(f"Critical: ID_COLUMN '{ID_COLUMN}' not found in loaded data. Halting pipeline.")
        return

    if test_run:
        logger.warning(f"--- TEST RUN --- Using only the first {test_rows} rows.")
        if len(df) > test_rows:
            # Ensure unique index for joining POI features later
            df = df.head(test_rows).reset_index(drop=True).copy()
        else:
             logger.warning(f"Dataset has fewer rows ({len(df)}) than requested test_rows ({test_rows}). Using all available rows.")
             df = df.reset_index(drop=True).copy() # Still reset index
        
        # Log the date range of the test data
        if TIMESTAMP_COLUMN in df.columns and not df.empty:
            min_ts = df[TIMESTAMP_COLUMN].min()
            max_ts = df[TIMESTAMP_COLUMN].max()
            logger.info(f"Test data timestamp range: {min_ts} to {max_ts}")
        else:
            logger.warning("Could not determine timestamp range for test data (column missing or empty dataframe).")

    # 2. Generate temporal features
    df = generate_temporal_features(df)

    # 3. Generate lag features
    # Define desired lags in hours, assuming 5-minute data frequency
    lag_hours_list = [1, 6, 12, 24, 48, 168] # 1h, 6h, 12h, 1d, 2d, 1w
    df = generate_lag_features(df, lag_hours=lag_hours_list, freq_minutes=5)

    # 4. Load and merge external data (weather, holidays, events)
    df = load_external_data(df) # This function currently contains placeholders

    # 5. Generate POI features
    # This is a complex step and might need to be run carefully, possibly in chunks.
    # The current generate_poi_features is a placeholder.
    df = generate_poi_features(df)

    # 6. Final processing (e.g., drop intermediate columns, reorder, type checks)
    logger.info("Final processing steps (placeholder)...")
    # Example: df = df.drop(columns=['unnecessary_intermediate_col'], errors='ignore')
    # Ensure ID and timestamp are first, target is last (common convention)
    # cols_to_move = [ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN]
    # df = df[[col for col in cols_to_move if col in df.columns] +
    #         [col for col in df.columns if col not in cols_to_move]]
    # Ensure index is reset before saving if it was manipulated
    df = df.reset_index(drop=True)


    # 7. Save enriched data
    output_path = OUTPUT_FEATURE_PATH_TEST if test_run else OUTPUT_FEATURE_PATH_FULL
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: # Ensure output_dir is not an empty string
             os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving enriched feature table to {output_path}...")
        df.to_parquet(output_path, index=False)
        logger.info(f"Successfully saved features: {df.shape[0]} rows, {df.shape[1]} columns.")
    except Exception as e:
        logger.error(f"Error saving feature table to {output_path}: {e}")

    end_time = time.time()
    logger.info(f"Historical feature engineering pipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Set test_run=False to run on the full dataset
    main(test_run=True) 