import pandas as pd
import numpy as np
import os
import logging
import time
import holidays
# import geopandas as gpd # Not needed for base features
# from pyrosm import OSM # Not needed for base features
# from shapely.geometry import Point, LineString, box # Not needed for base features
# from shapely.ops import unary_union # Not needed for base features
import warnings
import pandera as pa
from pandera.errors import SchemaError
import json

# --- Configuration ---
INPUT_DATA_PATH = "data/interim/parking_history_consolidated.parquet"
OUTPUT_BASE_FEATURES_PATH = "data/interim/df_with_base_features.parquet"

# External data paths
WEATHER_DATA_PATH = "data/weather/historical_weather.csv"
EVENT_DATA_PATH = "data/events/events.csv"
# PARKING_SEGMENTS_PATH = "data/processed/trams_geometries.gpkg" # Moved to Phase 2 script
# OSM_PBF_PATH = "data/raw/cataluna-latest.osm.pbf" # Moved to Phase 2 script

TARGET_COLUMN = 'actual_state'
TIMESTAMP_COLUMN = 'timestamp'
ID_COLUMN = 'ID_TRAMO'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger("1_prepare_base_features")


# --- Feature Engineering Functions (Copied and adapted from build_features_historical.py) ---

def load_data(file_path, date_filter_start=None, date_filter_end=None):
    """Loads the consolidated parking history data, optionally filtering by date."""
    logger.info(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns.")

        if TIMESTAMP_COLUMN not in df.columns:
            logger.error(f"Timestamp column '{TIMESTAMP_COLUMN}' not found.")
            return None
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

        if date_filter_start and date_filter_end:
            logger.info(f"Filtering data from {date_filter_start} to {date_filter_end} (inclusive)...")
            start_date = pd.to_datetime(date_filter_start)
            end_date = pd.to_datetime(date_filter_end).replace(hour=23, minute=59, second=59) # Ensure end_date is inclusive
            df = df[(df[TIMESTAMP_COLUMN] >= start_date) & (df[TIMESTAMP_COLUMN] <= end_date)]
            logger.info(f"Data filtered. New shape: {df.shape[0]} rows, {df.shape[1]} columns.")
            if df.empty:
                logger.warning("DataFrame is empty after date filtering.")
                return None
        elif date_filter_start or date_filter_end:
            logger.warning("Both date_filter_start and date_filter_end must be provided for filtering. Skipping date filter.")

        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def generate_temporal_features(df, timestamp_col=TIMESTAMP_COLUMN):
    """
    Generates temporal features from timestamp column.
    """
    logger.info("Generating temporal features...")
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found for temporal features.")
        return df

    # Basic temporal features
    df['hour'] = df[timestamp_col].dt.hour.astype('int64')
    df['dayofweek'] = df[timestamp_col].dt.dayofweek.astype('int64')
    df['dayofyear'] = df[timestamp_col].dt.dayofyear.astype('int64')
    df['month'] = df[timestamp_col].dt.month.astype('int64')
    df['year'] = df[timestamp_col].dt.year.astype('int64')
    df['weekofyear'] = df[timestamp_col].dt.isocalendar().week.astype('int64')
    df['quarter'] = df[timestamp_col].dt.quarter.astype('int64')
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype('int64')

    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    logger.info("Successfully generated temporal features.")
    return df

def generate_lag_features(df, id_col=ID_COLUMN, timestamp_col=TIMESTAMP_COLUMN, target_col=TARGET_COLUMN, lag_hours=None, freq_minutes=5):
    """Generates lag features for the target variable, grouped by ID."""
    if lag_hours is None:
        lag_hours = [1, 6, 12, 24, 48, 168]
    logger.info(f"Generating lag features for target '{target_col}' with lags (hours): {lag_hours}...")
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found for lag generation.")
        return df
    if freq_minutes <= 0:
        logger.error("freq_minutes must be positive.")
        return df

    # Data should be sorted by id_col and timestamp_col before calling this function.
    # logger.info(f"Ensuring data is sorted by {id_col} and {timestamp_col} for lag generation...")
    # df = df.sort_values(by=[id_col, timestamp_col]) # Assumed to be done in main()

    periods_per_hour = 60 // freq_minutes
    if 60 % freq_minutes != 0:
         logger.warning(f"Data frequency {freq_minutes} mins does not divide an hour evenly. Lags might be approx.")

    lag_col_names = []
    for lag_h in lag_hours:
        periods_to_shift = lag_h * periods_per_hour
        lag_col_name = f'{target_col}_lag_{lag_h}h'
        lag_col_names.append(lag_col_name)
        logger.info(f"Creating lag: {lag_col_name} ({periods_to_shift} periods)")
        df[lag_col_name] = df.groupby(id_col)[target_col].shift(periods_to_shift)
        nan_count = df[lag_col_name].isnull().sum()
        logger.info(f"{lag_col_name} created with {nan_count} NaN values (due to shift).")
    
    logger.info("Successfully generated lag features.")
    return df, lag_col_names

def generate_public_holiday_features(df, timestamp_col=TIMESTAMP_COLUMN, country='ES', subdiv='CT'):
    """Generates public holiday features using the 'holidays' library."""
    logger.info(f"Generating public holiday features for {country} subdivision {subdiv}...")
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found.")
        df['is_public_holiday'] = 0 # Fallback
        return df
    try:
        years = df[timestamp_col].dt.year.unique()
        logger.info(f"Determined years for public holidays: {list(years)}")
        country_holidays_obj = holidays.CountryHoliday(country, subdiv=subdiv, years=list(years))
        logger.info(f"Public holiday object created successfully for years: {list(years)}.")
        actual_public_holiday_dates = set(country_holidays_obj.keys()) # .keys() returns datetime.date objects
        
        # df['is_public_holiday'] = df[timestamp_col].dt.date.apply(lambda d: d in actual_public_holiday_dates).astype(int)
        # df['is_public_holiday'] = df[timestamp_col].dt.normalize().isin(actual_public_holiday_dates).astype(int)
        df['is_public_holiday'] = df[timestamp_col].dt.date.isin(actual_public_holiday_dates).astype(int)


        num_holidays_found = df['is_public_holiday'].sum()
        logger.info(f"Successfully generated 'is_public_holiday' feature. Found {num_holidays_found} public holiday records.")
    except ImportError:
        logger.error("ImportError: 'holidays' library not installed. pip install holidays. Skipping public holiday features.")
        df['is_public_holiday'] = 0
    except Exception as e:
        logger.error(f"Error generating public holiday features: {e}")
        df['is_public_holiday'] = 0
    return df

def generate_school_holiday_features(df, timestamp_col=TIMESTAMP_COLUMN):
    """
    Generates school holiday features for Catalonia.
    Loads holiday dates from a configuration file.
    """
    logger.info("Generating school holiday features (Catalonia)...")
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found for school holidays.")
        df['is_school_holiday'] = 0
        return df

    # Load school holiday dates from config file
    school_holidays_path = "data/config/school_holidays.json"
    try:
        if os.path.exists(school_holidays_path):
            with open(school_holidays_path, 'r') as f:
                school_holidays_data = json.load(f)
                school_holiday_periods = school_holidays_data['holiday_periods']
            logger.info(f"Successfully loaded school holiday periods from {school_holidays_path}")
        else:
            logger.warning(f"School holidays config file not found at {school_holidays_path}. Using default dates.")
            # Default dates as fallback
            school_holiday_periods = [
                # 2021-2022
                {"start": "2021-12-23", "end": "2022-01-07", "name": "Christmas 21-22"},
                {"start": "2022-04-11", "end": "2022-04-18", "name": "Easter 22"},
                {"start": "2022-06-23", "end": "2022-09-05", "name": "Summer 22"},
                # 2022-2023
                {"start": "2022-12-22", "end": "2023-01-08", "name": "Christmas 22-23"},
                {"start": "2023-04-03", "end": "2023-04-10", "name": "Easter 23"},
                {"start": "2023-06-22", "end": "2023-09-06", "name": "Summer 23"},
                # 2023-2024
                {"start": "2023-12-21", "end": "2024-01-07", "name": "Christmas 23-24"},
                {"start": "2024-03-25", "end": "2024-04-01", "name": "Easter 24"},
                {"start": "2024-06-21", "end": "2024-09-09", "name": "Summer 24"},
                # 2024-2025
                {"start": "2024-12-23", "end": "2025-01-07", "name": "Christmas 24-25"},
                {"start": "2025-04-14", "end": "2025-04-21", "name": "Easter 25"},
                {"start": "2025-06-20", "end": "2025-09-08", "name": "Summer 25"},
            ]
    except Exception as e:
        logger.error(f"Error loading school holiday dates: {e}")
        df['is_school_holiday'] = 0
        return df

    df['is_school_holiday'] = 0
    df_dates = df[timestamp_col].dt.date

    for period in school_holiday_periods:
        start_date = pd.to_datetime(period['start']).date()
        end_date = pd.to_datetime(period['end']).date()
        logger.info(f"Processing school holiday period: {period['name']} ({start_date} to {end_date})")
        df.loc[(df_dates >= start_date) & (df_dates <= end_date), 'is_school_holiday'] = 1

    logger.info(f"Successfully generated 'is_school_holiday' feature. Found {df['is_school_holiday'].sum()} school holiday records.")
    return df

def load_and_merge_external_data_chunked(df_chunk, weather_path=WEATHER_DATA_PATH, event_path=EVENT_DATA_PATH, timestamp_col=TIMESTAMP_COLUMN):
    """Loads and merges external data sources (weather, events) for a chunk of data."""
    logger.info("Loading and merging external data for chunk...")
    df_original_cols = df_chunk.columns.tolist()

    # --- Weather data ---
    if os.path.exists(weather_path):
        try:
            logger.info(f"Loading weather data from {weather_path}...")
            df_weather = pd.read_csv(weather_path)
            weather_timestamp_col = 'DATA_LECTURA'
            if weather_timestamp_col not in df_weather.columns:
                 raise ValueError(f"Weather data must contain '{weather_timestamp_col}'")
            df_weather[weather_timestamp_col] = pd.to_datetime(df_weather[weather_timestamp_col])
            df_weather = df_weather.sort_values(by=weather_timestamp_col)
            
            # Ensure chunk is sorted by timestamp_col
            df_chunk = df_chunk.sort_values(by=timestamp_col)
            
            weather_features = [col for col in df_weather.columns if col != weather_timestamp_col]
            
            # First try exact timestamp match
            df_chunk = pd.merge_asof(df_chunk, df_weather[[weather_timestamp_col] + weather_features],
                                   left_on=timestamp_col, right_on=weather_timestamp_col,
                                   direction='nearest', tolerance=pd.Timedelta('1hour'))
            
            # Log weather data coverage
            weather_coverage = df_chunk[weather_features].notna().mean() * 100
            logger.info(f"Weather data coverage after merge: {weather_coverage.to_dict()}")
            
            # Drop the original weather timestamp column if it's now redundant
            if weather_timestamp_col in df_chunk.columns and weather_timestamp_col != timestamp_col:
                df_chunk = df_chunk.drop(columns=[weather_timestamp_col])
                
            logger.info("Weather data merged for chunk.")
            
        except Exception as e:
            logger.error(f"Error with weather data from {weather_path}: {e}")
    else:
        logger.warning(f"Weather data file not found: {weather_path}. Skipping.")

    # --- Event data ---
    if os.path.exists(event_path):
        try:
            logger.info(f"Loading event data from {event_path}...")
            df_events = pd.read_csv(event_path)
            event_date_col = 'DataInici'
            if event_date_col not in df_events.columns:
                 raise ValueError(f"Event data must contain '{event_date_col}'")
            df_events[event_date_col] = pd.to_datetime(df_events[event_date_col]).dt.date
            event_dates_set = set(df_events[event_date_col].unique())
            
            df_chunk['is_event_day'] = df_chunk[timestamp_col].dt.date.apply(lambda d: 1 if d in event_dates_set else 0)
            event_day_count = df_chunk['is_event_day'].sum()
            logger.info(f"Event data merged for chunk. Found {event_day_count} event day records.")
        except Exception as e:
            logger.error(f"Error with event data from {event_path}: {e}")
    else:
        logger.warning(f"Event data file not found: {event_path}. Skipping.")

    new_cols = [col for col in df_chunk.columns if col not in df_original_cols]
    logger.info(f"External data processing (weather, events) added columns: {new_cols}")
    return df_chunk

def validate_output_schema(df):
    """Validate the schema of the output DataFrame using pandera."""
    output_schema = pa.DataFrameSchema(
        columns={
            "ID_TRAMO": pa.Column(pa.Int, required=True),
            "timestamp": pa.Column(pa.DateTime, required=True),
            "actual_state": pa.Column(pa.Int, required=True),
            "hour": pa.Column(pa.Int, checks=pa.Check.in_range(0, 23), required=True),
            "dayofweek": pa.Column(pa.Int, checks=pa.Check.in_range(0, 6), required=True),
            "dayofyear": pa.Column(pa.Int, checks=pa.Check.in_range(1, 366), required=True),
            "month": pa.Column(pa.Int, checks=pa.Check.in_range(1, 12), required=True),
            "year": pa.Column(pa.Int, required=True),
            "weekofyear": pa.Column(pa.Int, checks=pa.Check.in_range(1, 53), required=True),
            "quarter": pa.Column(pa.Int, checks=pa.Check.in_range(1, 4), required=True),
            "is_weekend": pa.Column(pa.Int, checks=pa.Check.isin([0, 1]), required=True),
            "hour_sin": pa.Column(float, required=True),
            "hour_cos": pa.Column(float, required=True),
            "dayofweek_sin": pa.Column(float, required=True),
            "dayofweek_cos": pa.Column(float, required=True),
            "month_sin": pa.Column(float, required=True),
            "month_cos": pa.Column(float, required=True),
            "actual_state_lag_1h": pa.Column(float, required=True),
            "actual_state_lag_6h": pa.Column(float, required=True),
            "actual_state_lag_12h": pa.Column(float, required=True),
            "actual_state_lag_24h": pa.Column(float, required=True),
            "actual_state_lag_48h": pa.Column(float, required=True),
            "actual_state_lag_168h": pa.Column(float, required=True),
            "is_public_holiday": pa.Column(pa.Int, checks=pa.Check.isin([0, 1]), required=True),
            "is_school_holiday": pa.Column(pa.Int, checks=pa.Check.isin([0, 1]), required=True),
            # Weather columns - allowing nulls
            "VALOR": pa.Column(float, nullable=True),
            "DATA_EXTREM": pa.Column(pa.String, nullable=True),
            "CODI_ESTACIO": pa.Column(pa.String, nullable=True),
            "ACRÒNIM": pa.Column(pa.String, nullable=True),
            # Event column
            "is_event_day": pa.Column(pa.Int, checks=pa.Check.isin([0, 1]), required=True),
        },
        strict=False
    )
    try:
        logger.info("Validating DataFrame schema with pandera...")
        validated_df = output_schema.validate(df, lazy=True)
        logger.info("Schema validation successful!")
        return validated_df
    except SchemaError as err:
        logger.error("Schema validation failed!")
        logger.error(f"Failure cases: {err.failure_cases.head()}")
        raise err
    except Exception as e:
        logger.error(f"Unexpected error during schema validation: {str(e)}")
        raise

def process_temporal_chunk(df_chunk, start_date, end_date, chunk_num, total_chunks):
    """Process a temporal chunk of data with holiday features and external data."""
    chunk_start_time = time.time()
    logger.info(f"Processing chunk {chunk_num}/{total_chunks}: {start_date} to {end_date}")
    logger.info(f"Chunk size: {df_chunk.shape[0]} rows, {df_chunk.shape[1]} columns")
    
    try:
        # Generate holiday features for the chunk
        logger.info(f"Generating holiday features for chunk {chunk_num}...")
        df_chunk = generate_public_holiday_features(df_chunk)
        df_chunk = generate_school_holiday_features(df_chunk)
        
        # Merge external data for the chunk
        logger.info(f"Merging external data for chunk {chunk_num}...")
        df_chunk = load_and_merge_external_data_chunked(df_chunk)
        
        # Validate chunk schema before returning
        logger.info(f"Validating schema for chunk {chunk_num}...")
        df_chunk = validate_output_schema(df_chunk)
        
        chunk_end_time = time.time()
        chunk_duration = chunk_end_time - chunk_start_time
        logger.info(f"Successfully processed chunk {chunk_num}/{total_chunks} in {chunk_duration:.2f} seconds")
        logger.info(f"Chunk {chunk_num} final shape: {df_chunk.shape}")
        
        return df_chunk
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_num}/{total_chunks}: {str(e)}")
        logger.error(f"Chunk details: {start_date} to {end_date}, shape: {df_chunk.shape}")
        raise

# --- Main Orchestration ---
def main():
    logger.info("Starting Phase 1: Base Feature Engineering Pipeline...")
    start_time = time.time()

    # 1. Load Full Historical Data (for 2022-2023)
    df = load_data(INPUT_DATA_PATH, date_filter_start="2022-01-01", date_filter_end="2023-12-31")
    if df is None or df.empty:
        logger.error("Halting pipeline: Data loading failed or resulted in empty DataFrame.")
        return

    # Ensure ID_COLUMN is numeric and sort
    if ID_COLUMN not in df.columns:
        logger.error(f"Critical: ID_COLUMN '{ID_COLUMN}' not found. Halting.")
        return
    try:
        df[ID_COLUMN] = pd.to_numeric(df[ID_COLUMN])
        logger.info(f"Converted '{ID_COLUMN}' to numeric. Dtype: {df[ID_COLUMN].dtype}")
    except Exception as e:
        logger.error(f"Could not convert '{ID_COLUMN}' to numeric: {e}. Halting.")
        return
        
    logger.info(f"Sorting data by {ID_COLUMN} and {TIMESTAMP_COLUMN}...")
    df = df.sort_values(by=[ID_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)
    logger.info("Data sorted.")

    # 2. Generate temporal features on full DataFrame
    logger.info("Generating temporal features on full DataFrame...")
    df = generate_temporal_features(df)
    logger.info(f"Temporal features generated. DataFrame shape: {df.shape}")

    # 3. Generate lag features on full DataFrame
    logger.info("Generating lag features on full DataFrame...")
    df, lag_feature_names = generate_lag_features(df)
    logger.info(f"Lag features generated. DataFrame shape: {df.shape}")

    # 4. Process data in temporal chunks
    logger.info("Starting temporal chunking for holiday and external data processing...")
    
    # Define chunk periods (e.g., by quarter)
    chunk_periods = pd.date_range(start="2022-01-01", end="2023-12-31", freq='QE')
    total_chunks = len(chunk_periods)
    processed_chunks = []
    failed_chunks = []
    
    # Validate data coverage
    data_start = df[TIMESTAMP_COLUMN].min()
    data_end = df[TIMESTAMP_COLUMN].max()
    logger.info(f"Data coverage: {data_start} to {data_end}")
    logger.info(f"Data will be processed in {total_chunks} quarterly chunks")
    
    # Create quarter labels for logging
    quarter_labels = []
    for i in range(total_chunks):
        start_date = chunk_periods[i] - pd.offsets.QuarterEnd(1) + pd.Timedelta(days=1)
        end_date = chunk_periods[i]
        quarter = (end_date.month - 1) // 3 + 1
        year = end_date.year
        quarter_labels.append(f"Q{quarter} {year}")
    
    logger.info(f"Quarter periods: {quarter_labels}")
    
    # Verify we have data for all chunks
    for i in range(total_chunks):
        start_date = chunk_periods[i] - pd.offsets.QuarterEnd(1) + pd.Timedelta(days=1)
        end_date = chunk_periods[i]
        chunk_mask = (df[TIMESTAMP_COLUMN] >= start_date) & (df[TIMESTAMP_COLUMN] <= end_date)
        chunk_size = df[chunk_mask].shape[0]
        logger.info(f"Chunk {i+1} ({quarter_labels[i]}): {chunk_size:,} rows")
    
    # Drop NaNs from lag features BEFORE chunking
    logger.info("Dropping NaNs from lag features before chunking...")
    initial_shape = df.shape
    df = df.dropna(subset=lag_feature_names)
    logger.info(f"Dropped {initial_shape[0] - df.shape[0]:,} rows due to NaNs in lag features. New shape: {df.shape}")
    
    # Process chunks
    for i in range(total_chunks):
        start_date = chunk_periods[i] - pd.offsets.QuarterEnd(1) + pd.Timedelta(days=1)
        end_date = chunk_periods[i]
        chunk_num = i + 1
        
        try:
            # Filter data for this chunk - use inclusive end date for last chunk
            chunk_mask = (df[TIMESTAMP_COLUMN] >= start_date) & (df[TIMESTAMP_COLUMN] <= end_date)
            df_chunk = df[chunk_mask].copy()
            
            if df_chunk.empty:
                logger.warning(f"Chunk {chunk_num}/{total_chunks} ({quarter_labels[i]}) is empty. Skipping.")
                continue
                
            # Process the chunk
            processed_chunk = process_temporal_chunk(df_chunk, start_date, end_date, chunk_num, total_chunks)
            processed_chunks.append(processed_chunk)
            
            # Free up memory
            del df_chunk
            del processed_chunk
            
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_num}/{total_chunks} ({quarter_labels[i]}): {str(e)}")
            failed_chunks.append({
                'chunk_num': chunk_num,
                'quarter': quarter_labels[i],
                'start_date': start_date,
                'end_date': end_date,
                'error': str(e)
            })
            continue
    
    # Report on chunk processing results
    logger.info(f"Chunk processing completed. Successfully processed {len(processed_chunks)}/{total_chunks} chunks")
    if failed_chunks:
        logger.warning(f"Failed to process {len(failed_chunks)} chunks:")
        for failed in failed_chunks:
            logger.warning(f"Chunk {failed['chunk_num']} ({failed['quarter']}): {failed['start_date']} to {failed['end_date']} - {failed['error']}")
    
    if not processed_chunks:
        logger.error("No chunks were successfully processed. Halting pipeline.")
        return
    
    # Combine processed chunks
    logger.info("Combining processed chunks...")
    df = pd.concat(processed_chunks, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {df.shape}")
    
    # Apply weather feature filling across chunk boundaries
    logger.info("Applying weather feature filling across chunk boundaries...")
    weather_features = ['DATA_EXTREM', 'CODI_ESTACIO', 'ACRÒNIM', 'VALOR']
    df[weather_features] = df[weather_features].ffill().bfill()
    
    # Log weather data coverage after filling
    weather_coverage = df[weather_features].notna().mean() * 100
    logger.info(f"Weather data coverage after filling: {weather_coverage.to_dict()}")
    
    # No need to drop NaNs from lag features again since we did it before chunking
    logger.info("Performing final checks...")
    logger.info("Final column list:")
    for col in df.columns:
        logger.info(f" - {col} (dtype: {df[col].dtype}, NaNs: {df[col].isnull().sum()})")

    # 7. Save Output
    try:
        output_dir = os.path.dirname(OUTPUT_BASE_FEATURES_PATH)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving base feature table to {OUTPUT_BASE_FEATURES_PATH}...")
        df.to_parquet(OUTPUT_BASE_FEATURES_PATH, index=False)
        logger.info(f"Successfully saved base features: {df.shape[0]} rows, {df.shape[1]} columns.")
    except Exception as e:
        logger.error(f"Error saving base feature table to {OUTPUT_BASE_FEATURES_PATH}: {e}")

    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(f"Phase 1: Base Feature Engineering Pipeline completed in {total_duration:.2f} seconds")
    logger.info(f"Average chunk processing time: {total_duration/total_chunks:.2f} seconds per chunk")

if __name__ == "__main__":
    main() 