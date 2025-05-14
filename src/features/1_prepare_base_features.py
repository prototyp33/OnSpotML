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
    """Generates temporal features from the timestamp column."""
    logger.info("Generating temporal features...")
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found in DataFrame.")
        return df

    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col]) # Ensure datetime
        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofyear'] = df[timestamp_col].dt.dayofyear
        df['month'] = df[timestamp_col].dt.month
        df['year'] = df[timestamp_col].dt.year
        df['weekofyear'] = df[timestamp_col].dt.isocalendar().week.astype(int)
        df['quarter'] = df[timestamp_col].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
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
    Placeholder: Dates need to be confirmed and provided.
    """
    logger.info("Generating school holiday features (Placeholder - dates need to be confirmed)...")
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found for school holidays.")
        df['is_school_holiday'] = 0 # Fallback
        return df

    # --- Define School Holiday Periods (Needs to be populated with actual dates) ---
    # Example structure: list of tuples (start_date_str, end_date_str, holiday_name)
    school_holiday_periods_2022 = [
        # Christmas 2021-2022 (affecting early 2022)
        ("2021-12-23", "2022-01-07", "Christmas Break 21-22"), 
        # Easter 2022 (Setmana Santa)
        ("2022-04-11", "2022-04-18", "Easter Break 22"),
        # Segona Pasqua 2022
        ("2022-06-06", "2022-06-06", "Segona Pasqua 22"),
        # Summer 2022 (approx)
        ("2022-06-23", "2022-09-10", "Summer Break 22"), # End date approx.
        # La Mercè 2022 (observed)
        ("2022-09-26", "2022-09-26", "La Mercè 22"),
        # Christmas 2022-2023 (affecting late 2022)
        ("2022-12-22", "2023-01-08", "Christmas Break 22-23"),
    ]
    school_holiday_periods_2023 = [
        # Christmas 2022-2023 is covered by the end of school_holiday_periods_2022 list
        # Easter 2023 (Setmana Santa)
        ("2023-04-03", "2023-04-10", "Easter Break 23"),
        # Segona Pasqua 2023
        ("2023-06-05", "2023-06-05", "Segona Pasqua 23"),
        # Summer 2023 (approx)
        ("2023-06-21", "2023-09-10", "Summer Break 23"), # Start date approx (some levels), end date approx.
        # La Mercè 2023 (observed)
        ("2023-09-25", "2023-09-25", "La Mercè 23"),
        # Christmas 2023-2024 (affecting late 2023, up to Dec 31 for this dataset)
        ("2023-12-21", "2023-12-31", "Christmas Break 23-24 (Partial)"),
    ]
    
    all_school_holidays = school_holiday_periods_2022 + school_holiday_periods_2023
    
    df['is_school_holiday'] = 0
    df_dates = df[timestamp_col].dt.date # Work with dates for comparison

    for start_str, end_str, name in all_school_holidays:
        try:
            start_date = pd.to_datetime(start_str).date()
            end_date = pd.to_datetime(end_str).date()
            logger.info(f"Processing school holiday period: {name} ({start_date} to {end_date})")
            df.loc[(df_dates >= start_date) & (df_dates <= end_date), 'is_school_holiday'] = 1
        except Exception as e:
            logger.error(f"Error processing school holiday period {name}: {e}")

    num_school_holidays_found = df['is_school_holiday'].sum()
    logger.info(f"Successfully generated 'is_school_holiday' feature. Found {num_school_holidays_found} school holiday records.")
    logger.warning("School holiday dates are based on general calendars and should be verified. School-specific 'days of free disposal' are not included.")
    return df

def load_and_merge_external_data(df, weather_path=WEATHER_DATA_PATH, event_path=EVENT_DATA_PATH, timestamp_col=TIMESTAMP_COLUMN):
    """Loads and merges external data sources (weather, events). Public and school holidays are separate."""
    logger.info("Loading and merging external data (Weather, Events)...")
    df_original_cols = df.columns.tolist()

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
            
            # Main df is assumed to be sorted by timestamp_col
            logger.info(f"Ensuring main DataFrame is sorted by '{timestamp_col}' before weather merge_asof...")
            df = df.sort_values(by=timestamp_col)
            # --- END FIX ---
            weather_features = [col for col in df_weather.columns if col != weather_timestamp_col]
            df = pd.merge_asof(df, df_weather[[weather_timestamp_col] + weather_features],
                               left_on=timestamp_col, right_on=weather_timestamp_col,
                               direction='nearest', tolerance=pd.Timedelta('1hour'))
            df[weather_features] = df[weather_features].ffill().bfill()
            logger.info("Weather data merged.")
            # Drop the original weather timestamp column if it's now redundant
            if weather_timestamp_col in df.columns and weather_timestamp_col != timestamp_col: # Avoid dropping main ts col
                logger.info(f"Dropping redundant weather timestamp column: {weather_timestamp_col}")
                df = df.drop(columns=[weather_timestamp_col])
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
            
            df['is_event_day'] = df[timestamp_col].dt.date.apply(lambda d: 1 if d in event_dates_set else 0)
            event_day_count = df['is_event_day'].sum()
            logger.info(f"Event data merged. Found {event_day_count} event day records.")
        except Exception as e:
            logger.error(f"Error with event data from {event_path}: {e}")
    else:
        logger.warning(f"Event data file not found: {event_path}. Skipping.")

    new_cols = [col for col in df.columns if col not in df_original_cols]
    logger.info(f"External data processing (weather, events) added columns: {new_cols}")
    return df

# --- Main Orchestration ---
def main():
    logger.info("Starting Phase 1: Base Feature Engineering Pipeline...")
    start_time = time.time()

    # 1. Load Full Historical Data (for 2022-2023)
    # Ensure parking_history_consolidated.parquet is filtered for 2022-2023 if not already
    # For now, assume it contains all data and filter here.
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

    # 2. Generate temporal features
    df = generate_temporal_features(df)

    # 3. Generate lag features
    df, lag_feature_names = generate_lag_features(df) # Returns lag column names for dropna

    # 4. Merge Weather Data (already part of load_and_merge_external_data)
    # 5. Generate Public Holiday Features
    # 6. Generate School Holiday Features (Placeholder)
    # 7. Merge Event Data (already part of load_and_merge_external_data)
    
    # Combine steps 4, 5, 6, 7
    df = generate_public_holiday_features(df) # Public holidays
    df = generate_school_holiday_features(df) # School holidays (uses placeholder dates)
    df = load_and_merge_external_data(df)     # Weather and Events

    # 8. Handle NaNs from Lags
    if lag_feature_names:
        logger.info(f"Shape before dropping NaNs from lag features: {df.shape}")
        initial_rows = len(df)
        df.dropna(subset=lag_feature_names, inplace=True)
        rows_dropped = initial_rows - len(df)
        logger.info(f"Dropped {rows_dropped} rows due to NaNs in lag features. New shape: {df.shape}")
    else:
        logger.warning("No lag feature names returned, skipping dropna for lags.")

    # 9. Final Checks (optional)
    logger.info("Final column list:")
    for col in df.columns:
        logger.info(f" - {col} (dtype: {df[col].dtype}, NaNs: {df[col].isnull().sum()})")


    # 10. Save Output
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
    logger.info(f"Phase 1: Base Feature Engineering Pipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main() 