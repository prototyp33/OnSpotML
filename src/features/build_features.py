# src/features/build_features.py

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from pathlib import Path
import logging
import json
from sklearn.cluster import DBSCAN
import holidays
from typing import Union, List, Tuple, Dict # <<< --- ENSURE THIS IMPORT IS PRESENT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = Path(".").resolve()
RAW_DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
GTFS_DIR = RAW_DATA_DIR / "transport"

# Ensure processed data directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Functions ---

def load_data(filepath: Path, file_type: str = 'csv', **kwargs) -> Union[pd.DataFrame, gpd.GeoDataFrame, dict, None]:
    """Load data from various file types."""
    logger.info(f"Loading data from: {filepath}")
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None
    try:
        if file_type == 'csv':
            return pd.read_csv(filepath, **kwargs)
        elif file_type == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_type == 'geojson' or file_type == 'shapefile':
            return gpd.read_file(filepath, **kwargs)
        elif file_type == 'parquet':
             return pd.read_parquet(filepath, **kwargs)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}", exc_info=True)
        return None

def save_data(df: Union[pd.DataFrame, gpd.GeoDataFrame], filepath: Path, file_type: str = 'parquet', **kwargs):
    """Save data to various file types."""
    logger.info(f"Saving data to: {filepath}")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if file_type == 'parquet':
            # For GeoDataFrames, to_parquet handles geometry correctly by default if pyarrow is installed.
            # If you want to ensure it's saved as GeoParquet, geopandas has specific arguments,
            # but pandas to_parquet with a geometry column usually works if pyarrow is up to date.
            df.to_parquet(filepath, index=kwargs.pop('index', False), **kwargs)
        elif file_type == 'csv':
            # If df is GeoDataFrame, convert geometry to WKT for CSV saving if desired
            if isinstance(df, gpd.GeoDataFrame) and 'geometry' in df.columns:
                df_copy = df.copy()
                df_copy['geometry'] = df_copy['geometry'].apply(lambda geom: geom.wkt if geom else None)
                df_copy.to_csv(filepath, index=kwargs.pop('index', False), **kwargs)
            else:
                df.to_csv(filepath, index=kwargs.pop('index', False), **kwargs)
        elif file_type == 'geojson':
             if isinstance(df, gpd.GeoDataFrame):
                  df.to_file(filepath, driver='GeoJSON', index=kwargs.pop('index', False), **kwargs)
             else:
                  raise ValueError("Input must be a GeoDataFrame to save as GeoJSON")
        else:
            logger.error(f"Unsupported file type for saving: {file_type}")
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {e}", exc_info=True)


# --- Feature Engineering Functions ---

def create_temporal_features(df_in: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Create time-based features (hour, day of week, month, cyclical).
       Returns a DataFrame with only the new temporal features and the original index.
    """
    logger.info(f"Creating temporal features based on column: {timestamp_col}")
    if timestamp_col not in df_in.columns:
         logger.error(f"Timestamp column '{timestamp_col}' not found.")
         return pd.DataFrame(index=df_in.index) 
    
    df_temporal = pd.DataFrame(index=df_in.index) # Start with an empty DF with the correct index
    
    try:
        # Work on a Series for dt accessor
        dt_series = pd.to_datetime(df_in[timestamp_col])

        df_temporal['hour'] = dt_series.dt.hour
        df_temporal['day_of_week'] = dt_series.dt.dayofweek
        df_temporal['day_of_month'] = dt_series.dt.day
        df_temporal['month'] = dt_series.dt.month
        df_temporal['year'] = dt_series.dt.year
        df_temporal['is_weekend'] = df_temporal['day_of_week'].isin([5, 6]).astype(int)
        df_temporal['is_weekday'] = (~df_temporal['day_of_week'].isin([5, 6])).astype(int)

        df_temporal['hour_sin'] = np.sin(2 * np.pi * df_temporal['hour'] / 24.0)
        df_temporal['hour_cos'] = np.cos(2 * np.pi * df_temporal['hour'] / 24.0)
        df_temporal['day_of_week_sin'] = np.sin(2 * np.pi * df_temporal['day_of_week'] / 7.0)
        df_temporal['day_of_week_cos'] = np.cos(2 * np.pi * df_temporal['day_of_week'] / 7.0)
        df_temporal['month_sin'] = np.sin(2 * np.pi * (df_temporal['month'] - 1) / 12.0)
        df_temporal['month_cos'] = np.cos(2 * np.pi * (df_temporal['month'] - 1) / 12.0)

        df_temporal['day_of_year'] = dt_series.dt.dayofyear
        df_temporal['week_of_year'] = dt_series.dt.isocalendar().week.astype(int)
        df_temporal['quarter'] = dt_series.dt.quarter

        time_bins = [0, 6, 12, 18, 24]
        time_labels = ['night', 'morning', 'afternoon', 'evening']
        df_temporal['time_of_day_segment'] = pd.cut(df_temporal['hour'], bins=time_bins, labels=time_labels, right=False, include_lowest=True)

        unique_years = df_temporal['year'].unique()
        try:
            # Using normalize() on the dt_series to compare dates correctly
            country_holidays = holidays.country_holidays('ES', years=unique_years)
            df_temporal['is_public_holiday'] = dt_series.dt.normalize().isin(country_holidays).astype(int)
            logger.info(f"Identified {df_temporal['is_public_holiday'].sum()} public holiday occurrences using 'ES' national holidays.")
        except KeyError as ke:
            logger.warning(f"Could not get regional holidays directly for 'ES' due to: {ke}. Using national holidays.")
            country_holidays_national = holidays.country_holidays('ES', years=unique_years) # Fallback
            df_temporal['is_public_holiday'] = dt_series.dt.normalize().isin(country_holidays_national).astype(int)
            logger.info(f"Identified {df_temporal['is_public_holiday'].sum()} public holiday occurrences using 'ES' national holidays as fallback.")
        except Exception as holiday_e:
            logger.error(f"Error fetching holidays: {holiday_e}", exc_info=True)
            df_temporal['is_public_holiday'] = 0 

        logger.info("Temporal features created.")
        return df_temporal
    except Exception as e:
         logger.error(f"Error creating temporal features: {e}", exc_info=True)
         return pd.DataFrame(index=df_in.index)

def create_weather_features(weather_data: dict, target_timestamps_df: pd.DataFrame, timestamp_col:str = 'timestamp') -> pd.DataFrame:
    """Extract relevant weather features and align with target timestamps DataFrame.
       Returns a DataFrame with new weather features, indexed like target_timestamps_df.
    """
    logger.info("Processing weather features...")
    
    if not weather_data or 'forecast' not in weather_data or 'forecastday' not in weather_data['forecast']:
        logger.warning("Weather data is missing or not in the expected format. Returning empty DataFrame.")
        return pd.DataFrame(index=target_timestamps_df.index)

    hourly_data = []
    for day_forecast in weather_data['forecast']['forecastday']:
        for hour_data in day_forecast.get('hour', []):
            hourly_data.append(hour_data)

    if not hourly_data:
        logger.warning("No hourly weather data found in the forecast. Returning empty DataFrame.")
        return pd.DataFrame(index=target_timestamps_df.index)

    weather_df = pd.DataFrame(hourly_data)

    try:
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        
        # ---- TIMEZONE HANDLING for weather_df['time'] ----
        if weather_df['time'].dt.tz is None:
            logger.info("Weather timestamps are naive. Localizing to 'Europe/Madrid'.")
            weather_df['time'] = weather_df['time'].dt.tz_localize('Europe/Madrid', ambiguous='infer', nonexistent='NaT')
        elif str(weather_df['time'].dt.tz) != 'Europe/Madrid':
            logger.info(f"Weather timestamps are timezone-aware ({weather_df['time'].dt.tz}). Converting to 'Europe/Madrid'.")
            weather_df['time'] = weather_df['time'].dt.tz_convert('Europe/Madrid')
        # ---- END TIMEZONE HANDLING ----

    except Exception as e:
        logger.error(f"Error converting or localizing weather 'time' to datetime: {e}. Returning empty DataFrame.", exc_info=True)
        return pd.DataFrame(index=target_timestamps_df.index)

    feature_map = {
        'time': 'weather_timestamp', 'temp_c': 'temp_c', 'precip_mm': 'precip_mm',
        'wind_kph': 'wind_kph', 'humidity': 'humidity', 'cloud': 'cloud_cover'
    }
    weather_df['condition_text'] = weather_df['condition'].apply(lambda x: x.get('text') if isinstance(x, dict) else None)
    
    relevant_weather_cols = list(feature_map.keys()) + ['condition_text']
    weather_df = weather_df[[col for col in relevant_weather_cols if col in weather_df.columns]].rename(columns=feature_map)
    
    if timestamp_col not in target_timestamps_df.columns:
        logger.error(f"Target timestamps DataFrame must have the column '{timestamp_col}'.")
        return pd.DataFrame(index=target_timestamps_df.index)
    
    target_df_copy = target_timestamps_df.copy() # Work with a copy
    # ---- TIMEZONE HANDLING for target_df_copy[timestamp_col] ----
    if target_df_copy[timestamp_col].dt.tz is None:
        logger.info(f"Target timestamp column '{timestamp_col}' is naive. Localizing to 'Europe/Madrid'.")
        target_df_copy[timestamp_col] = target_df_copy[timestamp_col].dt.tz_localize('Europe/Madrid', ambiguous='infer', nonexistent='NaT')
    elif str(target_df_copy[timestamp_col].dt.tz) != 'Europe/Madrid':
        logger.info(f"Target timestamp column '{timestamp_col}' is {target_df_copy[timestamp_col].dt.tz}. Converting to 'Europe/Madrid'.")
        target_df_copy[timestamp_col] = target_df_copy[timestamp_col].dt.tz_convert('Europe/Madrid')
    # ---- END TIMEZONE HANDLING ----
            
    target_df_sorted = target_df_copy.sort_values(by=timestamp_col)
    weather_df_sorted = weather_df.sort_values(by='weather_timestamp')
    
    logger.info(f"Attempting merge_asof. Left key type: {target_df_sorted[timestamp_col].dtype}, Right key type: {weather_df_sorted['weather_timestamp'].dtype}")
    
    merged_features = pd.merge_asof(
        target_df_sorted,
        weather_df_sorted,
        left_on=timestamp_col,
        right_on='weather_timestamp',
        direction='nearest',
    )
    
    merged_features = merged_features.set_index(target_df_sorted.index).reindex(target_timestamps_df.index)

    # Define expected weather feature columns
    expected_weather_feature_cols = ['temp_c', 'precip_mm', 'wind_kph', 'humidity', 'cloud_cover', 'condition_text']
    # Select only these columns if they exist in merged_features
    final_weather_cols = [col for col in expected_weather_feature_cols if col in merged_features.columns]
    final_weather_features = merged_features[final_weather_cols]

    if final_weather_features.empty and not target_timestamps_df.empty:
        logger.warning("Weather feature DataFrame is empty after processing. Returning DataFrame with NaNs.")
        return pd.DataFrame(index=target_timestamps_df.index, columns=expected_weather_feature_cols)

    logger.info(f"Weather features created successfully with {len(final_weather_features.columns)} columns.")
    return final_weather_features

def create_event_features(events_df: pd.DataFrame, target_timestamps_series: pd.Series) -> pd.DataFrame:
    logger.info("Creating event features...")
    event_features_out = pd.DataFrame(index=target_timestamps_series.index)
    event_features_out['is_event_ongoing'] = 0

    if events_df is None or events_df.empty:
        logger.warning("No event data provided or empty, skipping event feature creation.")
        return event_features_out[['is_event_ongoing']]
    
    try:
        if 'DataInici' not in events_df.columns or 'DataFi' not in events_df.columns:
            logger.error("Event data must contain 'DataInici' and 'DataFi' columns.")
            return event_features_out[['is_event_ongoing']]
            
        events_df_copy = events_df.copy()
        events_df_copy['DataInici'] = pd.to_datetime(events_df_copy['DataInici'], errors='coerce')
        events_df_copy['DataFi'] = pd.to_datetime(events_df_copy['DataFi'], errors='coerce')
        
        # Ensure DataFi is at the end of the day for date-based comparisons if it's just a date
        events_df_copy['DataFi'] = events_df_copy['DataFi'].apply(lambda x: x.replace(hour=23, minute=59, second=59) if pd.notnull(x) and x.time() == pd.Timestamp("00:00:00").time() else x)
        events_df_copy.dropna(subset=['DataInici', 'DataFi'], inplace=True)

        if events_df_copy.empty:
            logger.warning("No valid event date ranges after cleaning. Skipping event feature creation.")
            return event_features_out[['is_event_ongoing']]

        # Consistent Timezone Handling:
        # Ensure target_timestamps_series is aware and in 'Europe/Madrid'
        # (This should be guaranteed if it comes from parking_gdf['timestamp'])
        ts_series_for_comparison = target_timestamps_series.copy()
        if ts_series_for_comparison.dt.tz is None:
            logger.warning("Target timestamps for events are naive. Localizing to 'Europe/Madrid'.")
            ts_series_for_comparison = ts_series_for_comparison.dt.tz_localize('Europe/Madrid', ambiguous='infer', nonexistent='NaT')
        elif str(ts_series_for_comparison.dt.tz) != 'Europe/Madrid':
            ts_series_for_comparison = ts_series_for_comparison.dt.tz_convert('Europe/Madrid')

        # Make event dates timezone-aware to 'Europe/Madrid'
        # If they are already datetime objects from pd.to_datetime, they are naive.
        if events_df_copy['DataInici'].dt.tz is None:
            events_df_copy['DataInici'] = events_df_copy['DataInici'].dt.tz_localize('Europe/Madrid', ambiguous='infer', nonexistent='NaT')
        else:
            events_df_copy['DataInici'] = events_df_copy['DataInici'].dt.tz_convert('Europe/Madrid')

        if events_df_copy['DataFi'].dt.tz is None:
            events_df_copy['DataFi'] = events_df_copy['DataFi'].dt.tz_localize('Europe/Madrid', ambiguous='infer', nonexistent='NaT')
        else:
            events_df_copy['DataFi'] = events_df_copy['DataFi'].dt.tz_convert('Europe/Madrid')
        
        # Drop rows where localization might have failed (resulted in NaT)
        events_df_copy.dropna(subset=['DataInici', 'DataFi'], inplace=True)
        if events_df_copy.empty:
            logger.warning("No valid event date ranges after timezone localization. Skipping event feature creation.")
            return event_features_out[['is_event_ongoing']]

        is_event_ongoing_list = []
        for ts in ts_series_for_comparison:
            if pd.isna(ts): 
                is_event_ongoing_list.append(0)
                continue
            # Now both ts and event dates should be aware and in the same timezone
            is_during_event = any(
                (event['DataInici'] <= ts <= event['DataFi']) for _, event in events_df_copy.iterrows()
            )
            is_event_ongoing_list.append(1 if is_during_event else 0)
        
        event_features_out['is_event_ongoing'] = is_event_ongoing_list
        
        logger.info(f"Event features created. {event_features_out['is_event_ongoing'].sum()} event occurrences marked.")
    except Exception as e:
        logger.error(f"Error creating event features: {e}", exc_info=True)
        event_features_out['is_event_ongoing'] = 0 # Default to 0 in case of any error
        
    return event_features_out[['is_event_ongoing']]

def create_spatial_cluster_features(
    gdf: gpd.GeoDataFrame, 
    geometry_col: str = 'geometry', 
    id_col: str = 'temp_join_id', # Defaulting to temp_join_id as per build_all_features logic
    target_crs: str = "EPSG:25831", 
    eps: float = 100.0, 
    min_samples: int = 3
) -> pd.DataFrame:
    logger.info(f"Creating spatial cluster features using DBSCAN (eps={eps}, min_samples={min_samples})...")

    if gdf.empty:
        logger.warning("Input GeoDataFrame is empty. Skipping spatial clustering.")
        return pd.DataFrame(columns=[id_col, 'cluster_label'])

    if id_col not in gdf.columns:
        logger.error(f"ID column '{id_col}' not found. Clustering requires a unique ID.")
        # Attempt to use index if it's named and unique
        if gdf.index.name == id_col and gdf.index.is_unique:
            gdf_copy_for_cluster = gdf.reset_index() 
            logger.info(f"Using index '{id_col}' as ID column for clustering.")
        else:
            logger.error(f"Cannot proceed without a valid unique '{id_col}'.")
            return pd.DataFrame(columns=[id_col, 'cluster_label'])
    else:
        gdf_copy_for_cluster = gdf.copy()
            
    if geometry_col not in gdf_copy_for_cluster.columns:
        logger.error(f"Geometry column '{geometry_col}' not found.")
        return pd.DataFrame(columns=[id_col, 'cluster_label'])

    # Ensure id_col is unique before using it for merging results
    if not gdf_copy_for_cluster[id_col].is_unique:
        logger.error(f"ID column '{id_col}' is not unique. Clustering results may be incorrect.")
        # Potentially raise error or return empty if uniqueness is critical for downstream
        # For now, proceed with caution.
    
    gdf_with_geom = gdf_copy_for_cluster.dropna(subset=[geometry_col])
    if gdf_with_geom.empty:
        logger.warning(f"No valid geometries after dropna. Skipping clustering.")
        return pd.DataFrame(columns=[id_col, 'cluster_label'])

    # Keep only necessary columns for projection and clustering
    gdf_proj = gdf_with_geom[[id_col, geometry_col]].copy() 

    if gdf_proj.crs is None:
        logger.warning(f"GeoDataFrame CRS is not set. Assuming WGS84 (EPSG:4326) for reprojection to {target_crs}.")
        gdf_proj = gdf_proj.set_crs("EPSG:4326", allow_override=True).to_crs(target_crs)
    elif str(gdf_proj.crs).upper() != target_crs.upper():
        logger.info(f"Reprojecting geometries from {gdf_proj.crs} to {target_crs}...")
        gdf_proj = gdf_proj.to_crs(target_crs)
    
    gdf_proj[geometry_col] = gdf_proj[geometry_col].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    gdf_proj = gdf_proj[gdf_proj[geometry_col].is_valid & ~gdf_proj[geometry_col].is_empty]
    
    if gdf_proj.empty:
        logger.warning(f"No valid geometries after validation for centroids. Skipping clustering.")
        return pd.DataFrame(columns=[id_col, 'cluster_label'])
        
    centroids = gdf_proj[geometry_col].centroid
    coordinates = np.array(list(zip(centroids.x, centroids.y)))

    if coordinates.shape[0] < min_samples:
        logger.warning(f"Not enough samples ({coordinates.shape[0]}) for DBSCAN with min_samples={min_samples}. Assigning all to noise cluster (-1).")
        # Create a DataFrame with the id_col and -1 for cluster_label
        return pd.DataFrame({id_col: gdf_proj[id_col].tolist(), 'cluster_label': -1})


    logger.info(f"Running DBSCAN on {len(coordinates)} points...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(coordinates)
    
    gdf_proj['cluster_label'] = cluster_labels

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = np.sum(cluster_labels == -1)
    logger.info(f"Spatial clustering complete. Found {num_clusters} clusters and {num_noise} noise points.")

    return gdf_proj[[id_col, 'cluster_label']]


def create_gtfs_features(parking_gdf: gpd.GeoDataFrame, gtfs_dir: Path, geometry_col: str = 'geometry') -> pd.DataFrame:
    logger.info("Creating GTFS features...")
    gtfs_features_out = pd.DataFrame(index=parking_gdf.index)
    default_gtfs_cols = ['distance_nearest_stop', 'bus_stop_density_500m']
    for col in default_gtfs_cols: 
        gtfs_features_out[col] = np.nan if col == 'distance_nearest_stop' else 0


    stops_path = gtfs_dir / 'stops.txt'
    if not stops_path.exists():
        logger.error(f"GTFS stops file not found: {stops_path}")
        return gtfs_features_out
    
    try:
        stops_df = pd.read_csv(stops_path)
        if not all(col in stops_df.columns for col in ['stop_lon', 'stop_lat']):
            logger.error("GTFS stops.txt missing 'stop_lon' or 'stop_lat' columns.")
            return gtfs_features_out

        stops_gdf = gpd.GeoDataFrame(
            stops_df,
            geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
            crs="EPSG:4326"
        )
        
        if geometry_col not in parking_gdf.columns or parking_gdf[geometry_col].isnull().all():
             logger.error("Parking data GeoDataFrame has no valid geometry column.")
             return gtfs_features_out
        if parking_gdf.crs is None:
             logger.error("Parking data GeoDataFrame needs a CRS defined. Assuming EPSG:4326 for safety.")
             parking_gdf = parking_gdf.set_crs("EPSG:4326", allow_override=True)
        
        target_crs = "EPSG:25831" 
        logger.info(f"Reprojecting to {target_crs} for distance calculations...")
        # Work with copies for projection to avoid SettingWithCopyWarning if parking_gdf is a slice
        parking_proj = parking_gdf.copy().to_crs(target_crs)
        stops_proj = stops_gdf.to_crs(target_crs)

        logger.info("Calculating distance to nearest stop...")
        try:
            # sjoin_nearest requires GeoPandas >= 0.10.0
            # Pass only the geometry column and ensure index is preserved
            joined_gdf = gpd.sjoin_nearest(parking_proj[[geometry_col]], stops_proj, how="left", distance_col="distance_nearest_stop")
            # joined_gdf should have the same index as parking_proj
            gtfs_features_out['distance_nearest_stop'] = joined_gdf['distance_nearest_stop']
        except AttributeError:
            logger.error("sjoin_nearest not available (requires GeoPandas >= 0.10.0). Distance to nearest stop not calculated.")
        except Exception as e:
            logger.error(f"Error calculating distance to nearest stop: {e}", exc_info=True)

        logger.info("Calculating stop density within 500m...")
        buffer_radius = 500 
        parking_buffers = parking_proj.geometry.buffer(buffer_radius)
        parking_buffers_gdf = gpd.GeoDataFrame(geometry=parking_buffers, index=parking_proj.index, crs=parking_proj.crs)
        
        try:
            # Use predicate="within" for newer GeoPandas
            stops_in_buffer = gpd.sjoin(stops_proj, parking_buffers_gdf, how="inner", predicate="within") 
            if not stops_in_buffer.empty and 'index_right' in stops_in_buffer.columns:
                stop_counts = stops_in_buffer.groupby('index_right').size()
                gtfs_features_out['bus_stop_density_500m'] = stop_counts.reindex(parking_proj.index).fillna(0).astype(int)
            else:
                logger.info("No bus stops found within buffers or 'index_right' column missing. Density set to 0.")
                gtfs_features_out['bus_stop_density_500m'] = 0
        except TypeError as te: # Catch specific error for 'op' vs 'predicate'
            if "unexpected keyword argument 'op'" in str(te) or "unexpected keyword argument 'predicate'" in str(te):
                 logger.warning(f"Trying alternative sjoin syntax due to TypeError: {te}")
                 try: # Try with 'op' for older versions
                      stops_in_buffer = gpd.sjoin(stops_proj, parking_buffers_gdf, how="inner", op="within")
                      if not stops_in_buffer.empty and 'index_right' in stops_in_buffer.columns:
                          stop_counts = stops_in_buffer.groupby('index_right').size()
                          gtfs_features_out['bus_stop_density_500m'] = stop_counts.reindex(parking_proj.index).fillna(0).astype(int)
                      else:
                          logger.info("No bus stops found within buffers (fallback syntax) or 'index_right' column missing. Density set to 0.")
                          gtfs_features_out['bus_stop_density_500m'] = 0
                 except Exception as e_op:
                      logger.error(f"Error calculating stop density with fallback 'op' syntax: {e_op}", exc_info=True)
                      # Ensure default value if error occurs in fallback
                      gtfs_features_out['bus_stop_density_500m'] = 0
            else:
                 logger.error(f"TypeError calculating stop density: {te}", exc_info=True)
                 gtfs_features_out['bus_stop_density_500m'] = 0 # Ensure default
        except Exception as e:
            logger.error(f"Error calculating stop density: {e}", exc_info=True)
            gtfs_features_out['bus_stop_density_500m'] = 0 # Ensure default
        logger.info("GTFS features created.")
    except Exception as e:
        logger.error(f"Error creating GTFS features: {e}", exc_info=True)

    return gtfs_features_out


# --- Placeholder Functions (Implement based on your specific data and needs) ---

def load_and_prepare_parking_data(parking_data_path: Path, coord_cols: Tuple[str, str]) -> gpd.GeoDataFrame | None:
    """
    Loads raw parking data (e.g., from a CSV like 'trams_aparcament.csv'),
    creates a geometry column from coordinate columns, and returns a GeoDataFrame.
    This function is a placeholder. If your main parking data is already a GeoParquet
    (e.g., the output of `add_poi_features.py`), you would load that directly in 
    `build_all_features` and this function might not be used for that specific input.
    """
    logger.info(f"Placeholder: Loading and preparing parking data from {parking_data_path} using coord_cols: {coord_cols}...")
    df = load_data(parking_data_path, file_type='csv', low_memory=False) 
    if df is None:
        return None
    
    try:
        if not (coord_cols[0] in df.columns and coord_cols[1] in df.columns):
            logger.error(f"Coordinate columns {coord_cols} not found in {parking_data_path}.")
            return None

        # Create Point geometries. Adjust if your data has LineStrings (e.g., from WKT).
        geometry = [Point(xy) for xy in zip(df[coord_cols[0]], df[coord_cols[1]])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326") # Assume WGS84 for raw coords
        
        logger.info(f"Parking data prepared from CSV (placeholder): {len(gdf)} records.")
        return gdf
    except Exception as e:
        logger.error(f"Error preparing parking data from CSV (placeholder): {e}", exc_info=True)
        return None

def determine_id_column(df: pd.DataFrame, preferred_candidates: List[str]) -> str | None:
    """Determines a suitable unique ID column from the DataFrame."""
    logger.info(f"Determining ID column from candidates: {preferred_candidates}...")
    for col in preferred_candidates:
        if col in df.columns:
            if df[col].is_unique:
                logger.info(f"Using '{col}' as unique ID column.")
                return col
            else:
                logger.warning(f"Column '{col}' found but is not unique. Checking other candidates.")
    
    common_ids = ['id', 'ID', 'OBJECTID', 'fid'] 
    for col in common_ids:
        if col in df.columns:
            if df[col].is_unique:
                logger.info(f"Using fallback '{col}' as unique ID column.")
                return col
            else:
                logger.warning(f"Fallback column '{col}' found but is not unique.")

    if df.index.is_unique and df.index.name is not None and not isinstance(df.index, pd.RangeIndex):
        logger.info(f"Using named index '{df.index.name}' as unique ID column. Consider resetting index if column is needed for merging.")
        return df.index.name 

    logger.warning("No suitable unique ID column found. Features might not merge correctly if indices are not aligned or a temp ID is used.")
    return None


# --- Main Processing Function ---

def build_all_features():
    """Main function to orchestrate the creation of all features."""
    logger.info("Starting feature building process...")

    # --- Load Base Parking Data ---
    base_parking_data_path = PROCESSED_DATA_DIR / "parking_predictions_with_pois.parquet" 
    logger.info(f"Attempting to load base parking data from: {base_parking_data_path}")
    
    try:
        parking_gdf = gpd.read_parquet(base_parking_data_path)
        logger.info(f"Successfully loaded base parking data: {len(parking_gdf)} records with {len(parking_gdf.columns)} columns.")
        
        if 'geometry' not in parking_gdf.columns:
            raise ValueError("Loaded parking data is missing 'geometry' column.")
        if parking_gdf.crs is None:
            logger.warning(f"CRS for {base_parking_data_path} is None. Attempting to set to WGS84 (EPSG:4326).")
            parking_gdf = parking_gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            logger.info(f"Loaded parking data CRS: {parking_gdf.crs}")

    except Exception as e:
        logger.error(f"Could not load or prepare base parking data from {base_parking_data_path}: {e}", exc_info=True)
        return None

    if parking_gdf.empty:
        logger.error("Base parking data is empty after loading. Aborting feature building.")
        return None

    main_id_column = determine_id_column(parking_gdf, preferred_candidates=['ID_TRAMO', 'ID_TRAM', 'segment_id', 'parking_id', 'osmid'])
    
    if not main_id_column:
        logger.warning("No unique ID column found in base parking data. Creating 'temp_join_id' from index.")
        if parking_gdf.index.is_unique:
            parking_gdf = parking_gdf.reset_index().rename(columns={'index': 'temp_join_id'})
            main_id_column = 'temp_join_id'
        else:
            logger.error("Index is not unique, cannot create reliable 'temp_join_id'. Aborting.")
            return None
    logger.info(f"Using '{main_id_column}' as the main ID for joining features.")

    # Prepare a list to hold all feature DataFrames for the final merge
    # Start with the base data, ensuring it's indexed by the main_id_column
    if parking_gdf.index.name == main_id_column:
        parking_gdf_indexed = parking_gdf.copy() # Already indexed by main_id_column, main_id_column is not a regular column
    else:
        # main_id_column is a regular column (or temp_join_id was created as a column)
        # We need to set it as index.
        if main_id_column not in parking_gdf.columns:
             logger.error(f"Critical error: main_id_column '{main_id_column}' not found in parking_gdf columns for indexing.")
             return None
        # Set main_id_column as index and drop it as a regular column to avoid ambiguity
        parking_gdf_indexed = parking_gdf.set_index(main_id_column, drop=True)
    # --- Temporal Features ---
    timestamp_col_for_features = 'timestamp' 
    if timestamp_col_for_features not in parking_gdf.columns:
        logger.warning(f"'{timestamp_col_for_features}' column not found. Creating a dummy current timestamp for all records.")
        parking_gdf[timestamp_col_for_features] = pd.Timestamp.now(tz='Europe/Madrid').floor('H') 
    
    # Pass DataFrame with original index and the timestamp column
    temporal_features = create_temporal_features(parking_gdf_indexed[[timestamp_col_for_features]], timestamp_col_for_features)
    if not temporal_features.empty:
        save_data(temporal_features.reset_index(), PROCESSED_DATA_DIR / "temporal_features_set.parquet")
        # Merge back to parking_gdf_indexed using its index (which is main_id_column)
        parking_gdf_indexed = parking_gdf_indexed.join(temporal_features, how='left', rsuffix='_temp')


    # --- Weather Features ---
    weather_data_path = RAW_DATA_DIR / "weather" / "realtime_weather.json"
    weather_data = load_data(weather_data_path, file_type='json')
    if weather_data:
        # Pass DataFrame with original index and the timestamp column
        # Pass DataFrame indexed by main_id_column
        weather_features = create_weather_features(weather_data, parking_gdf_indexed[[timestamp_col_for_features]], timestamp_col=timestamp_col_for_features)
        if not weather_features.empty:
            save_data(weather_features.reset_index(), PROCESSED_DATA_DIR / "weather_features_set.parquet")
            parking_gdf_indexed = parking_gdf_indexed.join(weather_features, how='left', rsuffix='_weather')
    else:
        logger.warning("Weather data not found or failed to load. Skipping weather features.")

    # --- Event Features ---
    events_data_path = RAW_DATA_DIR / "events" / "cultural_events.csv"
    events_df_raw = load_data(events_data_path, file_type='csv') 
    if events_df_raw is not None:
        # Pass Series indexed by main_id_column
        event_features = create_event_features(events_df_raw, parking_gdf_indexed[timestamp_col_for_features])
        if not event_features.empty:
            save_data(event_features.reset_index(), PROCESSED_DATA_DIR / "event_features_set.parquet")
            parking_gdf_indexed = parking_gdf_indexed.join(event_features, how='left', rsuffix='_event')
    else:
        logger.warning("Events data not found or failed to load. Skipping event features.")

    # --- Spatial Cluster Features ---
    if main_id_column and 'geometry' in parking_gdf.columns:
        # Prepare input for create_spatial_cluster_features.
        # It needs main_id_column as a regular column and the geometry.
        if parking_gdf.index.name == main_id_column:
            # If main_id_column was the original index, reset it to make it a column for input
            input_gdf_for_spatial = parking_gdf.reset_index()
        else:
            # main_id_column is already a regular column in parking_gdf
            input_gdf_for_spatial = parking_gdf.copy()
        # create_spatial_cluster_features returns a DF with [id_col, 'cluster_label']
        spatial_cluster_df = create_spatial_cluster_features(
            input_gdf_for_spatial[[main_id_column, 'geometry']], 
            id_col=main_id_column,
            geometry_col='geometry',
            target_crs="EPSG:25831", 
            eps=100.0, 
            min_samples=5 
        )
        if not spatial_cluster_df.empty:
            save_data(spatial_cluster_df, PROCESSED_DATA_DIR / "spatial_cluster_features_set.parquet")
            # Convert spatial_cluster_df to be indexed by main_id_column for joining
            spatial_cluster_df_indexed = spatial_cluster_df.set_index(main_id_column)
            parking_gdf_indexed = parking_gdf_indexed.join(spatial_cluster_df_indexed, how='left', rsuffix='_cluster')
    else:
        logger.warning("Skipping spatial cluster features: missing ID or geometry in base parking data.")

    # --- GTFS Features ---
    # Check geometry in parking_gdf_indexed as it's the one being used for features
    # parking_gdf_indexed should retain geometry if it was in parking_gdf
    if 'geometry' in parking_gdf_indexed.columns: 
        # create_gtfs_features should return features indexed like its input gdf (parking_gdf_indexed)
        gtfs_features = create_gtfs_features(parking_gdf_indexed, GTFS_DIR, geometry_col='geometry')
        if not gtfs_features.empty:
            save_data(gtfs_features.reset_index(), PROCESSED_DATA_DIR / "gtfs_features_set.parquet")
            parking_gdf_indexed = parking_gdf_indexed.join(gtfs_features, how='left', rsuffix='_gtfs')
    else:
        logger.warning("Skipping GTFS features: missing geometry column in base parking data.")
    
    # Reset index if main_id_column was set as index for joining
    if parking_gdf_indexed.index.name == main_id_column:
        final_gdf = parking_gdf_indexed.reset_index()
    else: # Should not happen if logic is correct, but as a fallback
        final_gdf = parking_gdf_indexed

    # --- Save Final Combined DataFrame ---
    final_output_path = PROCESSED_DATA_DIR / "features_master_table.parquet"
    logger.info(f"Saving final combined feature table ({len(final_gdf)} rows, {len(final_gdf.columns)} columns) to {final_output_path}")
    save_data(final_gdf, final_output_path)
    
    logger.info("Feature building process completed.")
    return final_gdf


if __name__ == "__main__":
    master_features_df = build_all_features()
    if master_features_df is not None:
        logger.info("Master feature table generated successfully.")
        logger.info(f"Shape of master table: {master_features_df.shape}")
        logger.info(f"Columns: {master_features_df.columns.tolist()}")
        logger.info(f"First 5 rows of master table:\n{master_features_df.head()}")
    else:
        logger.error("Master feature table generation failed.")
