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
import osmnx as ox
import re
import time
from typing import Union, List, Tuple, Dict, Any, Optional
from tqdm import tqdm

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
OSM_PBF_PATH = RAW_DATA_DIR / "osm/cataluna-latest.osm.pbf"  # Define path for OSM data
# ox.config(use_cache=True, log_console=True) # Deprecated
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.default_crs = "EPSG:25831"

# Input data paths
PARKING_DATA_PATH = PROCESSED_DATA_DIR / "parking_predictions_phase1_enriched.parquet"
WEATHER_DATA_PATH = RAW_DATA_DIR / "weather" / "historical_weather.csv"
EVENTS_DATA_PATH = RAW_DATA_DIR / "events" / "events.csv"

# Output path
MASTER_TABLE_PATH = PROCESSED_DATA_DIR / "features_master_table.parquet"
# --- End Configuration ---

# Ensure processed data directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Functions ---

def load_data(filepath: Path, file_type: str = 'csv', **kwargs: Any) -> Union[pd.DataFrame, gpd.GeoDataFrame, dict, None]:
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
        elif file_type in ['geojson', 'shapefile', 'gpkg']:
            return gpd.read_file(filepath, **kwargs)
        elif file_type == 'parquet':
             # Use read_parquet for both pandas and geopandas dataframes
             # The user of the function must know if it's a geodataframe
             try:
                 return gpd.read_parquet(filepath, **kwargs)
             except (ValueError, TypeError, KeyError): # If it fails, assume it's a normal parquet
                 return pd.read_parquet(filepath, **kwargs)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}", exc_info=True)
        return None

def save_data(df: Union[pd.DataFrame, gpd.GeoDataFrame], filepath: Path, file_type: str = 'parquet', **kwargs: Any) -> None:
    """Save data to various file types."""
    logger.info(f"Saving data to: {filepath}")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if file_type == 'parquet':
            # This will save as GeoParquet if df is a GeoDataFrame
            df.to_parquet(filepath, index=kwargs.pop('index', False), **kwargs)
        elif file_type == 'csv':
            if isinstance(df, gpd.GeoDataFrame) and 'geometry' in df.columns:
                df_copy = df.copy()
                # Convert geometry to WKT for CSV compatibility
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

def create_temporal_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """Creates time-based features from a timestamp column."""
    logger.info(f"Creating temporal features from column '{timestamp_col}'...")
    
    timestamps = pd.to_datetime(df[timestamp_col])
    features_df = pd.DataFrame(index=df.index)
    
    features_df['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
    features_df['dayofweek_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
    features_df['dayofweek_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
    features_df['month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
    features_df['month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)
    features_df['is_weekend'] = (timestamps.dt.weekday >= 5).astype(int)
    
    es_holidays = holidays.CountryHoliday('ES', prov='CT')
    features_df['is_public_holiday'] = timestamps.dt.date.isin(es_holidays).astype(int)
    
    return features_df

def create_temporal_lag_features(df: pd.DataFrame,
                                 id_col: str = 'ID_TRAMO',
                                 timestamp_col: str = 'timestamp',
                                 lags: List[int] = [1],
                                 roll_windows: List[int] = [3, 6]) -> pd.DataFrame:
    """Create lagged occupancy and rolling mean features per segment.

    Parameters
    ----------
    df : DataFrame containing at least `id_col`, `timestamp_col`, and `occupancy_level`.
    lags : list of hours to shift backwards.
    roll_windows : list of hours over which to compute rolling mean.
    """
    logger.info("Creating temporal lag & rolling features …")

    required_cols = {id_col, timestamp_col, 'occupancy_level'}
    if not required_cols.issubset(df.columns):
        logger.warning("Missing columns for lag features – skipping.")
        return pd.DataFrame(index=df.index)

    temp = df[[id_col, timestamp_col, 'occupancy_level']].copy()
    temp = temp.sort_values([id_col, timestamp_col])

    features = pd.DataFrame(index=df.index)

    grouped = temp.groupby(id_col, group_keys=False)

    # Lags
    for h in lags:
        lag_name = f'occ_prev_{h}h'
        features[lag_name] = grouped['occupancy_level'].shift(h)

    # Rolling means (using window in number of periods assuming hourly data)
    for w in roll_windows:
        roll_name = f'occ_roll{w}_mean'
        features[roll_name] = grouped['occupancy_level'].apply(lambda x: x.rolling(window=w, min_periods=1).mean())

    return features

def create_weather_features(weather_data: pd.DataFrame, target_timestamps_df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """Extract weather features and align with target timestamps."""
    logger.info("Processing weather features...")
    if weather_data is None or weather_data.empty:
        logger.warning("Weather data is empty. Skipping weather features.")
        return pd.DataFrame(index=target_timestamps_df.index)

    weather_df = weather_data.copy()
    weather_df.rename(columns={'DATA_LECTURA': 'weather_timestamp', 'VALOR': 'value'}, inplace=True)
    weather_df['weather_timestamp'] = pd.to_datetime(weather_df['weather_timestamp'])
    
    # Pivot the table to get weather metrics as columns
    weather_df_pivot = weather_df.pivot_table(
        index='weather_timestamp', 
        columns='ACRÒNIM', 
        values='value',
        aggfunc='mean'
    ).reset_index()

    # Rename columns to be more descriptive, e.g., from 'TM' to 'weather_tm'
    weather_df_pivot.columns = ['weather_' + col.lower() if col != 'weather_timestamp' else col for col in weather_df_pivot.columns]
    
    # Ensure weather timestamp is timezone-aware to match the target dataframe
    if weather_df_pivot['weather_timestamp'].dt.tz is None:
        weather_df_pivot['weather_timestamp'] = weather_df_pivot['weather_timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid')

    target_df = pd.DataFrame(index=target_timestamps_df.index)
    # Ensure target timestamp is also timezone-aware
    target_timestamps = pd.to_datetime(target_timestamps_df[timestamp_col])
    if target_timestamps.dt.tz is None:
        target_timestamps = target_timestamps.dt.tz_localize('Europe/Madrid')
    
    target_df['timestamp_floor'] = target_timestamps.dt.floor('h')
    
    merged_df = pd.merge_asof(
        target_df.sort_values('timestamp_floor'), 
        weather_df_pivot.sort_values('weather_timestamp'), 
        left_on='timestamp_floor', 
        right_on='weather_timestamp', 
        direction='nearest'
    )
    
    # Define final feature columns based on what's available after pivot
    feature_cols = [col for col in merged_df.columns if col.startswith('weather_')]
    final_features = merged_df[feature_cols]

    # Temperature delta (if temperature metric exists, commonly 'weather_tm')
    if 'weather_tm' in final_features.columns:
        final_features['weather_tm_delta_1h'] = final_features['weather_tm'].diff().fillna(0)

    return final_features

def create_event_features(events_df: pd.DataFrame, target_timestamps_series: pd.Series) -> pd.DataFrame:
    """Create features based on proximity to special events."""
    logger.info("Creating event features...")
    if events_df is None or events_df.empty:
        return pd.DataFrame(index=target_timestamps_series.index)
        
    # Convert to datetime and make timezone-aware
    start_times = pd.to_datetime(events_df['DataInici']).dt.tz_localize('Europe/Madrid')
    end_times = pd.to_datetime(events_df['DataFi']).dt.tz_localize('Europe/Madrid')
    
    target_df = pd.DataFrame({'timestamp': pd.to_datetime(target_timestamps_series)})
    if target_df['timestamp'].dt.tz is None:
        target_df['timestamp'] = target_df['timestamp'].dt.tz_localize('Europe/Madrid')
    
    target_df['is_event'] = 0
    
    for i in range(len(events_df)):
        event_mask = (target_df['timestamp'] >= start_times.iloc[i]) & (target_df['timestamp'] <= end_times.iloc[i])
        target_df.loc[event_mask, 'is_event'] = 1
        
    return target_df.set_index(target_timestamps_series.index)[['is_event']]

def create_poi_features(
    gdf: gpd.GeoDataFrame,
    radii: List[int] = [100, 200, 500],
) -> gpd.GeoDataFrame:
    """Fetches POIs (cached) and computes radius-based counts.

    • Flattens the nested tag mapping into the *flat* ``{key: [values]}`` format
      required by `osmnx.features_from_bbox`.
    • After download, assigns a ``category`` label so we can count by our own
      business categories even though the Overpass query was aggregated.
    """

    logger.info("--- Generating and Joining POI Features (Optimized & Cached) ---")

    if 'geometry' not in gdf.columns:
        logger.error("Input GeoDataFrame must have a 'geometry' column – skipping POI features.")
        return gdf

    # ---------------- Category → tag mapping ----------------
    poi_categories: Dict[str, Dict[str, List[str]]] = {
        'education': {'amenity': ['school', 'university']},
        'health': {'amenity': ['hospital', 'clinic', 'doctors', 'pharmacy']},
        'transport': {
            'public_transport': ['station', 'stop_position'],
            'railway': ['station', 'subway_entrance'],
        },
        'sustenance': {'amenity': ['restaurant', 'cafe', 'fast_food', 'bar']},
        'shop': {'shop': ['supermarket', 'convenience', 'mall']},
    }

    poi_cache_path = PROCESSED_DATA_DIR / "pois_master_cache.parquet"

    # ---------------- Load or download POIs ----------------
    if poi_cache_path.exists():
        logger.info(f"Loading cached POI data from {poi_cache_path} …")
        pois_master = gpd.read_parquet(poi_cache_path)
    else:
        logger.info("No cached POI file found – fetching from Overpass API …")

        gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs != "EPSG:4326" else gdf
        north, south, east, west = gdf_wgs.total_bounds

        # ---- Flatten nested dict ----
        osmnx_tags: Dict[str, List[str]] = {}
        for tag_map in poi_categories.values():
            for key, values in tag_map.items():
                osmnx_tags.setdefault(key, []).extend(values)
        # dedupe values lists
        for k, vals in osmnx_tags.items():
            osmnx_tags[k] = list(set(vals))

        try:
            pois_master = ox.features_from_bbox(bbox=(north, south, east, west), tags=osmnx_tags)
            logger.info(f"Successfully downloaded {len(pois_master):,} POIs from Overpass.")

            # Assign category column
            pois_master['category'] = 'other'
            for category, tag_map in poi_categories.items():
                for key, values in tag_map.items():
                    if key in pois_master.columns:
                        mask = pois_master[key].isin(values)
                        pois_master.loc[mask, 'category'] = category

            logger.info(f"Saving POIs to cache → {poi_cache_path}")
            pois_master.to_parquet(poi_cache_path)
        except Exception as e:
            logger.error(f"POI download failed: {e}")
            logger.warning("Proceeding with zero-filled POI features.")
            pois_master = gpd.GeoDataFrame()

    # ---------------- Guard: empty POI table ----------------
    if pois_master.empty:
        logger.warning("POI table empty – all poi_* features will be zeros.")
        for radius in radii:
            for cat in poi_categories.keys():
                gdf[f'poi_{cat}_count_{radius}m'] = 0
        return gdf

    pois_master = pois_master[pois_master.geometry.geom_type == 'Point'].copy()

    # ---------------- Spatial counting ----------------
    gdf_proj = gdf.to_crs("EPSG:25831")
    pois_proj = pois_master.to_crs(gdf_proj.crs)
    poi_sindex = pois_proj.sindex  # spatial index accelerates queries

    for radius in tqdm(radii, desc="Calculating POI counts by radius"):
        buffers = gpd.GeoDataFrame(geometry=gdf_proj.geometry.centroid.buffer(radius), crs=gdf_proj.crs)

        for category in poi_categories.keys():
            col_name = f'poi_{category}_count_{radius}m'

            subset = pois_proj[pois_proj['category'] == category]
            if subset.empty:
                gdf[col_name] = 0
                continue

            # limit candidate POIs via bounding box intersection first
            possible_idx = list(poi_sindex.intersection(buffers.total_bounds))
            possible_pois = subset.iloc[possible_idx]

            join = gpd.sjoin(buffers, possible_pois, how='left', predicate='intersects')
            counts = join.index.value_counts()
            gdf[col_name] = counts.reindex(gdf.index, fill_value=0)

    return gdf

def create_spatial_cluster_features(gdf: gpd.GeoDataFrame, eps: float = 100.0, min_samples: int = 3) -> pd.DataFrame:
    """Create spatial cluster features using DBSCAN on the centroid of each geometry."""
    logger.info("Creating spatial cluster features...")
    
    if 'geometry' not in gdf.columns or gdf['geometry'].is_empty.all():
        logger.warning("No valid geometries found to create spatial clusters.")
        return pd.DataFrame(index=gdf.index)

    gdf_proj = gdf.to_crs("EPSG:25831") # Project to a meter-based CRS for DBSCAN
    
    # Calculate centroids from the projected geometries
    centroids = gdf_proj.geometry.centroid
    
    # Extract x and y coordinates from the centroids
    coords = np.array(list(zip(centroids.x, centroids.y)))

    # Now run DBSCAN on the point coordinates
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree')
    clusters = db.fit_predict(coords)
    
    return pd.DataFrame(clusters, index=gdf.index, columns=['spatial_cluster'])

def create_gtfs_features(parking_gdf: gpd.GeoDataFrame, gtfs_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with distance (in meters) to the nearest GTFS stop."""

    logger.info("Creating GTFS features …")

    stops_path = gtfs_dir / "stops.txt"
    if not stops_path.exists():
        logger.warning("GTFS stops.txt not found – skipping GTFS features.")
        return pd.DataFrame(index=parking_gdf.index)

    # --- Load stops as GeoDataFrame in WGS84 ---
    stops_df = pd.read_csv(stops_path)
    if {"stop_lon", "stop_lat"} - set(stops_df.columns):
        logger.error("GTFS stops.txt is missing stop_lon / stop_lat columns – skipping.")
        return pd.DataFrame(index=parking_gdf.index)

    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs="EPSG:4326",
    )

    # --- Project both layers to a metric CRS (ETRS89 / UTM 31N) ---
    metric_crs = "EPSG:25831"

    parking_metric = (
        parking_gdf.to_crs(metric_crs)
        if parking_gdf.crs != metric_crs
        else parking_gdf.copy()
    )
    stops_metric = stops_gdf.to_crs(metric_crs)

    # --- Nearest neighbour via spatial join ---
    nearest = gpd.sjoin_nearest(
        parking_metric,
        stops_metric[["geometry"]],
        how="left",
        distance_col="_dist_m",
    )

    # Collapse duplicates (sjoin_nearest can return >1 row per left geometry)
    dist_series = (
        nearest.groupby(level=0)["_dist_m"].min().reindex(parking_metric.index)
    )

    gtfs_features = pd.DataFrame({"distance_to_nearest_stop": dist_series})
    return gtfs_features

# region: Functions Integrated from static_feature_engineering.py

def create_static_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Orchestrator for creating all static features (capacity, zone properties)."""
    if gdf.empty:
        logger.warning("Input GeoDataFrame for static features is empty.")
        return gdf

    gdf_out = gdf.copy()
    logger.info("Creating 'estimated_capacity' feature...")
    gdf_out['estimated_capacity'] = estimate_capacity(gdf_out)
    logger.info("Creating zone property features from TARIFA and HORARIO...")
    gdf_out = parse_zone_properties(gdf_out)
    logger.info("Static features created successfully.")
    return gdf_out

def estimate_capacity(gdf: gpd.GeoDataFrame, meters_per_spot: float = 5.0) -> pd.Series:
    """Estimate parking capacity based on segment geometry length."""
    if not isinstance(gdf, gpd.GeoDataFrame) or 'geometry' not in gdf.columns:
        raise ValueError("Input must be a GeoDataFrame with a 'geometry' column.")
    if gdf.crs is None:
        logger.warning("CRS not set. Assuming EPSG:4326.")
        gdf = gdf.set_crs(epsg=4326)
    
    gdf_utm = gdf.to_crs(epsg=32631)
    lengths = gdf_utm.geometry.length
    capacity = (lengths / meters_per_spot).apply(np.floor)
    return capacity.astype(int)

def parse_zone_properties(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fast, vectorised parsing of TARIFA & HORARIO string columns."""

    gdf_copy = gdf.copy()

    # ------------- TARIFA (vectorised str.extract) -------------
    if 'TARIFA' in gdf_copy.columns:
        gdf_copy['tarifa_rate'] = (
            pd.to_numeric(
                gdf_copy['TARIFA'].str.extract(r'(\d+(?:\.\d+)?)', expand=False),
                errors='coerce',
            )
            .fillna(0)
        )

    # ------------- HORARIO (vectorised regex & numpy ops) -------------
    if 'HORARIO' in gdf_copy.columns:
        horario_series = gdf_copy['HORARIO'].astype(str).str.upper()

        # 24-hour flag
        gdf_copy['horario_is_24h'] = horario_series.str.contains('24H', na=False).astype(int)

        hours_extracted = horario_series.str.extract(r'(\d{1,2}):?\d{0,2}\s*-\s*(\d{1,2}):?\d{0,2}')
        start_hours = pd.to_numeric(hours_extracted[0], errors='coerce').fillna(8)
        end_hours = pd.to_numeric(hours_extracted[1], errors='coerce').fillna(20)

        gdf_copy['horario_start_hour'] = np.where(gdf_copy['horario_is_24h'] == 1, 0, start_hours)
        gdf_copy['horario_end_hour'] = np.where(gdf_copy['horario_is_24h'] == 1, 24, end_hours)
        gdf_copy['horario_total_hours'] = gdf_copy['horario_end_hour'] - gdf_copy['horario_start_hour']

    return gdf_copy

# endregion

def determine_id_column(df: pd.DataFrame, preferred_candidates: List[str]) -> Optional[str]:
    """Determines a suitable unique ID column from the DataFrame."""
    for col in preferred_candidates:
        if col in df.columns and df[col].is_unique:
            logger.info(f"Using '{col}' as unique ID column.")
            return col
    logger.warning("No suitable unique ID column found.")
    return None


# --- Main Processing Function (Final Refined Version) ---

def build_all_features() -> Optional[gpd.GeoDataFrame]:
    """
    Main function to orchestrate the creation of all features from a single,
    pre-processed input file.
    """
    logger.info("🚀 Starting integrated feature building process...")

    # --- Load a SINGLE Pre-processed Base Data File ---
    try:
        # This script's responsibility starts with an already-processed file.
        base_data_path = PARKING_DATA_PATH 
        logger.info(f"Loading single base data source: {base_data_path}")
        
        # Use gpd.read_parquet directly for GeoParquet files
        try:
            master_gdf = gpd.read_parquet(base_data_path)
        except Exception as e:
            logger.warning(f"Could not read as GeoParquet ({e}), attempting to read with pandas and convert.")
            df = pd.read_parquet(base_data_path)
            if 'geometry' in df.columns and isinstance(df['geometry'].iloc[0], dict):
                # Handle nested geometry dicts from raw JSON conversion
                def to_linestring(geom: Any) -> Optional[LineString]:
                    if not isinstance(geom, dict):
                        return None
                    coords = geom.get('coordinates')
                    if coords is None or len(coords) == 0:
                        return None
                    # Handle both LineString and MultiLineString (taking the first line)
                    if geom.get('type') == 'MultiLineString':
                        return LineString(coords[0])
                    return LineString(coords)

                df['geometry'] = df['geometry'].apply(to_linestring)
            
            master_gdf = gpd.GeoDataFrame(df, geometry='geometry')

        if master_gdf is None:
            raise FileNotFoundError(f"Base data file not found at {base_data_path}. Ensure a preceding script has run.")
        
        # Validate the loaded data
        if 'geometry' not in master_gdf.columns:
             raise ValueError("Base data must contain a 'geometry' column.")
        if master_gdf.crs is None:
            logger.warning("CRS not set on base data. Assuming EPSG:4326.")
            master_gdf.set_crs("EPSG:4326", inplace=True)
            
        logger.info(f"Successfully loaded base data. Shape: {master_gdf.shape}")

        # Ensure occupancy_level exists for downstream lag features
        if 'occupancy_level' not in master_gdf.columns:
            if 'prediction_code' in master_gdf.columns:
                logger.info("Creating 'occupancy_level' from 'prediction_code' column …")
                master_gdf['occupancy_level'] = master_gdf['prediction_code']
            else:
                logger.warning("Neither 'occupancy_level' nor 'prediction_code' present – lag features will be skipped.")

    except Exception as e:
        logger.error(f"Failed to load or prepare base data: {e}", exc_info=True)
        return None

    # --- Sequentially Create and Join All New Features ---

    # 1. Static Features (Capacity, Zone Properties)
    logger.info("--- Generating and Joining Static Features ---")
    # This function should add columns. Let's ensure it returns the full gdf.
    master_gdf = create_static_features(master_gdf)

    # 2. Temporal Features
    logger.info("--- Generating and Joining Temporal Features ---")
    temporal_features = create_temporal_features(master_gdf, 'timestamp')
    master_gdf = master_gdf.join(temporal_features)
    
    # 2b. Temporal Lag & Rolling
    lag_features = create_temporal_lag_features(master_gdf)
    master_gdf = master_gdf.join(lag_features)

    # 3. External Features (Weather, Events)
    logger.info("--- Generating and Joining External Features ---")
    weather_df = load_data(WEATHER_DATA_PATH, "csv")
    if weather_df is not None:
        weather_features = create_weather_features(weather_df, master_gdf, 'timestamp')
        master_gdf = master_gdf.join(weather_features)

    events_df = load_data(EVENTS_DATA_PATH, "csv")
    if events_df is not None and isinstance(events_df, pd.DataFrame):
        event_features = create_event_features(events_df, master_gdf['timestamp'])
        master_gdf = master_gdf.join(event_features)

    # 4. POI Features
    master_gdf = create_poi_features(master_gdf)

    # 5. Spatial Features (Clusters, GTFS)
    logger.info("--- Generating and Joining Spatial Features ---")
    spatial_cluster_features = create_spatial_cluster_features(master_gdf)
    master_gdf = master_gdf.join(spatial_cluster_features)

    gtfs_features = create_gtfs_features(master_gdf, GTFS_DIR)
    master_gdf = master_gdf.join(gtfs_features)

    # --- Finalize and Save ---
    logger.info("--- Finalizing and saving master table ---")
    final_df = master_gdf.loc[:,~master_gdf.columns.duplicated()]
    
    logger.info(f"Saving final combined feature table to {MASTER_TABLE_PATH}")
    save_data(final_df, MASTER_TABLE_PATH)
    
    logger.info("✅ Feature building process completed successfully.")
    return final_df

if __name__ == "__main__":
    master_features_df = build_all_features()
    if master_features_df is not None:
        logger.info(f"Master feature table generated successfully. Shape: {master_features_df.shape}")
    else:
        logger.error("Master feature table generation failed.")
