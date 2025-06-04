import time
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pyrosm import OSM
from shapely.geometry import box
from shapely.ops import unary_union
import gc
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from log import logger
import psutil
import signal
import sys
import warnings
import pickle
from shapely.geometry.base import BaseGeometry
from shapely import wkb, wkt
import dask.dataframe as dd
import glob

# Constants
BATCH_SIZE = 2000000  # 2M rows per batch for full run
TEMP_BATCH_DIR = "data/cache/temp_poi_batches"
MAX_RETRIES = 3
MEMORY_THRESHOLD = 0.9  # 90% memory usage threshold
POI_SJOIN_CHUNK_SIZE = 50000  # For internal sjoin processing within a batch

# Test run configuration
TEST_RUN_ROWS = None  # Set to None for full run
TEST_RUN_BATCH_SIZE = 50000  # Only used when TEST_RUN_ROWS is set

# File paths
BASE_FEATURES_INPUT_PATH = "data/interim/df_with_base_features.parquet"
PARKING_SEGMENTS_PATH = "data/processed/trams_geometries.gpkg"
OSM_PBF_PATH = "data/raw/cataluna-latest.osm.pbf"
FINAL_OUTPUT_PATH = "data/processed/features_master_table_historical_FULL.parquet"
PREPROCESSED_POIS_PICKLE_PATH = "data/cache/preprocessed_pois.pkl"

# Column names
ID_COLUMN_HISTORY = 'ID_TRAMO'  # ID column in base_features_input
ID_COLUMN_SEGMENTS = 'ID_TRAM'  # ID column in parking_segments
TIMESTAMP_COLUMN = 'timestamp'

# CRS and radii
TARGET_CRS = "EPSG:25831"
POI_RADII = [100, 200, 500]

# POI Categories
POI_CATEGORIES = {
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

class POIProcessingError(Exception):
    """Custom exception for POI processing errors"""
    pass

class MemoryError(Exception):
    """Custom exception for memory-related errors"""
    pass

def get_memory_usage() -> float:
    """Get current memory usage as a percentage"""
    return psutil.Process().memory_percent() / 100.0

def check_memory_usage() -> bool:
    """Check if memory usage is below threshold"""
    return get_memory_usage() < MEMORY_THRESHOLD

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()

def log_memory_usage():
    """Log current memory usage"""
    memory_percent = get_memory_usage() * 100
    logger.info(f"Current memory usage: {memory_percent:.1f}%")

def ensure_temp_dir():
    """Ensure temporary directory exists"""
    Path(TEMP_BATCH_DIR).mkdir(parents=True, exist_ok=True)

def get_existing_batches() -> List[int]:
    """Get list of existing batch numbers"""
    existing_batches = []
    for file in os.listdir(TEMP_BATCH_DIR):
        if file.startswith("features_batch_") and file.endswith(".parquet"):
            try:
                batch_num = int(file.split("_")[-1].split(".")[0])
                existing_batches.append(batch_num)
            except ValueError:
                logger.warning(f"Invalid batch file name: {file}")
    return sorted(existing_batches)

def preprocess_all_pois(osm_pbf_path: str, poi_categories: Dict[str, Any], target_crs: str, pickle_path: Optional[str] = None, force_recompute: bool = False) -> Dict[str, Optional[gpd.GeoDataFrame]]:
    """Extracts, projects, and cleans all POIs from the PBF for all defined categories."""
    logger.info(f"Starting POI preprocessing. Force recompute: {force_recompute}")
    
    if not force_recompute and pickle_path and os.path.exists(pickle_path):
        logger.info(f"Attempting to load preprocessed POIs from: {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                processed_pois = pickle.load(f)
            logger.info(f"Successfully loaded preprocessed POIs for {len(processed_pois)} categories from pickle.")
            return processed_pois
        except Exception as e:
            logger.error(f"Error loading POIs from pickle {pickle_path}: {e}. Recomputing.")

    try:
        osm = OSM(osm_pbf_path)
        logger.info("OSM reader initialized.")
    except Exception as e:
        logger.error(f"Error initializing OSM reader for {osm_pbf_path}: {e}", exc_info=True)
        raise POIProcessingError(f"Failed to initialize OSM reader: {e}")

    processed_pois_by_category = {}
    logger.info("--- Starting Upfront POI Extraction, Projection, Cleaning ---")
    
    for category, filter_dict in poi_categories.items():
        logger.info(f"Preprocessing POIs for category: {category} with filter: {filter_dict}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                category_gdf_all = osm.get_pois(custom_filter=filter_dict)

            if category_gdf_all is None or category_gdf_all.empty:
                logger.info(f"No POIs found for category '{category}' in the entire PBF. Storing None.")
                processed_pois_by_category[category] = None
                continue

            # Project and clean geometries
            if category_gdf_all.crs is None:
                category_gdf_all = category_gdf_all.set_crs("EPSG:4326", allow_override=True)
            
            if category_gdf_all.crs.to_string().lower() != target_crs.lower():
                category_gdf_all = category_gdf_all.to_crs(target_crs)

            # Clean geometries
            non_point_mask = category_gdf_all.geometry.geom_type != 'Point'
            if non_point_mask.any():
                category_gdf_all.loc[non_point_mask, 'geometry'] = category_gdf_all.loc[non_point_mask, 'geometry'].buffer(0)
            
            category_gdf_all = category_gdf_all[~category_gdf_all.geometry.is_empty & category_gdf_all.geometry.is_valid]
            
            if category_gdf_all.empty:
                processed_pois_by_category[category] = None
            else:
                # Optimize memory by selecting only essential columns
                processed_pois_by_category[category] = category_gdf_all[['geometry']].copy()
                logger.info(f"Successfully preprocessed {len(category_gdf_all)} POIs for '{category}'")

        except Exception as e:
            logger.error(f"Error preprocessing category {category}: {e}", exc_info=True)
            processed_pois_by_category[category] = None

    if pickle_path:
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(processed_pois_by_category, f)
            logger.info("Successfully saved preprocessed POIs to pickle.")
        except Exception as e:
            logger.error(f"Error saving POIs to pickle {pickle_path}: {e}")

    return processed_pois_by_category

def get_expected_poi_columns() -> List[str]:
    """Get the complete list of expected POI feature columns."""
    expected_columns = []
    for category in POI_CATEGORIES.keys():
        for radius in POI_RADII:
            # Raw count columns
            expected_columns.append(f'poi_{category}_count_{radius}m')
            # Presence flag columns
            expected_columns.append(f'poi_{category}_present_{radius}m')
            # Log-transformed count columns
            expected_columns.append(f'poi_{category}_log1p_count_{radius}m')
    return expected_columns

def ensure_poi_columns(df: pd.DataFrame, batch_num: int) -> pd.DataFrame:
    """Ensure all expected POI columns exist with correct dtypes."""
    expected_columns = get_expected_poi_columns()
    
    for col in expected_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} missing from batch {batch_num}. Adding with defaults.")
            if '_present' in col:
                df[col] = False
            elif '_log1p_count_' in col:
                df[col] = 0.0  # Float for log-transformed counts
            else:  # Raw count columns
                df[col] = 0
    
    # Ensure consistent dtypes
    for col in df.columns:
        if '_present' in col:
            df[col] = df[col].astype(bool)
        elif '_log1p_count_' in col:
            df[col] = df[col].astype(float)
        elif '_count_' in col:
            df[col] = df[col].astype(int)
    
    return df

def process_batch(df_batch: pd.DataFrame, processed_pois_by_category: Dict[str, Optional[gpd.GeoDataFrame]], 
                 gdf_segments_buffered: gpd.GeoDataFrame, batch_num: int, target_crs: str) -> bool:
    """Process a single batch with retry logic and memory management"""
    batch_file = os.path.join(TEMP_BATCH_DIR, f"features_batch_{batch_num:03d}.parquet")
    
    for attempt in range(MAX_RETRIES):
        try:
            if not check_memory_usage():
                raise MemoryError("Memory usage above threshold")

            # Initialize all POI columns with default values
            df_batch = ensure_poi_columns(df_batch, batch_num)

            # Process each POI category
            for category, _ in POI_CATEGORIES.items():
                gdf_pois_cat = processed_pois_by_category.get(category)
                if gdf_pois_cat is None or gdf_pois_cat.empty:
                    continue

                # Clip POIs to batch extent
                batch_bounds = df_batch.geometry.total_bounds
                clip_box = box(*batch_bounds)
                clip_gdf = gpd.GeoDataFrame([{'geometry': clip_box}], crs=target_crs)
                
                try:
                    gdf_pois_clipped = gpd.clip(gdf_pois_cat, clip_gdf)
                except Exception as e:
                    logger.error(f"Error clipping POIs for category {category}: {e}")
                    continue

                if gdf_pois_clipped.empty:
                    continue

                # Process each radius
                for radius in POI_RADII:
                    try:
                        # Spatial join with retry logic
                        for sjoin_attempt in range(MAX_RETRIES):
                            try:
                                sjoin_result = gpd.sjoin(
                                    df_batch,
                                    gdf_pois_clipped,
                                    how='left',
                                    predicate='intersects'
                                )
                                break
                            except Exception as e:
                                if sjoin_attempt == MAX_RETRIES - 1:
                                    raise
                                logger.warning(f"Sjoin attempt {sjoin_attempt + 1} failed: {e}")
                                time.sleep(2 ** sjoin_attempt)

                        # Aggregate counts
                        counts = sjoin_result.groupby(sjoin_result.index).size()
                        count_col = f'poi_{category}_count_{radius}m'
                        present_col = f'poi_{category}_present_{radius}m'
                        log1p_col = f'poi_{category}_log1p_count_{radius}m'
                        
                        # Update count and presence columns
                        df_batch[count_col] = df_batch.index.map(counts).fillna(0).astype(int)
                        df_batch[present_col] = df_batch[count_col] > 0
                        
                        # Calculate log1p transformation
                        df_batch[log1p_col] = np.log1p(df_batch[count_col]).astype(float)

                    except Exception as e:
                        logger.error(f"Error processing radius {radius}m for category {category}: {e}")
                        continue

            # Ensure all columns exist and have correct dtypes before saving
            df_batch = ensure_poi_columns(df_batch, batch_num)
            
            # Handle geometry column conversion to WKT
            if 'geometry' in df_batch.columns:
                if isinstance(df_batch['geometry'], gpd.GeoSeries):
                    logger.info(f"Converting 'geometry' GeoSeries to WKT for batch {batch_num}...")
                    # Convert valid geometries to WKT, handle invalid ones gracefully
                    df_batch['geometry_wkt'] = df_batch['geometry'].apply(
                        lambda geom: geom.wkt if hasattr(geom, 'wkt') and geom is not None and not geom.is_empty else None
                    )
                    # Drop original geometry column and rename WKT column
                    df_batch = df_batch.drop(columns=['geometry'])
                    df_batch = df_batch.rename(columns={'geometry_wkt': 'geometry'})
                else:
                    logger.warning(
                        f"Batch {batch_num}: 'geometry' column is present but not a GeoSeries. "
                        f"Type: {type(df_batch['geometry'])}. "
                        f"Attempting to handle as existing WKT or convert if needed."
                    )
                    # Handle non-GeoSeries geometry column
                    def to_wkt_safe(geom):
                        if geom is None:
                            return None
                        if isinstance(geom, str):
                            return geom  # Already WKT
                        if hasattr(geom, 'wkt'):
                            return geom.wkt
                        if isinstance(geom, dict) and 'type' in geom and 'coordinates' in geom:
                            try:
                                from shapely.geometry import shape
                                return shape(geom).wkt
                            except Exception as e:
                                logger.warning(f"Error converting dict geometry to WKT: {e}")
                                return None
                        return None
                    
                    df_batch['geometry'] = df_batch['geometry'].apply(to_wkt_safe)
            
            # Save batch with error handling
            df_batch.to_parquet(batch_file, index=False)
            return True

        except MemoryError as me:
            logger.error(f"Memory error in batch {batch_num}: {me}")
            cleanup_memory()
            if attempt == MAX_RETRIES - 1:
                return False
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Error processing batch {batch_num} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                return False
            time.sleep(2 ** attempt)
    
    return False

def signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    logger.info("Received interrupt signal. Saving progress...")
    sys.exit(0)

def load_and_prepare_segment_buffers(parking_segments_path: str, id_column: str, radii: List[int], target_crs: str) -> Optional[gpd.GeoDataFrame]:
    """Load parking segments and prepare buffers for POI analysis."""
    try:
        logger.info(f"Loading parking segments from {parking_segments_path}")
        gdf_segments = gpd.read_file(parking_segments_path)
        
        if gdf_segments.empty:
            logger.error("Loaded parking segments GeoDataFrame is empty")
            return None
            
        # Ensure proper CRS
        if gdf_segments.crs is None:
            logger.warning("No CRS found in parking segments. Assuming EPSG:4326")
            gdf_segments = gdf_segments.set_crs("EPSG:4326", allow_override=True)
            
        if gdf_segments.crs.to_string().lower() != target_crs.lower():
            logger.info(f"Projecting parking segments from {gdf_segments.crs} to {target_crs}")
            gdf_segments = gdf_segments.to_crs(target_crs)
            
        # Clean geometries
        logger.info("Cleaning parking segment geometries...")
        invalid_mask = ~gdf_segments.geometry.is_valid
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} invalid geometries. Attempting to fix...")
            gdf_segments.loc[invalid_mask, 'geometry'] = gdf_segments.loc[invalid_mask, 'geometry'].buffer(0)
            
        # Remove empty geometries
        empty_mask = gdf_segments.geometry.is_empty
        if empty_mask.any():
            logger.warning(f"Removing {empty_mask.sum()} empty geometries")
            gdf_segments = gdf_segments[~empty_mask]
            
        if gdf_segments.empty:
            logger.error("No valid geometries remaining after cleaning")
            return None
            
        # Create buffers for each radius
        logger.info("Creating buffers for each radius...")
        for radius in radii:
            buffer_col = f'buffer_{radius}m'
            gdf_segments[buffer_col] = gdf_segments.geometry.buffer(radius)
            
        # Ensure ID column exists
        if id_column not in gdf_segments.columns:
            logger.error(f"Required ID column '{id_column}' not found in parking segments")
            return None
            
        logger.info(f"Successfully prepared {len(gdf_segments)} parking segments with buffers")
        return gdf_segments
        
    except Exception as e:
        logger.error(f"Error loading and preparing segment buffers: {e}", exc_info=True)
        return None

def validate_batch_files(batch_files: List[str]) -> Tuple[bool, List[str], List[str]]:
    """Validate batch files and return expected and actual columns."""
    try:
        # Read first batch to get expected columns
        first_batch = pd.read_parquet(batch_files[0])
        expected_columns = set(first_batch.columns)
        
        # Check a few random batches
        sample_size = min(5, len(batch_files))
        sample_batches = np.random.choice(batch_files, sample_size, replace=False)
        
        all_columns = set()
        for batch_file in sample_batches:
            df = pd.read_parquet(batch_file)
            all_columns.update(df.columns)
        
        # Find missing and extra columns
        missing_columns = expected_columns - all_columns
        extra_columns = all_columns - expected_columns
        
        if missing_columns or extra_columns:
            logger.warning(f"Column mismatch detected:")
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            if extra_columns:
                logger.warning(f"Extra columns: {extra_columns}")
            return False, list(missing_columns), list(extra_columns)
        
        return True, [], []
        
    except Exception as e:
        logger.error(f"Error validating batch files: {e}")
        return False, [], []

def combine_batches_with_dask(batch_files: List[str], final_output_path: str, temp_dir: str) -> bool:
    """Combine batch files using Dask for memory-efficient concatenation."""
    try:
        logger.info(f"Starting Dask-based batch combination for {len(batch_files)} files")
        
        # Validate batch files first
        is_valid, missing_cols, extra_cols = validate_batch_files(batch_files)
        if not is_valid:
            logger.error("Batch validation failed. Please check the column mismatches above.")
            return False
        
        # Remove existing output file if it exists
        if os.path.exists(final_output_path):
            logger.info(f"Removing existing output file: {final_output_path}")
            os.remove(final_output_path)
        
        # Read all Parquet files into a Dask DataFrame
        logger.info("Loading batch files into Dask DataFrame...")
        ddf = dd.read_parquet(batch_files)
        
        # Convert geometry column to WKT format if needed
        logger.info("Converting geometry column to WKT format...")
        def ensure_wkt(geom):
            if geom is None:
                return None
            if isinstance(geom, str):
                return geom
            if hasattr(geom, 'wkt'):
                return geom.wkt
            if isinstance(geom, dict) and 'type' in geom and 'coordinates' in geom:
                try:
                    from shapely.geometry import shape
                    return shape(geom).wkt
                except Exception as e:
                    logger.warning(f"Error converting dict geometry to WKT: {e}")
                    return None
            return None
        
        ddf['geometry'] = ddf['geometry'].map_partitions(lambda s: s.apply(ensure_wkt))
        
        # Log initial shape
        logger.info("Computing initial shape...")
        num_rows, num_cols = ddf.shape[0].compute(), len(ddf.columns)
        logger.info(f"Total rows: {num_rows}, Total columns: {num_cols}")
        
        # Calculate optimal number of partitions
        # Aim for partitions of ~128MB
        partition_size = 128 * 1024**2  # 128MB in bytes
        estimated_size = ddf.memory_usage(deep=True).sum().compute()
        npartitions = max(1, int(np.ceil(estimated_size / partition_size)))
        logger.info(f"Repartitioning to {npartitions} partitions for optimal memory usage")
        
        # Repartition for better memory management
        ddf = ddf.repartition(npartitions=npartitions)
        
        # Define schema for geometry column
        schema = {
            'geometry': 'string'  # Specify geometry as string type
        }
        
        # Save to final output
        logger.info(f"Saving final output to {final_output_path}")
        ddf.to_parquet(
            final_output_path,
            write_index=False,
            engine='pyarrow',
            schema=schema
        )
        
        logger.info("Successfully combined all batches using Dask")
        return True
        
    except Exception as e:
        logger.error(f"Error during Dask-based batch combination: {e}", exc_info=True)
        return False

def cleanup_temp_directory():
    """Clean up temporary batch files and directory."""
    try:
        if os.path.exists(TEMP_BATCH_DIR):
            logger.info(f"Cleaning up temporary directory: {TEMP_BATCH_DIR}")
            for file in os.listdir(TEMP_BATCH_DIR):
                file_path = os.path.join(TEMP_BATCH_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Error removing file {file_path}: {e}")
            try:
                os.rmdir(TEMP_BATCH_DIR)
            except Exception as e:
                logger.warning(f"Error removing directory {TEMP_BATCH_DIR}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_time_main = time.time()
    ensure_temp_dir()
    
    try:
        # Clean up any existing batch files for a fresh start
        cleanup_temp_directory()
        ensure_temp_dir()
        
        # Check if we're resuming from a specific batch
        resume_from_batch = int(os.environ.get("RESUME_FROM_BATCH", 0))
        
        # Load base features
        logger.info(f"Loading base features from {BASE_FEATURES_INPUT_PATH}")
        try:
            df_base_full = pd.read_parquet(BASE_FEATURES_INPUT_PATH)
            if df_base_full.empty:
                raise ValueError("Base features DataFrame is empty")
            logger.info(f"Loaded {len(df_base_full)} rows from base features")
        except Exception as e:
            logger.error(f"Error loading base features: {e}", exc_info=True)
            raise

        # Apply test run slicing if configured
        if TEST_RUN_ROWS and TEST_RUN_ROWS > 0 and TEST_RUN_ROWS < len(df_base_full):
            logger.warning(f"--- TEST RUN --- Slicing df_base_full to first {TEST_RUN_ROWS} rows")
            df_base_full = df_base_full.head(TEST_RUN_ROWS).copy()
            logger.info(f"Test data shape: {df_base_full.shape}")
            
            # Use larger batch size for test run
            global BATCH_SIZE
            BATCH_SIZE = TEST_RUN_BATCH_SIZE
            logger.info(f"Using test run batch size: {BATCH_SIZE}")

        # Load parking segments to get geometry
        logger.info(f"Loading parking segments from {PARKING_SEGMENTS_PATH}")
        try:
            gdf_segments = gpd.read_file(PARKING_SEGMENTS_PATH)
            if gdf_segments.empty:
                raise ValueError("Parking segments GeoDataFrame is empty")
            
            # Ensure proper CRS
            if gdf_segments.crs is None:
                gdf_segments = gdf_segments.set_crs("EPSG:4326", allow_override=True)
            if gdf_segments.crs.to_string().lower() != TARGET_CRS.lower():
                gdf_segments = gdf_segments.to_crs(TARGET_CRS)
            
            logger.info(f"Loaded {len(gdf_segments)} parking segments")
        except Exception as e:
            logger.error(f"Error loading parking segments: {e}", exc_info=True)
            raise

        # Merge geometry with base features
        logger.info("Merging geometry from parking segments with base features")
        try:
            # Create a mapping of ID_TRAM to geometry
            geometry_map = gdf_segments.set_index(ID_COLUMN_SEGMENTS)['geometry']
            
            # Map geometries to base features
            df_base_full['geometry'] = df_base_full[ID_COLUMN_HISTORY].map(geometry_map)
            
            # Convert to GeoDataFrame
            df_base_full = gpd.GeoDataFrame(df_base_full, geometry='geometry', crs=TARGET_CRS)
            
            # Drop rows where geometry is None (if any)
            missing_geom = df_base_full['geometry'].isna()
            if missing_geom.any():
                logger.warning(f"Dropping {missing_geom.sum()} rows with missing geometry")
                df_base_full = df_base_full[~missing_geom]
            
            logger.info(f"Successfully merged geometry. Final DataFrame has {len(df_base_full)} rows")
        except Exception as e:
            logger.error(f"Error merging geometry: {e}", exc_info=True)
            raise
        
        # Preprocess POIs
        processed_pois_by_category = preprocess_all_pois(
            OSM_PBF_PATH, POI_CATEGORIES, TARGET_CRS,
            pickle_path=PREPROCESSED_POIS_PICKLE_PATH,
            force_recompute=False
        )
        
        if not processed_pois_by_category:
            raise POIProcessingError("POI preprocessing failed or returned empty results")
        
        # Load and prepare segment buffers
        gdf_segments_buffered = load_and_prepare_segment_buffers(
            PARKING_SEGMENTS_PATH, ID_COLUMN_SEGMENTS, POI_RADII, TARGET_CRS
        )
        
        if gdf_segments_buffered is None or gdf_segments_buffered.empty:
            raise POIProcessingError("Failed to load or prepare segment buffers")
        
        total_rows = len(df_base_full)
        num_batches = int(np.ceil(total_rows / BATCH_SIZE))
        logger.info(f"Total rows: {total_rows}. Batch size: {BATCH_SIZE}. Number of batches: {num_batches}")
        
        # Get existing batches
        existing_batches = get_existing_batches()
        if resume_from_batch > 0:
            logger.info(f"Resuming from batch {resume_from_batch}")
        
        # Process batches
        for batch_idx in range(resume_from_batch - 1 if resume_from_batch > 0 else 0, num_batches):
            current_batch_num = batch_idx + 1
            
            # Skip if already processed
            if current_batch_num in existing_batches:
                logger.info(f"Batch {current_batch_num} already exists, skipping...")
                continue
            
            # Check memory before processing
            if not check_memory_usage():
                logger.warning("High memory usage detected. Forcing cleanup...")
                cleanup_memory()
            
            batch_start_time = time.time()
            start_row = batch_idx * BATCH_SIZE
            end_row = min(start_row + BATCH_SIZE, total_rows)
            
            logger.info(f"--- Processing Batch {current_batch_num}/{num_batches} (rows {start_row}-{end_row}) ---")
            
            # Process batch with error handling
            df_batch = df_base_full.iloc[start_row:end_row].copy()
            if not process_batch(df_batch, processed_pois_by_category, gdf_segments_buffered, current_batch_num, TARGET_CRS):
                raise POIProcessingError(f"Failed to process batch {current_batch_num} after {MAX_RETRIES} attempts")
            
            # Cleanup after each batch
            del df_batch
            cleanup_memory()
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch {current_batch_num} completed in {batch_time:.2f} seconds")
        
        # After all batches are processed, combine them using Dask
        logger.info("Combining all processed batches using Dask...")
        try:
            # Get list of all batch files
            batch_files = [os.path.join(TEMP_BATCH_DIR, f"features_batch_{i:03d}.parquet") 
                          for i in range(1, num_batches + 1)]
            batch_files = [f for f in batch_files if os.path.exists(f)]
            
            if not batch_files:
                raise POIProcessingError("No batch files found to combine")
            
            logger.info(f"Found {len(batch_files)} batch files to combine")
            
            # Combine batches using Dask
            if not combine_batches_with_dask(batch_files, FINAL_OUTPUT_PATH, TEMP_BATCH_DIR):
                raise POIProcessingError("Failed to combine batches using Dask")
            
            # Cleanup temporary files
            logger.info("Cleaning up temporary files...")
            for batch_file in batch_files:
                try:
                    os.remove(batch_file)
                except Exception as e:
                    logger.warning(f"Error removing temporary file {batch_file}: {e}")
            
            try:
                os.rmdir(TEMP_BATCH_DIR)
            except Exception as e:
                logger.warning(f"Error removing temporary directory {TEMP_BATCH_DIR}: {e}")
            
            total_time = time.time() - start_time_main
            logger.info(f"All batches processed and combined successfully in {total_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Fatal error in batch processing: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Fatal error in batch processing: {str(e)}")
        raise
    finally:
        cleanup_memory()

if __name__ == "__main__":
    main() 