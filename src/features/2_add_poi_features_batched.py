import pandas as pd
import numpy as np
import geopandas as gpd
from pyrosm import OSM
from shapely.geometry import box # Required for creating BBOX polygon for clipping
from shapely.ops import unary_union # Required for calculating total bounds of buffers
import os
import logging
import time
import warnings
import pickle # For saving/loading preprocessed POIs
import gc # For garbage collection if memory issues arise

# --- Configuration ---
BASE_FEATURES_INPUT_PATH = "data/interim/df_with_base_features.parquet"
PARKING_SEGMENTS_PATH = "data/processed/trams_geometries.gpkg"
OSM_PBF_PATH = "data/raw/cataluna-latest.osm.pbf"
FINAL_OUTPUT_PATH = "data/processed/features_master_table_historical_FULL.parquet"
TEMP_BATCH_DIR = "data/cache/temp_poi_batches"
PREPROCESSED_POIS_PICKLE_PATH = "data/cache/preprocessed_pois.pkl"

ID_COLUMN_HISTORY = 'ID_TRAMO'  # ID column in base_features_input (df_batch_input)
ID_COLUMN_SEGMENTS = 'ID_TRAM' # ID column in parking_segments (gdf_segments_buffered)
TIMESTAMP_COLUMN = 'timestamp' # Needed if any time-based operations were part of POI logic (not typical for counts)

TARGET_CRS = "EPSG:25831"
POI_RADII = [100, 200, 500]

# Batch processing parameters - EXPERIMENT with BATCH_SIZE
BATCH_SIZE = 1_000_000 # e.g., 1M, 2M, 5M rows from df_with_base_features

# POI_SJOIN_CHUNK_SIZE for internal sjoin processing within a batch (from build_features_historical.py)
POI_SJOIN_CHUNK_SIZE = 50000

# POI categories - Should be consistent with any previous definitions if models depend on exact names
POI_CATEGORIES = {
    'sustenance': {'amenity': ['restaurant', 'cafe', 'fast_food', 'pub', 'bar']},
    'shop': {'shop': True}, 
    'education': {'amenity': ['school', 'university', 'college', 'kindergarten']},
    'health': {'amenity': ['hospital', 'clinic', 'doctors', 'pharmacy']},
    'transport': {'public_transport': ['stop_position', 'platform', 'station'], 'highway': ['bus_stop']},
    'leisure': {'leisure': True},
    'tourism': {'tourism': ['hotel', 'hostel', 'guest_house', 'motel', 'attraction']},
    'parking_poi': {'amenity': ['parking', 'parking_entrance', 'parking_space']}, # Renamed to avoid clash with parking_segments
    'finance': {'amenity': ['bank', 'atm']},
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger("2_add_poi_features_batched")

# --- Helper Functions ---

def check_rtree():
    """Checks for rtree library and logs availability."""
    try:
        import rtree
        logger.info(f"rtree library found. Spatial indexing for sjoin will be enabled. Version: {rtree.__version__}")
        return True
    except ImportError:
        logger.warning("rtree library not found. pip install rtree for significantly faster spatial joins. sjoin will still work but will be slower.")
        return False

def preprocess_all_pois(osm_pbf_path, poi_categories, target_crs, pickle_path=None, force_recompute=False):
    """
    Extracts, projects, and cleans all POIs from the PBF for all defined categories.
    Optionally saves/loads the processed POIs dictionary using pickle.
    """
    logger.info(f"Starting POI preprocessing. Force recompute: {force_recompute}")
    check_rtree() # Log rtree status

    if not force_recompute and pickle_path and os.path.exists(pickle_path):
        logger.info(f"Attempting to load preprocessed POIs from: {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                processed_pois = pickle.load(f)
            logger.info(f"Successfully loaded preprocessed POIs for {len(processed_pois)} categories from pickle.")
            return processed_pois
        except Exception as e:
            logger.error(f"Error loading POIs from pickle {pickle_path}: {e}. Recomputing.")

    logger.info(f"Initializing OSM data reader for: {osm_pbf_path}")
    try:
        osm = OSM(osm_pbf_path)
        logger.info("OSM reader initialized.")
    except Exception as e:
        logger.error(f"Error initializing OSM reader for {osm_pbf_path}: {e}", exc_info=True)
        return None

    processed_pois_by_category = {}
    logger.info("--- Starting Upfront POI Extraction, Projection, Cleaning ---")
    for category, filter_dict in poi_categories.items():
        logger.info(f"Preprocessing POIs for category: {category} with filter: {filter_dict}")
        try:
            with warnings.catch_warnings(): # Suppress pyrosm geometry warnings during extraction
                warnings.simplefilter("ignore", UserWarning)
                category_gdf_all = osm.get_pois(custom_filter=filter_dict)

            if category_gdf_all is None or category_gdf_all.empty:
                logger.info(f"No POIs found for category '{category}' in the entire PBF. Storing None.")
                processed_pois_by_category[category] = None
                continue
            logger.info(f"Extracted {len(category_gdf_all)} raw POIs for '{category}'. Columns: {category_gdf_all.columns.tolist()}")

            # Ensure 'geometry' column exists and is active
            if 'geometry' not in category_gdf_all.columns:
                logger.error(f"No 'geometry' column in raw POIs for '{category}'. Skipping.")
                processed_pois_by_category[category] = None
                continue
            if category_gdf_all.geometry.name != 'geometry':
                 category_gdf_all = category_gdf_all.set_geometry('geometry')

            # Project to target_crs
            if category_gdf_all.crs is None:
                logger.warning(f"Raw POIs for '{category}' have no CRS. Assuming EPSG:4326.")
                category_gdf_all = category_gdf_all.set_crs("EPSG:4326", allow_override=True)
            
            if category_gdf_all.crs.to_string().lower() != target_crs.lower():
                logger.info(f"Projecting raw POIs for {category} from {category_gdf_all.crs} to {target_crs}.")
                category_gdf_all = category_gdf_all.to_crs(target_crs)
            
            # --- MODIFIED CLEANING LOGIC ---
            logger.info(f"--- Pre-cleaning Analysis for '{category}' ({len(category_gdf_all)} POIs) ---")
            if not category_gdf_all.empty:
                logger.info(f"Geometry types before cleaning for '{category}':\n{category_gdf_all.geom_type.value_counts(dropna=False)}")
                logger.info(f"Validity before cleaning for '{category}':\n{category_gdf_all.is_valid.value_counts(dropna=False)}")
            else:
                logger.info(f"No geometries to analyze for '{category}' (already empty).")

            logger.info(f"Cleaning geometries for '{category}' POIs ({len(category_gdf_all)} initially)...")
            
            projected_gdf = category_gdf_all # Use projected_gdf as the working variable for clarity

            # Separate points from other geometries for cleaning strategy
            points_gdf = projected_gdf[projected_gdf.geometry.geom_type == 'Point'].copy()
            other_geoms_gdf = projected_gdf[~projected_gdf.geometry.index.isin(points_gdf.index)].copy()

            cleaned_other_geoms_list = []
            if not other_geoms_gdf.empty:
                original_crs = other_geoms_gdf.crs
                logger.info(f"Applying buffer(0) to {len(other_geoms_gdf)} non-Point geometries for '{category}'.")
                # Apply buffer(0) only to non-point geometries that might be invalid
                other_geoms_gdf['geometry'] = other_geoms_gdf.geometry.buffer(0)
                # Filter out any that became empty or invalid
                other_geoms_gdf = other_geoms_gdf[~other_geoms_gdf.geometry.is_empty & other_geoms_gdf.geometry.is_valid]
                if other_geoms_gdf.crs is None and original_crs is not None: # Reassign CRS if lost
                    logger.info(f"Reassigning CRS ({original_crs}) to cleaned non-Point GDF for '{category}'.")
                    other_geoms_gdf = other_geoms_gdf.set_crs(original_crs)
                if not other_geoms_gdf.empty:
                    cleaned_other_geoms_list.append(other_geoms_gdf)
                    logger.info(f"{len(other_geoms_gdf)} non-Point geometries remain after cleaning for '{category}'.")
                else:
                    logger.info(f"No non-Point geometries remain after buffer(0) and filtering for '{category}'.")

            # Combine cleaned non-points with original points
            final_cleaned_list = []
            if not points_gdf.empty:
                final_cleaned_list.append(points_gdf)
                logger.info(f"Retaining {len(points_gdf)} Point geometries as-is for '{category}'.")
            
            if cleaned_other_geoms_list: # If there were non-empty other geoms after cleaning
                 final_cleaned_list.extend(cleaned_other_geoms_list)

            if final_cleaned_list:
                cleaned_gdf = pd.concat(final_cleaned_list)
                # Ensure consistent columns with original, in case concat changes them
                cleaned_gdf = cleaned_gdf.reindex(columns=projected_gdf.columns) 
            elif not projected_gdf.empty: # If projected_gdf was not empty but final_cleaned_list is
                logger.warning(f"All {len(projected_gdf)} POIs for '{category}' became empty/invalid after cleaning or were filtered out.")
                cleaned_gdf = gpd.GeoDataFrame(columns=projected_gdf.columns, geometry=[], crs=projected_gdf.crs)
            else: # projected_gdf was already empty
                cleaned_gdf = projected_gdf # an empty GDF
            
            rows_dropped_cat = len(projected_gdf) - len(cleaned_gdf)
            if rows_dropped_cat > 0:
                 logger.warning(f"Dropped {rows_dropped_cat} POIs for '{category}' during cleaning process. Initial: {len(projected_gdf)}, Final: {len(cleaned_gdf)}")
            
            # --- END MODIFIED CLEANING LOGIC ---
            
            if cleaned_gdf.empty:
                logger.info(f"No valid POIs remaining for category '{category}' after revised cleaning. Storing None.")
                processed_pois_by_category[category] = None
            else:
                # Select only essential columns to save memory
                processed_pois_by_category[category] = cleaned_gdf.loc[:, cleaned_gdf.notna().any(axis=0)]
                logger.info(f"Successfully preprocessed {len(processed_pois_by_category[category])} POIs for '{category}' using revised cleaning.")

        except ImportError as ie:
            logger.error(f"ImportError preprocessing category {category}: {ie}. Wildcard filters might require additional libraries.", exc_info=True)
            processed_pois_by_category[category] = None
        except Exception as e:
            logger.error(f"Error preprocessing category {category}: {e}", exc_info=True)
            processed_pois_by_category[category] = None
    logger.info("--- Finished Upfront POI Preprocessing ---")

    if pickle_path:
        logger.info(f"Attempting to save preprocessed POIs to: {pickle_path}")
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(processed_pois_by_category, f)
            logger.info("Successfully saved preprocessed POIs to pickle.")
        except Exception as e:
            logger.error(f"Error saving POIs to pickle {pickle_path}: {e}")
            
    return processed_pois_by_category

def load_and_prepare_segment_buffers(segments_path, id_col_segments, radii, target_crs):
    """
    Loads parking segment geometries, projects, and pre-calculates buffers.
    Returns a GeoDataFrame indexed by id_col_segments with buffer geometry columns.
    """
    logger.info(f"Loading parking segments from: {segments_path}")
    if not os.path.exists(segments_path):
        logger.error(f"Parking segments file not found: {segments_path}.")
        return None
    try:
        gdf_segments_raw = gpd.read_file(segments_path)
        logger.info(f"Loaded segments: {gdf_segments_raw.shape[0]} rows. Columns: {gdf_segments_raw.columns.tolist()}")

        if id_col_segments not in gdf_segments_raw.columns:
            logger.error(f"Critical: ID column '{id_col_segments}' not found in {segments_path}. Check GeoPackage content.")
            return None
        
        try:
            gdf_segments_raw[id_col_segments] = pd.to_numeric(gdf_segments_raw[id_col_segments])
            logger.info(f"Converted segment ID column '{id_col_segments}' to numeric.")
        except Exception as e:
            logger.error(f"Could not convert segment ID column '{id_col_segments}' to numeric: {e}. This is critical.")
            return None

        if gdf_segments_raw.crs is None:
            logger.warning(f"CRS for {segments_path} is None. Assuming EPSG:4326.")
            gdf_segments_raw = gdf_segments_raw.set_crs("EPSG:4326", allow_override=True)
        elif gdf_segments_raw.crs.to_string().lower() != "epsg:4326":
             logger.warning(f"CRS for {segments_path} is {gdf_segments_raw.crs.to_string()}. Re-projecting to EPSG:4326 first.")
             gdf_segments_raw = gdf_segments_raw.to_crs("EPSG:4326")
        
        logger.info(f"Projecting segment geometries to target CRS: {target_crs}...")
        gdf_segments = gdf_segments_raw[[id_col_segments, 'geometry']].copy()
        gdf_segments = gdf_segments.to_crs(target_crs)

        initial_count = len(gdf_segments)
        gdf_segments = gdf_segments.drop_duplicates(subset=[id_col_segments], keep='first')
        if len(gdf_segments) < initial_count:
            logger.info(f"Deduplicated segments by '{id_col_segments}': {initial_count} -> {len(gdf_segments)}")

        logger.info(f"Pre-calculating buffers for radii: {radii}m...")
        for r in radii:
            buffer_col_name = f'buffer_{r}m'
            gdf_segments[buffer_col_name] = gdf_segments.geometry.buffer(r)
        
        buffer_cols = [f'buffer_{r}m' for r in radii]
        gdf_segments_buffered = gdf_segments[[id_col_segments] + buffer_cols].copy()
        
        logger.info(f"Setting '{id_col_segments}' as index for buffered segments GeoDataFrame.")
        gdf_segments_buffered = gdf_segments_buffered.set_index(id_col_segments)
        if gdf_segments_buffered.index.has_duplicates:
            logger.warning(f"Duplicate indices found in gdf_segments_buffered. Keeping first.")
            gdf_segments_buffered = gdf_segments_buffered[~gdf_segments_buffered.index.duplicated(keep='first')]
        
        logger.info(f"Prepared buffered segments GeoDataFrame with {len(gdf_segments_buffered)} unique segments.")
        return gdf_segments_buffered

    except Exception as e:
        logger.error(f"Failed to load or prepare parking segments: {e}", exc_info=True)
        return None

def generate_poi_features_for_batch_chunk(df_sub_chunk_with_buffers, processed_pois_by_category, radii, poi_categories, target_crs):
    """
    Core POI feature generation for a sub-chunk of a batch.
    `df_sub_chunk_with_buffers` already has buffer geometries and original_batch_index.
    Returns a DataFrame with POI features, indexed by original_batch_index.
    """
    # Initialize POI columns for this sub-chunk
    poi_feature_columns_names = []
    for category in poi_categories:
        for r in radii:
            count_col = f'poi_{category}_count_{r}m'
            present_col = f'poi_{category}_present_{r}m'
            df_sub_chunk_with_buffers[count_col] = 0
            df_sub_chunk_with_buffers[present_col] = False
            poi_feature_columns_names.extend([count_col, present_col])
    
    # Determine the total bounding box for the buffers in this sub-chunk to clip POIs
    all_sub_chunk_buffers_list = []
    for r in radii:
        # Ensure buffer column exists and is not all NaT/NaN (though dropna should handle this before)
        if f'buffer_{r}m' in df_sub_chunk_with_buffers.columns:
            all_sub_chunk_buffers_list.extend(df_sub_chunk_with_buffers[f'buffer_{r}m'].dropna().tolist())
    
    if not all_sub_chunk_buffers_list:
        logger.warning("Sub-chunk has no valid buffer geometries. Skipping POI calculation, returning initialized columns.")
        return df_sub_chunk_with_buffers[['original_batch_index'] + poi_feature_columns_names].set_index('original_batch_index')

    # Calculate BBOX for clipping POIs
    # Need to handle cases where all_sub_chunk_buffers_list might be empty if all buffers were NaN
    try:
        with warnings.catch_warnings(): # Suppress potential empty geometry warnings if all buffers were bad
            warnings.simplefilter("ignore", RuntimeWarning)
            total_bounds_geom = unary_union(all_sub_chunk_buffers_list) # Returns MultiPolygon or single Polygon
        if total_bounds_geom.is_empty:
            logger.warning("Total bounds geometry for sub-chunk is empty. Skipping POI sjoin.")
            return df_sub_chunk_with_buffers[['original_batch_index'] + poi_feature_columns_names].set_index('original_batch_index')
        sub_chunk_bbox = total_bounds_geom.bounds # (minx, miny, maxx, maxy)
    except Exception as bbox_err:
        logger.error(f"Error calculating BBOX for sub-chunk: {bbox_err}. Skipping POI sjoin for this sub-chunk", exc_info=True)
        return df_sub_chunk_with_buffers[['original_batch_index'] + poi_feature_columns_names].set_index('original_batch_index')

    minx, miny, maxx, maxy = sub_chunk_bbox
    sub_chunk_bbox_poly = box(minx, miny, maxx, maxy)
    clip_gdf = gpd.GeoDataFrame([{'geometry': sub_chunk_bbox_poly}], crs=target_crs)

    # POI Category Loop
    for category, _ in poi_categories.items():
        gdf_pois_cat_all_preprocessed = processed_pois_by_category.get(category)
        if gdf_pois_cat_all_preprocessed is None or gdf_pois_cat_all_preprocessed.empty:
            continue # Counts will remain 0, present flags False

        # Clip preprocessed POIs to the sub-chunk's BBOX
        # Ensure CRS match, though it should be target_crs from preprocessing
        if gdf_pois_cat_all_preprocessed.crs is None or gdf_pois_cat_all_preprocessed.crs.to_string().upper() != target_crs.upper():
             logger.warning(f"Re-projecting preprocessed '{category}' POIs to {target_crs} before clip.")
             gdf_pois_cat_all_preprocessed = gdf_pois_cat_all_preprocessed.to_crs(target_crs)
        
        gdf_pois_sub_chunk_clipped = gpd.clip(gdf_pois_cat_all_preprocessed, clip_gdf)
        if gdf_pois_sub_chunk_clipped is None or gdf_pois_sub_chunk_clipped.empty:
            continue

        # Perform spatial join for each radius
        for r in radii:
            buffer_col = f'buffer_{r}m'
            count_col = f'poi_{category}_count_{r}m'
            present_col = f'poi_{category}_present_{r}m'

            # Prepare buffers for sjoin (only for rows that have valid buffer geometry for this radius)
            # df_sub_chunk_with_buffers is already indexed by original_batch_index here
            current_buffers_gdf = gpd.GeoDataFrame(
                df_sub_chunk_with_buffers[[buffer_col, 'original_batch_index']], 
                geometry=buffer_col, 
                crs=target_crs
            ).dropna(subset=[buffer_col]).set_index('original_batch_index')
            
            if current_buffers_gdf.empty: continue

            try:
                # Ensure gdf_pois_sub_chunk_clipped is not empty and has geometry
                if 'geometry' not in gdf_pois_sub_chunk_clipped.columns or gdf_pois_sub_chunk_clipped.geometry.name != 'geometry':
                    gdf_pois_sub_chunk_clipped = gdf_pois_sub_chunk_clipped.set_geometry('geometry')
                
                joined_pois = gpd.sjoin(current_buffers_gdf, gdf_pois_sub_chunk_clipped, how='left', predicate='intersects')
                
                # Aggregate counts: group by original_batch_index (which is the index of current_buffers_gdf)
                # index_right is the index from gdf_pois_sub_chunk_clipped, count non-NaNs
                counts = joined_pois.groupby(joined_pois.index)['index_right'].count() 
                
                # Update main sub-chunk DataFrame using its original_batch_index
                # The .add is if multiple sjoin_chunks contribute to the same original_batch_index (not the case here with this func structure)
                # So, direct assignment is fine. Or use .update if partial updates are complex.
                df_sub_chunk_with_buffers[count_col] = df_sub_chunk_with_buffers[count_col].add(counts, fill_value=0).astype(int)
                df_sub_chunk_with_buffers[present_col] = df_sub_chunk_with_buffers[present_col] | (counts > 0)

            except Exception as sjoin_err:
                logger.error(f"Error during sjoin/aggregation for {category}, radius {r}m: {sjoin_err}", exc_info=True)
    
    # Return only original_batch_index and the POI feature columns
    return df_sub_chunk_with_buffers[['original_batch_index'] + poi_feature_columns_names].set_index('original_batch_index')


def transform_poi_counts(df_with_poi_features, poi_categories, radii):
    """Applies log1p to count columns and ensures presence flags are boolean."""
    logger.info("Applying log transformation and type casting to POI features...")
    log_transform_cols_count = 0
    for category in poi_categories:
        for r in radii:
            count_col = f'poi_{category}_count_{r}m'
            present_col = f'poi_{category}_present_{r}m'
            log_col_name = count_col.replace('_count_', '_log1p_count_')

            if count_col in df_with_poi_features.columns:
                 df_with_poi_features[count_col] = pd.to_numeric(df_with_poi_features[count_col], errors='coerce').fillna(0).astype(int)
                 df_with_poi_features[log_col_name] = np.log1p(df_with_poi_features[count_col])
                 log_transform_cols_count +=1
            else:
                 logger.warning(f"Count column {count_col} not found for log transformation.")

            if present_col in df_with_poi_features.columns:
                 df_with_poi_features[present_col] = df_with_poi_features[present_col].fillna(False).astype(bool)
            else:
                 logger.warning(f"Present column {present_col} not found for type casting.")
    logger.info(f"Log transformation applied, creating {log_transform_cols_count} log count columns. Presence flags cast to bool.")
    return df_with_poi_features

# --- Main Orchestration ---
def main():
    start_time_main = time.time()
    logger.info("--- Starting Main POI Feature Engineering Process ---")
    os.makedirs(TEMP_BATCH_DIR, exist_ok=True) # Ensure temp directory exists

    # --- 0. Pre-checks and Preparations ---\n    if not os.path.exists(BASE_FEATURES_INPUT_PATH):\n        logger.error(f"Base features input file not found: {BASE_FEATURES_INPUT_PATH}. Exiting.")\n        return\n    if not os.path.exists(PARKING_SEGMENTS_PATH):\n        logger.error(f"Parking segments file not found: {PARKING_SEGMENTS_PATH}. Exiting.")\n        return\n    if not os.path.exists(OSM_PBF_PATH):\n        logger.error(f"OSM PBF file not found: {OSM_PBF_PATH}. Exiting.")\n        return\n\n    # --- 1. Load Base Features Data ---\n    logger.info(f"Loading base features from: {BASE_FEATURES_INPUT_PATH}")\n    df_base_full = pd.read_parquet(BASE_FEATURES_INPUT_PATH)\n    logger.info(f"Loaded base features: {df_base_full.shape[0]} rows, {df_base_full.shape[1]} columns.")\n\n    # --- For testing with a slice ---\n    # TEST_SLICE_SIZE = 1000  # Or any small number for a quick test\n    # df_base_full = df_base_full.head(TEST_SLICE_SIZE)\n    # logger.info(f"--- !!! TESTING WITH A SLICE OF {TEST_SLICE_SIZE} ROWS !!! ---")\n    # ------------------------------------\n\n    if ID_COLUMN_HISTORY not in df_base_full.columns:\n        logger.error(f"Critical: ID column '{ID_COLUMN_HISTORY}' not found in base features. Available: {df_base_full.columns.tolist()}")\n        return\n    \n    try:\n        df_base_full[ID_COLUMN_HISTORY] = pd.to_numeric(df_base_full[ID_COLUMN_HISTORY])\n        logger.info(f"Converted base feature ID column \\'{ID_COLUMN_HISTORY}\\' to numeric.")\n    except Exception as e:\n        logger.error(f"Could not convert base feature ID column \\'{ID_COLUMN_HISTORY}\\' to numeric: {e}. This is critical.")\n        return\n\n// ... existing code ...

    logger.info("--- Step 1: Upfront POI Preprocessing ---")
    processed_pois_by_category = preprocess_all_pois(
        OSM_PBF_PATH, POI_CATEGORIES, TARGET_CRS, 
        pickle_path=PREPROCESSED_POIS_PICKLE_PATH, force_recompute=False)
    if not processed_pois_by_category:
        logger.error("Halting: POI preprocessing returned an empty or None dictionary.")
        return

    # Log the contents of the preprocessed POIs dictionary for verification
    logger.info("--- Contents of preprocessed_pois_by_category: ---")
    for category_name, gdf_val in processed_pois_by_category.items():
        if gdf_val is None:
            logger.info(f"Category '{category_name}': None (no POIs found or error during processing)")
        else:
            logger.info(f"Category '{category_name}': GeoDataFrame with shape {gdf_val.shape}, CRS: {gdf_val.crs}")
    
    # Check if all values in the dictionary are None or empty GeoDataFrames
    all_values_are_none_or_empty = all(
        gdf is None or gdf.empty for gdf in processed_pois_by_category.values()
    )

    if all_values_are_none_or_empty:
        logger.error("Halting: POI preprocessing completed, but all categories resulted in no valid POIs (all are None or empty GeoDataFrames).")
        return
    logger.info("POI preprocessing check passed: Found valid POIs for at least one category.")

    logger.info("--- Step 2: Load and Prepare Segment Buffers ---")
    gdf_segments_buffered = load_and_prepare_segment_buffers(
        PARKING_SEGMENTS_PATH, ID_COLUMN_SEGMENTS, POI_RADII, TARGET_CRS)
    if gdf_segments_buffered is None or gdf_segments_buffered.empty:
        logger.error("Halting: Loading or preparing segment buffers failed.")
        return

    logger.info(f"--- Step 3: Batch Processing of {BASE_FEATURES_INPUT_PATH} ---")
    logger.info(f"Loading base features from {BASE_FEATURES_INPUT_PATH} for batching...")
    try:
        df_base_full = pd.read_parquet(BASE_FEATURES_INPUT_PATH)
        logger.info(f"Loaded base features: {df_base_full.shape[0]} rows, {df_base_full.shape[1]} columns.")
    except Exception as e:
        logger.error(f"Failed to load base features file: {e}. Halting.", exc_info=True)
        return

    total_rows = len(df_base_full)
    num_batches = int(np.ceil(total_rows / BATCH_SIZE))
    logger.info(f"Total rows: {total_rows}. Batch size: {BATCH_SIZE}. Number of batches: {num_batches}")
    processed_batch_files = []

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        current_batch_num = batch_idx + 1
        start_row = batch_idx * BATCH_SIZE
        end_row = min(start_row + BATCH_SIZE, total_rows)
        
        logger.info(f"--- Processing Batch {current_batch_num}/{num_batches} (rows {start_row}-{end_row}) ---")
        # Use .copy() to avoid SettingWithCopyWarning on slices
        df_batch_input = df_base_full.iloc[start_row:end_row].copy()
        # Preserve original index from df_base_full for this batch input
        df_batch_input['original_full_index'] = df_batch_input.index.copy()

        logger.info(f"Merging batch with segment buffers (ID: '{ID_COLUMN_HISTORY}' with index '{ID_COLUMN_SEGMENTS}')...")
        df_batch_with_geoms = pd.merge(
            df_batch_input,
            gdf_segments_buffered, # Has ID_COLUMN_SEGMENTS as index
            left_on=ID_COLUMN_HISTORY,
            right_index=True, 
            how='left',
            suffixes=('', '_segment') # Avoid column name clashes if any
        )
        # df_batch_with_geoms now has 'original_full_index' and buffer columns
        # The index of df_batch_with_geoms is the original index from df_batch_input (slice of df_base_full)

        # Sub-chunking for POI sjoin within the batch
        all_sjoin_chunk_results = []
        num_sjoin_chunks = int(np.ceil(len(df_batch_with_geoms) / POI_SJOIN_CHUNK_SIZE))
        logger.info(f"Starting POI feature generation for batch {current_batch_num} in {num_sjoin_chunks} sjoin-chunks (size={POI_SJOIN_CHUNK_SIZE})...")

        for sjoin_chunk_idx in range(num_sjoin_chunks):
            sjoin_chunk_start_row = sjoin_chunk_idx * POI_SJOIN_CHUNK_SIZE
            sjoin_chunk_end_row = min(sjoin_chunk_start_row + POI_SJOIN_CHUNK_SIZE, len(df_batch_with_geoms))
            
            df_sjoin_sub_chunk = df_batch_with_geoms.iloc[sjoin_chunk_start_row:sjoin_chunk_end_row].copy()
            # IMPORTANT: Pass 'original_full_index' to generate_poi_features_for_batch_chunk
            # and ensure it's used for indexing the results. Let's rename 'original_batch_index' 
            # in the function to 'original_df_index' for clarity
            df_sjoin_sub_chunk.rename(columns={'original_full_index': 'original_df_index'}, inplace=True)
            # Ensure the index for sjoin sub chunk is the original_df_index for proper rejoining later
            df_sjoin_sub_chunk_indexed = df_sjoin_sub_chunk.set_index('original_df_index', drop=False) # Keep column too

            logger.info(f"  Processing sjoin-chunk {sjoin_chunk_idx+1}/{num_sjoin_chunks} for batch {current_batch_num}...")
            
            # Drop rows where buffer merge failed (no segment match -> any buffer col is NaN)
            # This is critical before calculating BBOX or passing to sjoin func
            buffer_check_cols = [f'buffer_{r}m' for r in POI_RADII]
            df_sjoin_sub_chunk_valid_geoms = df_sjoin_sub_chunk_indexed.dropna(subset=buffer_check_cols).copy()

            if df_sjoin_sub_chunk_valid_geoms.empty:
                logger.info(f"  Sjoin-chunk {sjoin_chunk_idx+1} has no valid geometries after merge check. Initializing empty POI features.")
                # Create empty POI features for all rows in the original df_sjoin_sub_chunk_indexed
                empty_poi_cols_df = pd.DataFrame(index=df_sjoin_sub_chunk_indexed.index)
                for category in POI_CATEGORIES: 
                    for r_val in POI_RADII: 
                        empty_poi_cols_df[f'poi_{category}_count_{r_val}m'] = 0
                        empty_poi_cols_df[f'poi_{category}_present_{r_val}m'] = False
                all_sjoin_chunk_results.append(empty_poi_cols_df)
                continue
            
            # The generate_poi_features_for_batch_chunk expects 'original_batch_index'
            # so we pass the correctly indexed and named df_sjoin_sub_chunk_valid_geoms
            # It expects 'original_batch_index' in its input columns. Renaming to original_df_index to match general processing.
            df_sjoin_sub_chunk_valid_geoms.rename(columns={'original_df_index': 'original_batch_index'}, inplace=True) 
            
            poi_features_sjoin_chunk = generate_poi_features_for_batch_chunk(
                df_sjoin_sub_chunk_valid_geoms, # Already indexed by original_df_index
                processed_pois_by_category, POI_RADII, POI_CATEGORIES, TARGET_CRS
            )
            all_sjoin_chunk_results.append(poi_features_sjoin_chunk)
        
        if not all_sjoin_chunk_results:
            logger.warning(f"Batch {current_batch_num} produced no POI feature results from sjoin chunks. Skipping.")
            # Create empty POI features for the whole batch_input if needed for robust concat
            df_poi_features_for_batch = pd.DataFrame(index=df_batch_input.index) # Use original batch input index
            for category in POI_CATEGORIES:
                for r_val in POI_RADII:
                    df_poi_features_for_batch[f'poi_{category}_count_{r_val}m'] = 0
                    df_poi_features_for_batch[f'poi_{category}_present_{r_val}m'] = False
        else:
            df_poi_features_for_batch = pd.concat(all_sjoin_chunk_results) # Index is original_df_index
        
        # df_poi_features_for_batch is now indexed by 'original_df_index'
        # which corresponds to the index of df_base_full for the rows in this batch.
        # We need to join it back to df_batch_input which has 'original_full_index' column
        # and its own iloc-based slice index.
        
        logger.info(f"Transforming POI counts for batch {current_batch_num}...")
        df_poi_features_transformed = transform_poi_counts(df_poi_features_for_batch, POI_CATEGORIES, POI_RADII)

        logger.info(f"Joining transformed POI features back to batch {current_batch_num}...")
        # Use the 'original_full_index' from df_batch_input to align with df_poi_features_transformed's index
        df_batch_output = df_batch_input.set_index('original_full_index', drop=False).join(df_poi_features_transformed, how='left')
        df_batch_output = df_batch_output.set_index(df_batch_input.index) # Restore original iloc-based index for consistency if needed, or drop original_full_index
        df_batch_output = df_batch_output.drop(columns=['original_full_index'], errors='ignore')
        
        poi_cols_final = [col for col in df_batch_output.columns if col.startswith('poi_')]
        for col in poi_cols_final:
            if '_present_' in col: df_batch_output[col] = df_batch_output[col].fillna(False).astype(bool)
            elif '_count_' in col: df_batch_output[col] = df_batch_output[col].fillna(0).astype(int)
            elif '_log1p_count_' in col : df_batch_output[col] = df_batch_output[col].fillna(0)

        batch_output_filename = os.path.join(TEMP_BATCH_DIR, f"features_batch_{current_batch_num:03d}.parquet")
        try:
            logger.info(f"Saving processed batch {current_batch_num} to {batch_output_filename}...")
            df_batch_output.to_parquet(batch_output_filename, index=False)
            processed_batch_files.append(batch_output_filename)
            logger.info(f"Batch {current_batch_num} saved. Shape: {df_batch_output.shape}")
        except Exception as e:
            logger.error(f"Error saving batch {current_batch_num}: {e}", exc_info=True)
            continue
        
        batch_end_time = time.time()
        logger.info(f"--- Batch {current_batch_num}/{num_batches} completed in {batch_end_time - batch_start_time:.2f} seconds ---")
        del df_batch_input, df_batch_with_geoms, df_poi_features_for_batch, df_poi_features_transformed, df_batch_output
        gc.collect()

    logger.info(f"--- Step 4: Concatenating All Processed Batches ({len(processed_batch_files)} files) ---")
    if not processed_batch_files:
        logger.error("No batch files were processed or saved. Halting.")
        return
    all_batches_dfs = [pd.read_parquet(f) for f in processed_batch_files]
    if not all_batches_dfs:
        logger.error("Could not load any batch DataFrames for concatenation. Halting.")
        return
    
    df_final_master_table = pd.concat(all_batches_dfs, ignore_index=True)
    logger.info(f"Final master table concatenated. Shape: {df_final_master_table.shape}")

    logger.info(f"--- Step 5: Saving Final Master Table and Cleaning Up ---")
    try:
        df_final_master_table.to_parquet(FINAL_OUTPUT_PATH, index=False)
        logger.info(f"Final master table saved to {FINAL_OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Error saving final master table: {e}", exc_info=True)

    logger.info("Cleaning up temporary batch files...")
    for batch_file in processed_batch_files:
        try: os.remove(batch_file); logger.info(f"Removed {batch_file}")
        except Exception as e: logger.error(f"Error removing {batch_file}: {e}")
    if os.path.exists(TEMP_BATCH_DIR) and not os.listdir(TEMP_BATCH_DIR):
        try: os.rmdir(TEMP_BATCH_DIR); logger.info(f"Removed {TEMP_BATCH_DIR}")
        except Exception as e: logger.error(f"Error removing {TEMP_BATCH_DIR}: {e}")
    
    overall_end_time = time.time()
    logger.info(f"Phase 2: Add POI Features (Batched) completed in {overall_end_time - start_time_main:.2f} seconds.")

if __name__ == "__main__":
    main() 