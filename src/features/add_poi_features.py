"""
POI Feature Engineering Pipeline for Barcelona Parking Analysis

Enriches parking segment data with Points of Interest (POI) proximity features.
"""

import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
from shapely.validation import make_valid
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
import time # Added for retry delay

# Configure logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "add_poi_features.log")

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Set root logger level

# Remove any existing handlers to avoid duplicate console logs if re-running in same session
# This is important if the script might be run multiple times in an interactive session
# or if the logger is configured elsewhere in a way that might lead to duplicates.
current_handlers = root_logger.handlers[:] # Iterate over a copy
for handler in current_handlers:
    # Defensive check if it's a handler we might have added or a default one
    # This aims to prevent removing handlers not managed by this script, though
    # generally for a script-based logger, clearing all previous is fine.
    # For more complex apps, one might be more selective.
    root_logger.removeHandler(handler)


# Configure StreamHandler for console output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO) # Console can be less verbose
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# Configure FileHandler for file output
file_handler = logging.FileHandler(LOG_FILE, mode='w') # 'w' to overwrite log each run
file_handler.setLevel(logging.DEBUG) # File log can be very verbose
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__) # Get logger for the current module

# Configuration
class Config:
    # --- Paths ---
    INPUT_PATH = 'data/processed/parking_predictions_phase1_enriched.parquet'
    OUTPUT_PATH = 'data/processed/parking_predictions_with_pois.parquet'
    STATS_PATH = 'reports/poi_statistics.csv'
    FIGURES_DIR = 'reports/figures' 

    # --- Geospatial Settings ---
    PROJECTED_CRS = 'EPSG:32631'  # UTM Zone 31N for Barcelona
    DEFAULT_CRS = 'EPSG:4326'     # WGS84

    # --- POI Fetching Settings ---
    FETCH_BUFFER_DEGREES = 0.01 
    BARCELONA_FALLBACK_BBOX = (41.30, 41.47, 2.05, 2.25) # S, N, W, E 
    TILE_SIZE = 0.001              # Degrees (~100m tiles)
    MAX_WORKERS = 4               # Parallel workers for tile fetching
    TIMEOUT = 600                 # OSMnx Global Timeout (seconds)
    MAX_RETRIES = 3               # Maximum retries for fetching a tile
    RETRY_DELAY = 5               # Seconds to wait between retries

    # POI categories for feature engineering
    POI_CATEGORIES = {
        'retail': {'shop': True},
        'food': {'amenity': ['restaurant', 'cafe', 'bar']},
        'office': {'office': True},
        'transport_stops': {'highway': 'bus_stop', 'public_transport': 'stop_position'},
        'transport_stations': {'railway': ['station', 'subway_entrance', 'tram_stop']},
        'leisure': {'leisure': True},
        'tourism': {'tourism': True},
        'healthcare': {'amenity': ['hospital', 'clinic', 'doctors', 'pharmacy']},
        'education': {'amenity': ['school', 'university', 'college', 'kindergarten']}
    }

    # --- Counting Settings ---
    POI_SEARCH_RADII = [100, 200, 500]  # Meters

# --- OSMnx Global Settings ---
ox.settings.timeout = (Config.TIMEOUT, Config.TIMEOUT)
ox.settings.log_console = False # Quieter OSMnx output, use our logger
ox.settings.use_cache = True
ox.settings.requests_kwargs = {}  # Ensure no timeout here
ox.settings.max_query_area_size = 1e12 # Keep this to prevent excessive subdivision

def create_empty_projected_poi_gdf() -> gpd.GeoDataFrame:
    """Helper to create an empty GeoDataFrame for POIs with the projected CRS."""
    logger.info("Creating an empty GeoDataFrame with projected CRS for POIs.")
    return gpd.GeoDataFrame({'geometry': []}, crs=Config.PROJECTED_CRS)

def load_data(file_path: str) -> gpd.GeoDataFrame:
    """Load and preprocess parking segment data."""
    logger.info(f"Loading parking segment data from {file_path}...")
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} records.")
    
    if 'geometry' not in df.columns:
        logger.error("'geometry' column missing from input data.")
        raise ValueError("'geometry' column missing.")
    
    logger.info("Converting geometries...")
    tqdm.pandas(desc="Parsing Geometries")
    df['geometry'] = df['geometry'].progress_apply(parse_geometry)
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=Config.DEFAULT_CRS)
    
    initial_rows = len(gdf)
    gdf = gdf.dropna(subset=['geometry'])
    logger.warning(f"Dropped {initial_rows - len(gdf)} rows with null geometry after parsing.")
        
    invalid_geom_mask = ~gdf.geometry.is_valid
    if invalid_geom_mask.any():
        logger.info(f"Found {invalid_geom_mask.sum()} invalid geometries. Attempting to make valid...")
        gdf.loc[invalid_geom_mask, 'geometry'] = gdf.loc[invalid_geom_mask, 'geometry'].progress_apply(make_valid)
        still_invalid = ~gdf.geometry.is_valid
        if still_invalid.any():
             logger.warning(f"{still_invalid.sum()} geometries remain invalid after make_valid.")
             gdf = gdf[gdf.geometry.is_valid]
             
    initial_rows = len(gdf)
    gdf = gdf[~gdf.geometry.is_empty]
    if len(gdf) < initial_rows:
         logger.warning(f"Dropped {initial_rows - len(gdf)} rows with empty geometry.")
         
    if gdf.empty:
         logger.error("GeoDataFrame is empty after cleaning. Cannot proceed.")
         raise ValueError("No valid parking geometries found.")
         
    logger.info(f"Preprocessing complete. {len(gdf)} valid geometries remaining.")
    return gdf

def parse_geometry(geom_dict: dict):
    logger.debug(f"Attempting to parse input: {str(geom_dict)[:200]}...") 
    if not isinstance(geom_dict, dict):
        logger.debug("Input is not a dictionary.")
        return None
    try:
        geom_type = geom_dict.get('type')
        coords = geom_dict.get('coordinates')

        if not geom_type:
            logger.debug("Geometry 'type' is missing.")
            return None
        if coords is None:
            logger.debug("Geometry 'coordinates' is None.")
            return None

        if geom_type == 'LineString':
            logger.debug(f"Processing LineString with coords type: {type(coords)}")
            if isinstance(coords, np.ndarray):
                processed_coords = [tuple(point) for point in coords if isinstance(point, (np.ndarray, list, tuple)) and len(point) == 2]
            elif isinstance(coords, list):
                processed_coords = [tuple(point) for point in coords if isinstance(point, (list, tuple)) and len(point) == 2]
            else:
                logger.debug(f"LineString coords type not list or ndarray: {type(coords)}")
                return None
            
            if len(processed_coords) < 2:
                logger.debug(f"LineString requires at least 2 valid points, got {len(processed_coords)}")
                return None
            geom = LineString(processed_coords)
            # logger.debug(f"Successfully created LineString: {geom.wkt[:50]}...")

        elif geom_type == 'MultiLineString':
            logger.debug(f"Processing MultiLineString with coords type: {type(coords)}")
            if not isinstance(coords, np.ndarray) or coords.ndim == 0 :
                logger.debug(f"MultiLineString coords not a non-empty numpy array: type={type(coords)}, ndim={getattr(coords, 'ndim', 'N/A')}")
                return None
            
            if coords.shape[0] == 0 : 
                 logger.debug("MultiLineString coords array is empty (shape[0] == 0). Cannot extract list of lines.")
                 return None

            # Based on logs, coords[0] is the array of points for the single line
            # in this MultiLineString (e.g., np.array([P1_arr, P2_arr, ...]))
            single_line_points_array = coords[0]
            # logger.debug(f"Extracted single_line_points_array: {single_line_points_array}")

            if not isinstance(single_line_points_array, np.ndarray):
                logger.debug(f"MultiLineString's single_line_points_array is not a numpy array: {type(single_line_points_array)}")
                return None
            
            if single_line_points_array.ndim == 0 or single_line_points_array.shape[0] == 0:
                 logger.debug(f"MultiLineString's single_line_points_array is empty or not an array of points. Shape: {single_line_points_array.shape}")
                 return None

            current_line_points = []
            # Iterate over the points in this single line's coordinate array
            for point_item in single_line_points_array:
                # logger.debug(f"  Processing point_item from single_line_points_array: {point_item}")
                if isinstance(point_item, np.ndarray):
                    # Expecting point_item to be np.array([lon, lat])
                    if point_item.ndim == 1 and point_item.shape == (2,):
                        current_line_points.append(tuple(point_item))
                    else:
                        logger.debug(f"    Skipping malformed point ndarray (ndim!=1 or shape!=(2,)): ndim={point_item.ndim}, shape={point_item.shape}")
                elif isinstance(point_item, (list, tuple)) and len(point_item) == 2:
                    # Expecting point_item to be [lon, lat] or (lon, lat)
                    # Further check if elements are numbers, though Shapely might handle mixed types
                    if all(isinstance(coord_val, (int, float, np.number)) for coord_val in point_item):
                        current_line_points.append(tuple(point_item))
                    else:
                        logger.debug(f"    Skipping malformed point list/tuple (non-numeric elements): {point_item}")
                else:
                    logger.debug(f"    Skipping malformed point item (not a 2-element array/list/tuple): {type(point_item)}")
                    continue
            
            if len(current_line_points) < 2:
                logger.debug(f"No valid LineString created from MultiLineString data (only {len(current_line_points)} valid points found).")
                return None
            
            # Create a single LineString from the extracted points
            the_line = LineString(current_line_points)
            # Wrap this single LineString in a MultiLineString
            geom = MultiLineString([the_line])
            # logger.debug(f"Successfully created MultiLineString with 1 line: {geom.wkt[:50]}...")
        else:
            logger.warning(f"Unsupported geometry type: {geom_type}")
            return None
        
        # logger.debug(f"Attempting make_valid on {geom.wkt[:50]}...")
        valid_geom = make_valid(geom)
        # logger.debug(f"make_valid returned: {valid_geom.wkt[:50]}...")
        return valid_geom
    
    except FloatingPointError as e: # Keep this specific for now if it was useful
        logger.error(f"FloatingPointError parsing: {e} for dict: {str(geom_dict)[:200]}...", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"UNHANDLED Exception during parsing: {e} for dict: {str(geom_dict)[:200]}...", exc_info=True)
        raise # Re-raise to see the full traceback in this test script

def fetch_poi_data(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch POI data using tiled parallel requests."""
    bounds = gdf.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    fetch_bbox = (
        max(min_lat - Config.FETCH_BUFFER_DEGREES, Config.BARCELONA_FALLBACK_BBOX[0]),
        min(max_lat + Config.FETCH_BUFFER_DEGREES, Config.BARCELONA_FALLBACK_BBOX[1]),
        max(min_lon - Config.FETCH_BUFFER_DEGREES, Config.BARCELONA_FALLBACK_BBOX[2]),
        min(max_lon + Config.FETCH_BUFFER_DEGREES, Config.BARCELONA_FALLBACK_BBOX[3])
    )
    logger.info(f"Calculated fetch bounding box (S, N, W, E): {fetch_bbox}")

    tiles = generate_tiles(fetch_bbox, Config.TILE_SIZE)
    if not tiles: 
        logger.error("No tiles generated.")
        return create_empty_projected_poi_gdf()

    unique_tags = get_unique_tags(Config.POI_CATEGORIES)
    if not unique_tags: 
        logger.warning("No POI tags defined in config.")
        return create_empty_projected_poi_gdf()

    logger.info(f"Fetching POIs for {len(unique_tags)} tag groups across {len(tiles)} tiles using up to {Config.MAX_WORKERS} workers...")
    all_pois_list = []
    
    fetch_args = [(tile, unique_tags) for tile in tiles]

    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        # Using a wrapper function for exception handling and logging within threads might be beneficial
        # For simplicity, direct map is used here.
        # results_iterable = executor.map(fetch_pois_for_tile_wrapper, fetch_args)
        # For now, assuming fetch_pois_for_tile can be called directly and handles its own errors/logging for POIs
        # The original notebook had a more complex structure for fetch_pois_for_tile, which needs to be adapted or simplified.
        # Let's assume a simplified fetch_pois_for_tile that takes (tile_bbox, tags_dict) and returns a GDF or None
        
        # Placeholder for the actual parallel fetching logic. 
        # The original code snippet for parallel execution was incomplete.
        # We need a function like `fetch_pois_for_tile(tile_bbox, tags_dict)` that osmnx.features_from_bbox can use.
        # For now, I'll adapt the sequential logic to fit the parallel structure conceptually.
        # This part will likely need further refinement based on the exact implementation of `fetch_pois_for_tile`.

        # Simplified conceptual parallel execution:
        # This requires a function `_fetch_tile_data(args)` where args is (tile, unique_tags)
        # and it returns the GeoDataFrame for that tile.
        
        # Let's define a simple wrapper for the purpose of this example, 
        # assuming ox.features_from_bbox is the core fetching mechanism per tile.
        def _fetch_tile_data(tile_args):
            tile_bbox, tags = tile_args
            retries = 0
            while retries <= Config.MAX_RETRIES:
                try:
                    logger.info(f"Attempt {retries + 1}/{Config.MAX_RETRIES + 1} for tile {tile_bbox}")
                    # This is a conceptual call; actual POI fetching logic might be more complex
                    # and involve iterating through categories within `tags` if `ox.features_from_bbox` 
                    # doesn't handle multiple tag groups well in one go, or if specific logging per category is needed.
                    # For simplicity, assuming `tags` is directly usable by `features_from_bbox`
                    # or that the function handles the structure of `unique_tags`.
                    gdf_tile = ox.features_from_bbox(bbox=tile_bbox, tags=tags)
                    if not gdf_tile.empty:
                        # Ensure CRS is set before projecting, OSMnx usually returns WGS84
                        if gdf_tile.crs is None:
                            gdf_tile = gdf_tile.set_crs(Config.DEFAULT_CRS, allow_override=True)
                        elif gdf_tile.crs != Config.DEFAULT_CRS:
                            logger.warning(f"Tile {tile_bbox} fetched with unexpected CRS {gdf_tile.crs}. Re-setting to {Config.DEFAULT_CRS}")
                            gdf_tile = gdf_tile.set_crs(Config.DEFAULT_CRS, allow_override=True)
                        return gdf_tile.to_crs(Config.PROJECTED_CRS)
                    else:
                        logger.info(f"Tile {tile_bbox} fetched successfully but returned no POIs.")
                        return None # Explicitly return None for empty successful fetches
                except Exception as e:
                    logger.error(f"Error fetching POIs for tile {tile_bbox} on attempt {retries + 1}: {e}")
                    retries += 1
                    if retries <= Config.MAX_RETRIES:
                        logger.info(f"Retrying in {Config.RETRY_DELAY} seconds...")
                        time.sleep(Config.RETRY_DELAY)
                    else:
                        logger.error(f"Max retries reached for tile {tile_bbox}. Skipping.")
                        return None # Return None after max retries
            return None # Should be reached if loop finishes due to retries exhausted

        logger.info(f"Submitting {len(fetch_args)} tile fetching tasks to ThreadPoolExecutor...")
        # Use tqdm to show progress for the parallel execution
        results_iterable = tqdm(executor.map(_fetch_tile_data, fetch_args), total=len(fetch_args), desc="Fetching POI Tiles")

        for gdf_tile in results_iterable:
            if gdf_tile is not None and not gdf_tile.empty:
                all_pois_list.append(gdf_tile)

    # Consolidate all POIs into a single GeoDataFrame
    if not all_pois_list:
        logger.warning("No POIs were fetched after parallel processing. Returning an empty GeoDataFrame.")
        return create_empty_projected_poi_gdf()

    logger.info(f"Concatenating results from {len(all_pois_list)} successful tiles...")
    # Ensure individual GDFs have CRS before concat, if fetch_tile guarantees it.
    # OSMnx usually returns in WGS84 (DEFAULT_CRS)
    pois = pd.concat(all_pois_list, ignore_index=True)
    if pois.empty: # Check if concat resulted in an empty DataFrame
        logger.warning("Concatenated POI DataFrame is empty.")
        return create_empty_projected_poi_gdf()
        
    pois = gpd.GeoDataFrame(pois, geometry='geometry', crs=Config.DEFAULT_CRS)

    if 'osmid' in pois.columns and 'element_type' in pois.columns:
        initial_count = len(pois)
        pois = pois.drop_duplicates(subset=['element_type', 'osmid'], keep='first')
        logger.info(f"Removed {initial_count - len(pois)} duplicate POIs.")
    
    if pois.empty: 
        logger.warning("POI GeoDataFrame empty after deduplication.")
        return create_empty_projected_poi_gdf()
        
    # Keep only geometry and the tag keys we queried for, if they exist as columns
    relevant_cols = ['geometry'] + [key for key in unique_tags.keys() if key in pois.columns]
    # Filter out columns not present to avoid KeyErrors
    pois = pois[[col for col in relevant_cols if col in pois.columns]]
    
    if 'geometry' not in pois.columns or pois['geometry'].isnull().all():
        logger.warning("No valid geometries in POI data after filtering columns.")
        return create_empty_projected_poi_gdf()

    logger.info(f"Projecting {len(pois)} unique POIs to {Config.PROJECTED_CRS}...")
    try: 
        return pois.to_crs(Config.PROJECTED_CRS)
    except Exception as e: 
        logger.error(f"Error projecting POIs: {e}")
        return create_empty_projected_poi_gdf()

def generate_tiles(bbox: Tuple[float], tile_size: float) -> List[Tuple]:
    min_lat, max_lat, min_lon, max_lon = bbox
    if not (min_lat < max_lat and min_lon < max_lon): logger.error(f"Invalid bbox for tiling: {bbox}"); return []
    tiles = []
    lon_steps = np.arange(min_lon, max_lon, tile_size)
    lat_steps = np.arange(min_lat, max_lat, tile_size)
    for west in lon_steps:
        for south in lat_steps:
            east = min(west + tile_size, max_lon)
            north = min(south + tile_size, max_lat)
            if north > south and east > west: tiles.append((north, south, east, west))
    logger.info(f"Generated {len(tiles)} tiles")
    return tiles

def get_unique_tags(categories: Dict) -> Dict:
    final_tags: Dict[str, Union[bool, set[str]]] = {}

    for cat_name, cat_details in categories.items():
        # logger.debug(f"Processing category: {cat_name}")
        for tag_key, tag_val_spec in cat_details.items():
            # logger.debug(f"  Raw tag: {tag_key}={tag_val_spec} (current final_tags[{tag_key}] = {final_tags.get(tag_key)})")
            if tag_val_spec is True:
                # If True is specified, this key should fetch all. Overrides specific strings.
                if final_tags.get(tag_key) is True:
                    pass # Already True, no change needed
                else:
                    # logger.debug(f"    Setting final_tags[{tag_key}] to True (was {final_tags.get(tag_key)})")
                    final_tags[tag_key] = True
            elif isinstance(tag_val_spec, str):
                # If it's a string, add to set if the key is not already True.
                if final_tags.get(tag_key) is not True:
                    if tag_key not in final_tags:
                        final_tags[tag_key] = set()
                    # logger.debug(f"    Adding str '{tag_val_spec}' to final_tags[{tag_key}] set")
                    if isinstance(final_tags[tag_key], set): # mypy check
                        final_tags[tag_key].add(tag_val_spec) 
            elif isinstance(tag_val_spec, list):
                # If it's a list of strings, add them to set if the key is not already True.
                if final_tags.get(tag_key) is not True:
                    if tag_key not in final_tags:
                        final_tags[tag_key] = set()
                    if isinstance(final_tags[tag_key], set): # mypy check
                        for item in tag_val_spec:
                            if isinstance(item, str):
                                # logger.debug(f"    Adding list item str '{item}' to final_tags[{tag_key}] set")
                                final_tags[tag_key].add(item)
                            else:
                                logger.warning(f"    Item '{item}' in list for tag '{tag_key}' is not a string, skipping.")
            else:
                logger.warning(f"  Unsupported tag value specification for {tag_key}={tag_val_spec} in category {cat_name}")
            # logger.debug(f"  Updated final_tags[{tag_key}] = {final_tags.get(tag_key)}")

    # Convert sets to lists for the final output, leave True as True
    output_tags = {}
    for key, value in final_tags.items():
        if value is True:
            output_tags[key] = True
        elif isinstance(value, set):
            if value:  # Only add if set is not empty
                output_tags[key] = list(value)
        # If value is an empty set, it means no valid string tags were added and it wasn't True,
        # so we don't add this key to output_tags, to avoid e.g. {'some_key': []}
        # which might be interpreted by osmnx as "tag must be an empty list".
    
    logger.debug(f"Generated unique tags for OSMnx: {output_tags}")
    return output_tags

def fetch_tile(tile_bbox_nswe: Tuple, tags: Dict) -> gpd.GeoDataFrame:
    # We rely on the globally set ox.settings.requests_kwargs for timeout.
    # Explicitly setting ox.settings.timeout here might conflict if features_from_bbox also tries to pass it.
    # ox.settings.timeout = Config.TIMEOUT # REMOVE THIS LINE
    
    # logger.debug(f"Fetching features for tile {tile_bbox_nswe} with global timeout settings applied.") # Replaced by more specific logging below

    try:
        logger.info(f"STARTING ox.features_from_bbox for tile {tile_bbox_nswe} with tags: {tags}")
        gdf = ox.features_from_bbox(tile_bbox_nswe, tags) # Positional bbox
        logger.info(f"COMPLETED ox.features_from_bbox for tile {tile_bbox_nswe}, got {len(gdf)} rows.")
        
        # Ensure CRS is set immediately after fetching
        if not gdf.empty:
            gdf = gdf.set_crs(Config.DEFAULT_CRS, allow_override=True)
            if 'osmid' not in gdf.columns:
                 # Use index directly if simple index, otherwise extract from multi-index
                if isinstance(gdf.index, pd.MultiIndex):
                    gdf['osmid'] = gdf.index.get_level_values('osmid') if 'osmid' in gdf.index.names else gdf.index.map(lambda x: x[1] if isinstance(x, tuple) and len(x)>1 else x)
                else:
                    gdf['osmid'] = gdf.index
            if 'element_type' not in gdf.columns:
                if isinstance(gdf.index, pd.MultiIndex):
                    gdf['element_type'] = gdf.index.get_level_values('element_type') if 'element_type' in gdf.index.names else gdf.index.map(lambda x: x[0] if isinstance(x, tuple) else 'node')
                else: # Fallback for simple index, assume 'node' or handle based on geometry?
                     # This case might need refinement depending on actual index structure for non-MultiIndex features
                    gdf['element_type'] = 'node' # Default assumption 
        
        # logger.debug(f"Successfully fetched {len(gdf)} features for tile {tile_bbox_nswe}") # Covered by new COMPLETED log
        return gdf
    
    except Exception as e:
        logger.error(f"EXCEPTION in fetch_tile for {tile_bbox_nswe}: {e}", exc_info=True) # Added exc_info=True
        # Return a valid empty GeoDataFrame with geometry column and CRS
        return gpd.GeoDataFrame({'geometry': []}, crs=Config.DEFAULT_CRS) 
    finally:
        logger.info(f"ENDING fetch_tile function for tile {tile_bbox_nswe}")

def count_nearby_pois(buffer_geom, pois_gdf: gpd.GeoDataFrame, pois_sindex):
    if buffer_geom is None or buffer_geom.is_empty or pois_gdf.empty or pois_sindex is None: return 0
    try:
        possible_matches_idx = list(pois_sindex.intersection(buffer_geom.bounds))
        if not possible_matches_idx: return 0
        possible_matches = pois_gdf.iloc[possible_matches_idx]
        precise_matches = possible_matches[possible_matches.intersects(buffer_geom)]
        return len(precise_matches)
    except Exception as e: logger.debug(f"Error in count_nearby_pois: {e}"); return 0

def calculate_poi_counts(gdf: gpd.GeoDataFrame, pois_projected: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if pois_projected.empty or 'geometry' not in pois_projected.columns:
        logger.warning("Skipping POI counting - no valid POI data.")
        for radius in Config.POI_SEARCH_RADII:
            for category in Config.POI_CATEGORIES.keys(): gdf[f'poi_{category}_count_{radius}m'] = 0
        return gdf

    logger.info(f"Preparing for POI counting against {len(pois_projected)} POIs...")
    if gdf.crs.to_string() != Config.PROJECTED_CRS: # More robust CRS comparison
        logger.info(f"Projecting {len(gdf)} parking segments to {Config.PROJECTED_CRS}...")
        gdf_proj = gdf.to_crs(Config.PROJECTED_CRS)
    else: 
        gdf_proj = gdf.copy() 

    for radius in Config.POI_SEARCH_RADII:
        logger.info(f"Processing {radius}m buffers...")
        buffers = gdf_proj.geometry.buffer(radius)
        for category, cat_tags_dict in Config.POI_CATEGORIES.items():
            col_name = f'poi_{category}_count_{radius}m'
            logger.info(f"Counting {category} for radius {radius}m...")
            mask = pd.Series(False, index=pois_projected.index)
            for tag_key, tag_values_criteria in cat_tags_dict.items():
                if tag_key not in pois_projected.columns: continue
                poi_col_values = pois_projected[tag_key]
                if isinstance(tag_values_criteria, bool) and tag_values_criteria is True:
                    mask |= poi_col_values.notna() & (poi_col_values != '') 
                else:
                    crit_list = [str(c).lower() for c in (tag_values_criteria if isinstance(tag_values_criteria, list) else [tag_values_criteria])]
                    mask |= poi_col_values.astype(str).str.lower().isin(crit_list)
            
            category_pois = pois_projected[mask]
            if category_pois.empty: gdf[col_name] = 0; logger.info(f"  No POIs for '{category}', count set to 0."); continue
            
            category_sindex = category_pois.sindex
            logger.info(f"  Applying spatial count for {len(category_pois)} '{category}' POIs...")
            tqdm.pandas(desc=f"Counting {category} ({radius}m)")
            gdf[col_name] = buffers.progress_apply(lambda buf: count_nearby_pois(buf, category_pois, category_sindex))
    return gdf

def generate_reports(gdf: gpd.GeoDataFrame):
    poi_cols = gdf.filter(regex='poi_.*_count').columns
    if poi_cols.empty: logger.warning("No POI cols. Skipping stats."); return
    logger.info("Generating POI statistics report...")
    stats = gdf[poi_cols].agg(['mean', 'std', 'max', lambda x: (x > 0).mean() * 100]).T
    stats.columns = ['mean', 'std', 'max', 'non_zero_pct']
    stats.index.name = 'feature'
    try:
        stats.to_csv(Config.STATS_PATH)
        logger.info(f"Saved statistics to {Config.STATS_PATH}")
        print("\n--- POI Statistics Summary ---"); print(stats.round(2))
    except Exception as e: logger.error(f"Failed to save stats: {e}")
    plot_sample_data(gdf) # Call plotting

def plot_sample_data(gdf: gpd.GeoDataFrame):
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plot_path = os.path.join(Config.FIGURES_DIR, 'poi_distribution_sample.png')
    plot_col = next((col for col in gdf.columns if col.startswith('poi_hospital_count')), None)
    if not plot_col:
         plot_col = next((col for col in gdf.columns if col.startswith('poi_') and col.endswith('m')), None)
    if not plot_col: logger.warning("No suitable POI column found for plotting."); return

    try:
        import matplotlib.pyplot as plt
        sample_size = min(1000, len(gdf))
        if sample_size == 0 : logger.warning("GDF is empty, skipping plot."); return
        sample = gdf.sample(sample_size)
        fig, ax = plt.subplots(figsize=(12, 12))
        sample.plot(ax=ax, column=plot_col, legend=True, markersize=5, cmap='viridis', alpha=0.7, missing_kwds={'color': 'lightgrey'})
        plt.title(f'{plot_col} Distribution (Sample of {sample_size})'); plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.tight_layout(); plt.savefig(plot_path); plt.close()
        logger.info(f"Saved sample plot to {plot_path}")
    except Exception as e: logger.warning(f"Plotting failed: {e}")

def main():
    logger.info("Starting POI feature engineering pipeline...")
    try:
        parking_data = load_data(Config.INPUT_PATH)
        pois_projected = fetch_poi_data(parking_data)
        enriched_data = calculate_poi_counts(parking_data.copy(), pois_projected)
        logger.info(f"Saving enriched data ({len(enriched_data)} rows) to {Config.OUTPUT_PATH}...")
        enriched_data.to_parquet(Config.OUTPUT_PATH, index=False)
        logger.info(f"Successfully saved enriched data.")
        generate_reports(enriched_data)
    except Exception as e: logger.error(f"Pipeline failed: {e}", exc_info=True)
    logger.info("POI pipeline finished.")

if __name__ == "__main__":
    main()