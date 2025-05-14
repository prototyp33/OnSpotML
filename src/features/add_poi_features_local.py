import os
import logging
import geopandas as gpd
from pyrosm import OSM
import pandas as pd
import warnings
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
import numpy as np

# --- Config ---
PARKING_PATH = "data/processed/parking_predictions_phase1_enriched.parquet"
OSM_PBF_PATH = "data/raw/cataluna-latest.osm.pbf" # Make sure this file exists
OUTPUT_PATH = "data/processed/parking_predictions_with_pois_local_filtered.parquet" # New output path
STATS_PATH = "reports/poi_statistics_local_filtered.csv" # New stats path
CRS = "EPSG:32631"  # UTM 31N - Appropriate for Barcelona/Catalonia
TARGET_CITY_NAME = "Barcelona" # Used for filtering boundary

# Define POI categories to extract - EXPAND AS NEEDED
POI_FILTER = {
    "amenity": [
        'school', 'university', 'hospital', 'clinic', # Original filter
        'restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'food_court', # Food
        'cinema', 'theatre', 'nightclub', 'community_centre', # Entertainment/Social
        'pharmacy', 'doctors', 'dentist', # Health
        'bank', 'atm', # Finance
        'kindergarten', 'library' # Education/Public
        ],
    "shop": True,  # All shops
    "tourism": [
        'hotel', 'hostel', 'guest_house', 'motel', # Accommodation
        'museum', 'attraction', 'gallery', 'viewpoint', 'theme_park' # Attractions
        ],
    "leisure": [
        'stadium', 'sports_centre', 'pitch', 'swimming_pool', 'fitness_centre', # Sports
        'park', 'playground', 'garden', 'marina' # Recreation
        ],
    # Add more categories like 'office': True, 'building': [...], 'sport': True if relevant
}
# List of specific amenity types you want individual counts for (must match keys in POI_FILTER or be subtypes)
POI_TYPES = ["school", "university", "hospital", "clinic", "restaurant", "shop", "hotel", "museum", "park", "stadium"] # Example expanded list
RADII = [100, 200, 500] # Buffer radii in meters

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("add_poi_features_local_filtered")

# Ignore warnings that might come from pyrosm/geopandas internal operations
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Setting custom osm file format.*')


# --- Geometry Parsing Helper ---
# Integrated from previous debugging steps
def parse_geometry(geom_dict):
    """Parses a geometry dictionary (potentially with NumPy arrays) into a Shapely object."""
    if not isinstance(geom_dict, dict) or 'type' not in geom_dict or 'coordinates' not in geom_dict:
        # If input is already a Shapely object, return it directly
        if hasattr(geom_dict, 'geom_type'):
             return geom_dict
        logger.warning(f"Invalid geometry structure encountered: {geom_dict}")
        return None
    
    geom_type = geom_dict['type']
    coords = geom_dict['coordinates']

    try:
        if geom_type == 'Point':
            if isinstance(coords, (np.ndarray, list, tuple)) and len(coords) == 2:
                return Point(float(coords[0]), float(coords[1]))
            else:
                 raise ValueError(f"Invalid coordinates for Point: {coords}")
        elif geom_type == 'LineString':
            processed_coords = [tuple(map(float, p)) for p in coords if isinstance(p, (np.ndarray, list, tuple)) and len(p) == 2]
            if len(processed_coords) < 2:
                 raise ValueError(f"Invalid coordinates for LineString (need >= 2 points): {processed_coords}")
            return LineString(processed_coords)
        elif geom_type == 'MultiLineString':
             processed_lines = []
             if len(coords) == 1 and isinstance(coords[0], (np.ndarray, list)) and len(coords[0]) > 0 and isinstance(coords[0][0], (np.ndarray, list)):
                 single_line_points = coords[0]
                 processed_coords = [tuple(map(float, p)) for p in single_line_points if isinstance(p, (np.ndarray, list, tuple)) and len(p) == 2]
                 if len(processed_coords) >= 2:
                     processed_lines.append(LineString(processed_coords))
                 else:
                     logger.warning(f"Skipping invalid single LineString within MultiLineString (needs >= 2 points): {single_line_points}")
             else:
                 lines_to_process = coords
                 for line in lines_to_process:
                     if isinstance(line, (np.ndarray, list)):
                         processed_coords = [tuple(map(float, p)) for p in line if isinstance(p, (np.ndarray, list, tuple)) and len(p) == 2]
                         if len(processed_coords) >= 2:
                             processed_lines.append(LineString(processed_coords))
                         else:
                             logger.warning(f"Skipping invalid LineString within MultiLineString (needs >= 2 points): {line}")
                     else:
                         logger.warning(f"Skipping non-list/array element within MultiLineString coords: {line}")
                         
             if not processed_lines:
                 logger.error(f"Could not extract any valid LineStrings from MultiLineString coords: {coords}")
                 return None 
             return MultiLineString(processed_lines)
        elif geom_type == 'Polygon':
            exterior = [tuple(map(float, p)) for p in coords[0] if isinstance(p, (np.ndarray, list, tuple)) and len(p) == 2]
            interiors = []
            if len(coords) > 1:
                for ring in coords[1:]:
                    interiors.append([tuple(map(float, p)) for p in ring if isinstance(p, (np.ndarray, list, tuple)) and len(p) == 2])
            if len(exterior) < 3:
                 raise ValueError(f"Invalid exterior ring for Polygon (need >= 3 points): {exterior}")
            return Polygon(exterior, interiors if interiors else None)
        elif geom_type == 'MultiPolygon':
            polygons = []
            for poly_coords in coords:
                 # Add robust handling for polygon structure variations if needed
                if not poly_coords or not poly_coords[0]: continue # Skip empty polygon coords
                exterior = [tuple(map(float, p)) for p in poly_coords[0] if isinstance(p, (np.ndarray, list, tuple)) and len(p) == 2]
                interiors = []
                if len(poly_coords) > 1:
                     for ring in poly_coords[1:]:
                        if ring: # Check if interior ring is not empty
                            interiors.append([tuple(map(float, p)) for p in ring if isinstance(p, (np.ndarray, list, tuple)) and len(p) == 2])
                if len(exterior) >= 3:
                    polygons.append(Polygon(exterior, interiors if interiors else None))
            if not polygons:
                 raise ValueError(f"No valid polygons found for MultiPolygon: {coords}")
            return MultiPolygon(polygons)
        else:
            logger.warning(f"Unsupported geometry type: {geom_type}")
            return None
    except (ValueError, TypeError, IndexError) as e: # Catch potential errors during conversion/indexing
        logger.error(f"Error parsing geometry type {geom_type} with coords {coords}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing geometry type {geom_type}: {e}")
        return None


# --- Function to Safely Load and Prepare Parking Data ---
def load_parking_data(path, target_crs):
    logger.info(f"Loading parking segments from {path} ...")
    if not os.path.exists(path):
        logger.error(f"Parking data file not found at {path}")
        raise FileNotFoundError(f"Parking data file not found at {path}")
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
         logger.error(f"Failed to read Parquet file {path}: {e}")
         raise

    # Check if geometry column exists
    if 'geometry' not in df.columns:
         logger.error("Input parking data MUST contain a 'geometry' column.")
         raise ValueError("Input parking data MUST contain a 'geometry' column.")
         
    try:
        # ----> PARSE GEOMETRY COLUMN using helper function <----
        logger.info("Parsing 'geometry' column into Shapely objects...")
        # Use .loc to avoid potential SettingWithCopyWarning
        df.loc[:, 'geometry_parsed'] = df['geometry'].apply(parse_geometry)
        
        # Drop rows where geometry parsing failed
        original_count = len(df)
        df = df.dropna(subset=['geometry_parsed'])
        parsed_count = len(df)
        if parsed_count < original_count:
            logger.warning(f"Dropped {original_count - parsed_count} rows due to geometry parsing errors.")

        if parsed_count == 0:
             logger.error("No valid geometries found after parsing. Cannot proceed.")
             raise ValueError("No valid geometries found after parsing.")

        # Convert to GeoDataFrame using the newly parsed geometry column
        # Assume original CRS is WGS84 (EPSG:4326) if not specified otherwise
        gdf = gpd.GeoDataFrame(df, geometry="geometry_parsed", crs="EPSG:4326")
        # Optional: drop the original geometry column if no longer needed
        gdf = gdf.drop(columns=['geometry'], errors='ignore') # errors='ignore' handles case where it might not exist if gdf reused df directly

        logger.info(f"Loaded and parsed {len(gdf)} parking segments. Initial CRS: {gdf.crs}")
        
        # Project to target metric CRS
        if gdf.crs != target_crs:
            logger.info(f"Projecting parking data to {target_crs}...")
            gdf = gdf.to_crs(target_crs)
            logger.info(f"Parking data projected. Current CRS: {gdf.crs}")
        else:
             logger.info(f"Parking data already in target CRS: {target_crs}")
             
        # Ensure index is unique and suitable for joining later
        gdf = gdf.reset_index(drop=True)
        
        return gdf
    except Exception as e:
        logger.error(f"Error processing parking data after loading: {e}", exc_info=True)
        raise


# --- Main Script Logic ---
try:
    # --- Load Parking Data ---
    parking_gdf = load_parking_data(PARKING_PATH, CRS)

    # --- Load POIs from OSM PBF ---
    logger.info(f"Loading POIs from {OSM_PBF_PATH} ...")
    if not os.path.exists(OSM_PBF_PATH):
         raise FileNotFoundError(f"OSM PBF file not found at {OSM_PBF_PATH}")

    osm = OSM(OSM_PBF_PATH)
    all_pois = osm.get_pois(custom_filter=POI_FILTER)

    if all_pois is None or all_pois.empty:
        logger.warning(f"No POIs found matching the filter {POI_FILTER} in {OSM_PBF_PATH}. Saving data without POI features.")
        # Save the processed parking data without POIs and exit
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        parking_gdf.to_parquet(OUTPUT_PATH, index=False)
        logger.info(f"Saved data without POIs to {OUTPUT_PATH}")
        exit()
    logger.info(f"Loaded {len(all_pois)} POIs matching filter in Catalonia.")

    # --- Project POIs to metric CRS ---
    logger.info(f"Projecting POIs to {CRS}...")
    if all_pois.crs != CRS:
        all_pois = all_pois.to_crs(CRS)
    logger.info(f"POIs projected. Current CRS: {all_pois.crs}")


    # --- Get Target City Boundary ---
    boundary_gdf = None
    logger.info(f"Attempting to fetch boundary for {TARGET_CITY_NAME}...")
    try:
        # Note: Boundary fetching by name can be unreliable.
        # Consider providing a path to a local boundary file (Shapefile, GeoJSON) as a fallback.
        admin_boundaries = osm.get_boundaries(boundary_type="administrative")
        if admin_boundaries is not None and not admin_boundaries.empty:
             target_boundary = admin_boundaries[admin_boundaries["name"].str.contains(TARGET_CITY_NAME, case=False, na=False)]
             if not target_boundary.empty:
                 boundary_gdf = target_boundary.iloc[[0]] # Take the first match
                 if boundary_gdf.crs != CRS:
                      boundary_gdf = boundary_gdf.to_crs(CRS)
                 logger.info(f"Boundary for {TARGET_CITY_NAME} obtained and projected.")
             else:
                 logger.warning(f"Could not find boundary for '{TARGET_CITY_NAME}' using pyrosm.")
        else:
             logger.warning("pyrosm.get_boundaries returned no administrative boundaries.")

    except Exception as e:
        logger.warning(f"Error fetching boundary: {e}. Will proceed without geographic filtering of POIs.")


    # --- Filter POIs Geographically (if boundary available) ---
    if boundary_gdf is not None:
        logger.info(f"Filtering POIs to within {TARGET_CITY_NAME} boundary...")
        try:
            # Ensure boundary has a valid geometry
            boundary_geom = boundary_gdf.geometry.iloc[0]
            if boundary_geom is None or boundary_geom.is_empty:
                 raise ValueError("Boundary geometry is invalid or empty.")
                 
            # Use spatial index on POIs for faster filtering (implicitly used by geopandas)
            pois_within_boundary = all_pois[all_pois.geometry.within(boundary_geom)]
            
            if pois_within_boundary.empty:
                logger.warning(f"No POIs found strictly within the {TARGET_CITY_NAME} boundary after filtering.")
                # Optional: Try intersects if within yields nothing
                # logger.info("Trying intersects query for POIs...")
                # pois_within_boundary = all_pois[all_pois.geometry.intersects(boundary_geom)]
                # if pois_within_boundary.empty:
                #      logger.warning(f"No POIs found intersecting the {TARGET_CITY_NAME} boundary either.")
                #      pois_to_use = gpd.GeoDataFrame(columns=all_pois.columns, geometry=[], crs=CRS) # Use empty GDF
                # else:
                #      pois_to_use = pois_within_boundary
                #      logger.info(f"Found {len(pois_to_use)} POIs intersecting the boundary.")
                # For now, stick to 'within' and use empty if none found
                pois_to_use = gpd.GeoDataFrame(columns=all_pois.columns, geometry=[], crs=CRS) # Use empty GDF

            else:
                 pois_to_use = pois_within_boundary
                 logger.info(f"Filtered down to {len(pois_to_use)} POIs within the {TARGET_CITY_NAME} area.")

        except Exception as e:
            logger.error(f"Error during geographic filtering of POIs: {e}. Using all Catalonia POIs.")
            pois_to_use = all_pois # Fallback to using all POIs
    else:
        logger.warning("Proceeding without geographic filtering as boundary was not available.")
        pois_to_use = all_pois # Use all extracted POIs


    # --- Process Parking Data in Chunks --- 
    CHUNK_SIZE = 20000 # Adjust based on available memory (Reduced from 100000)
    num_chunks = int(np.ceil(len(parking_gdf) / CHUNK_SIZE))
    logger.info(f"Parking data has {len(parking_gdf)} rows. Processing in {num_chunks} chunks of size {CHUNK_SIZE}.")
    
    processed_chunks = [] # List to hold processed chunks

    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(parking_gdf))
        chunk_gdf = parking_gdf.iloc[start_idx:end_idx].copy() # Get current chunk
        
        logger.info(f"--- Processing Chunk {i+1}/{num_chunks} (rows {start_idx}-{end_idx-1}) ---")

        # --- Pre-calculate Buffers for the current chunk --- 
        buffers = {}
        logger.info("Pre-calculating buffers for the current chunk...")
        for radius in RADII:
            # Ensure index is preserved from the chunk
            buffers[radius] = gpd.GeoDataFrame(geometry=chunk_gdf.geometry.buffer(radius), crs=CRS, index=chunk_gdf.index)
        logger.info("Chunk buffer calculation complete.")

        # --- Count POIs for each specific type and radius (for the chunk) ---
        logger.info("Starting POI counting for specific types for the chunk...")
        for poi_type_key in POI_TYPES: 
            # (POI filtering logic remains the same as it's applied to the full pois_to_use set)
            pois_type_gdf = gpd.GeoDataFrame(columns=pois_to_use.columns, crs=CRS, geometry=[]) # Initialize as empty
            if poi_type_key == 'shop' and 'shop' in pois_to_use.columns:
                pois_type_gdf = pois_to_use[pois_to_use['shop'].notna()]
            elif 'amenity' in pois_to_use.columns and poi_type_key in pois_to_use['amenity'].unique():
                 pois_type_gdf = pois_to_use[pois_to_use["amenity"] == poi_type_key]
            elif 'tourism' in pois_to_use.columns and poi_type_key in pois_to_use['tourism'].unique():
                 pois_type_gdf = pois_to_use[pois_to_use["tourism"] == poi_type_key]
            elif 'leisure' in pois_to_use.columns and poi_type_key in pois_to_use['leisure'].unique():
                 pois_type_gdf = pois_to_use[pois_to_use["leisure"] == poi_type_key]
            else:
                 if 'shop' in pois_to_use.columns and poi_type_key in pois_to_use['shop'].unique():
                     pois_type_gdf = pois_to_use[pois_to_use['shop'] == poi_type_key]
                 else:
                     # Log warning, create empty cols on the CHUNK
                     logger.warning(f"Chunk {i+1}: Cannot directly filter for POI type '{poi_type_key}'. Columns will be zero.")
                     for radius in RADII:
                         col = f"poi_{poi_type_key}_count_{radius}m"
                         chunk_gdf[col] = 0
                     continue 

            if pois_type_gdf.empty:
                # Log info, create empty cols on the CHUNK
                logger.info(f"Chunk {i+1}: No POIs of type '{poi_type_key}' found in the filtered area. Columns will be zero.")
                for radius in RADII:
                    col = f"poi_{poi_type_key}_count_{radius}m"
                    chunk_gdf[col] = 0
                continue 

            for radius in RADII:
                buffer_gdf = buffers[radius]
                
                try:
                    if buffer_gdf.geometry.isnull().all() or pois_type_gdf.geometry.isnull().all():
                        logger.warning(f"Chunk {i+1}: Skipping sjoin for {poi_type_key} at {radius}m due to empty geometries.")
                        count_col_name = f"poi_{poi_type_key}_count_{radius}m"
                        chunk_gdf[count_col_name] = 0 # Add column to chunk
                        continue
                        
                    joined = gpd.sjoin(buffer_gdf, pois_type_gdf, how="left", predicate="intersects")
                    
                    count_col_name = f"poi_{poi_type_key}_count_{radius}m"
                    if 'index_right' in joined.columns and joined['index_right'].notna().any():
                         counts = joined.groupby(level=0)['index_right'].count()
                         # Map counts back TO THE CHUNK, fill missing buffer indices with 0
                         chunk_gdf[count_col_name] = counts.reindex(buffer_gdf.index).fillna(0).astype(int)
                    else: 
                         chunk_gdf[count_col_name] = 0 # Add column to chunk

                except Exception as e:
                     logger.error(f"Chunk {i+1}: Error during sjoin/count for {poi_type_key} at {radius}m: {e}", exc_info=False) # Reduce noise maybe
                     col = f"poi_{poi_type_key}_count_{radius}m"
                     chunk_gdf[col] = -1 # Indicate error in the chunk
            # Log chunk progress periodically
            # logger.info(f"Chunk {i+1}: Finished counting for POI type: {poi_type_key}")
        logger.info(f"Chunk {i+1}: Finished counting for specific POI types.")

        # --- Count all filtered POIs combined (for the chunk) ---
        logger.info(f"Chunk {i+1}: Counting all filtered POIs combined...")
        if pois_to_use.empty:
            logger.warning(f"Chunk {i+1}: No POIs found in the filtered area for 'all' count. Columns will be zero.")
            for radius in RADII:
                col = f"poi_all_count_{radius}m"
                chunk_gdf[col] = 0 # Add column to chunk
        else:
            for radius in RADII:
                buffer_gdf = buffers[radius]
                try:
                    if buffer_gdf.geometry.isnull().all() or pois_to_use.geometry.isnull().all():
                         logger.warning(f"Chunk {i+1}: Skipping sjoin for ALL POIs at {radius}m due to empty geometries.")
                         count_col_name = f"poi_all_count_{radius}m"
                         chunk_gdf[count_col_name] = 0 # Add column to chunk
                         continue

                    joined = gpd.sjoin(buffer_gdf, pois_to_use, how="left", predicate="intersects")

                    count_col_name = f"poi_all_count_{radius}m"
                    if 'index_right' in joined.columns and joined['index_right'].notna().any():
                         counts = joined.groupby(level=0)['index_right'].count()
                         chunk_gdf[count_col_name] = counts.reindex(buffer_gdf.index).fillna(0).astype(int) # Add column to chunk
                    else:
                         chunk_gdf[count_col_name] = 0 # Add column to chunk
                         
                except Exception as e:
                     logger.error(f"Chunk {i+1}: Error during sjoin/count for ALL POIs at {radius}m: {e}", exc_info=False) # Reduce noise maybe
                     col = f"poi_all_count_{radius}m"
                     chunk_gdf[col] = -1 # Indicate error in the chunk
            logger.info(f"Chunk {i+1}: Finished counting for ALL POI types.")

        # Append the processed chunk (now with POI columns) to the list
        # Drop geometry temporarily if concatenation issues arise, or ensure consistent schema
        processed_chunks.append(chunk_gdf)
        logger.info(f"--- Finished Processing Chunk {i+1}/{num_chunks} ---")
        # Optional: Clean up memory explicitly
        del chunk_gdf, buffers, joined, counts
        import gc
        gc.collect()

    # --- Concatenate processed chunks --- 
    logger.info("Concatenating processed chunks...")
    if not processed_chunks:
        logger.error("No chunks were processed. Exiting.")
        exit()
    # Ensure all chunks have the same columns before concatenating, handle potential errors
    # (Error columns (-1) might cause dtype issues if not handled carefully)
    final_gdf = pd.concat(processed_chunks, ignore_index=True)
    # Convert back to GeoDataFrame if needed (depends if geometry was dropped)
    # We kept geometry, so it should still be a GeoDataFrame after concat if dtypes align
    logger.info("Concatenation complete.")

    # --- Save enriched data ---
    logger.info(f"Saving enriched data to {OUTPUT_PATH} ...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Geopandas to_parquet handles geometry types directly with pyarrow backend
    final_gdf.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved enriched data.")


    # --- Save POI statistics ---
    logger.info("Calculating POI statistics...")
    poi_cols = [col for col in final_gdf.columns if col.startswith("poi_")]
    if poi_cols: # Check if any POI columns were actually added
        # Ensure columns used for stats are numeric, coerce errors to NaN
        stats_df = final_gdf[poi_cols].apply(pd.to_numeric, errors='coerce')
        # Drop columns that are all NaN after coercion (e.g., if all had errors like -1)
        stats_df = stats_df.dropna(axis=1, how='all')
        
        if not stats_df.empty:
            stats = stats_df.agg(['mean', 'std', 'min', 'max', lambda x: (x > 0).mean() * 100]).T
            stats.columns = ['mean', 'std', 'min', 'max', 'non_zero_pct']
            stats.index.name = 'feature'
            os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
            stats.to_csv(STATS_PATH)
            logger.info(f"Saved POI statistics to {STATS_PATH}")
            print("--- POI Statistics Summary ---")
            # Format output for better readability
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(stats.round(2))
        else:
             logger.warning("No valid numeric POI columns found for statistics after coercion.")
    else:
        logger.warning("No POI columns found in the final dataframe. Skipping statistics.")

    logger.info("Script finished successfully.")

except FileNotFoundError as e:
     logger.error(f"File not found error: {e}")
except ValueError as e:
     logger.error(f"Value error: {e}")
except ImportError as e:
     logger.error(f"Import error: {e}. Make sure all libraries (pyrosm, geopandas, pandas, numpy, shapely, pyarrow) are installed.")
except Exception as e:
     logger.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback 