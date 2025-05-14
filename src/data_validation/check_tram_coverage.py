import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys

# Add project root to sys.path to allow importing BarcelonaDataCollector
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data_ingestion.barcelona_data_collector import BarcelonaDataCollector

# --- Configuration ---
HISTORY_PARQUET_PATH = project_root / "data" / "interim" / "parking_history_consolidated.parquet"
# This is the output of get_traffic_segment_geometries
GEOMETRY_GPKG_PATH = project_root / "data" / "processed" / "trams_geometries.gpkg"
# This is your local raw CSV file for tram definitions
LOCAL_TRAM_RELATION_CSV = project_root / "data" / "raw" / "transit_relacio_trams_format_long.csv"

# Column name for tram IDs in your history data (adjust if different)
HISTORY_TRAM_ID_COLUMN = "ID_TRAMO" 
GEOMETRY_TRAM_ID_COLUMN = "ID_TRAM" # This is set by get_traffic_segment_geometries

def run_geometry_generation():
    """Ensures the geometry file is generated using the local CSV."""
    print(f"--- Running geometry generation using local CSV: {LOCAL_TRAM_RELATION_CSV} ---")
    collector = BarcelonaDataCollector(base_dir=str(project_root / "data"))
    gpkg_path = collector.get_traffic_segment_geometries(local_csv_path=str(LOCAL_TRAM_RELATION_CSV))
    
    if gpkg_path and gpkg_path.exists():
        print(f"Successfully generated/updated geometry file: {gpkg_path}")
        return True
    else:
        print(f"Failed to generate geometry file. Check logs from BarcelonaDataCollector.")
        return False

def check_coverage():
    """Checks the coverage of idTrams from history data in the geometry file."""
    print(f"\n--- Checking idTram coverage ---")

    if not GEOMETRY_GPKG_PATH.exists():
        print(f"Error: Geometry file not found at {GEOMETRY_GPKG_PATH}. Please run generation first.")
        return

    if not HISTORY_PARQUET_PATH.exists():
        print(f"Error: Parking history Parquet file not found at {HISTORY_PARQUET_PATH}.")
        return

    try:
        # Load IDs from the GeoPackage
        print(f"Loading geometries from: {GEOMETRY_GPKG_PATH}")
        gdf_segments = gpd.read_file(GEOMETRY_GPKG_PATH, layer="trams_geometries")
        geom_ids = set(gdf_segments[GEOMETRY_TRAM_ID_COLUMN].unique())
        print(f"Found {len(geom_ids)} unique segment IDs in {GEOMETRY_GPKG_PATH}")

        # Load IDs from the history Parquet
        print(f"Loading history data from: {HISTORY_PARQUET_PATH}")
        df_history = pd.read_parquet(HISTORY_PARQUET_PATH, columns=[HISTORY_TRAM_ID_COLUMN])
        history_ids = set(df_history[HISTORY_TRAM_ID_COLUMN].unique())
        print(f"Found {len(history_ids)} unique segment IDs in {HISTORY_PARQUET_PATH} (column: {HISTORY_TRAM_ID_COLUMN})")

        # Find missing IDs
        missing_ids = history_ids - geom_ids
        
        if not missing_ids:
            print("\nCoverage Check: PASSED! All idTrams from history data are present in the geometry file.")
        else:
            print(f"\nCoverage Check: FAILED. {len(missing_ids)} idTrams from history data are MISSING in the geometry file.")
            print("First 10 missing IDs:")
            for i, missing_id in enumerate(list(missing_ids)[:10]):
                print(f"  - {missing_id}")
            if len(missing_ids) > 10:
                print("  ...")
        
        # Optional: IDs in geometry but not in history (less critical for this specific check)
        # extra_geom_ids = geom_ids - history_ids
        # if extra_geom_ids:
        #     print(f"\nNote: {len(extra_geom_ids)} IDs are in the geometry file but not in this history parquet sample.")
        #     print(f"First 10 extra geometry IDs: {list(extra_geom_ids)[:10]}")

    except Exception as e:
        print(f"An error occurred during coverage check: {e}")

if __name__ == "__main__":
    # Step 1: Ensure the geometry file is up-to-date using the local CSV
    if run_geometry_generation():
        # Step 2: Perform the coverage check
        check_coverage()
    else:
        print("Skipping coverage check due to failure in geometry generation.") 