import pandas as pd
import geopandas as gpd
import os

def diagnose_segment_ids():
    print("Starting ID diagnosis...")
    # --- Configure for aparcaments_sota_superficie.csv ---
    SEGMENT_FILE_PATH = "data/parking/aparcaments_sota_superficie.csv"
    SEGMENT_ID_COLUMN_NAME = 'codi' # Potential ID column in this file
    # This file is expected to be a CSV, not a GeoPackage for this run.
    IS_GEOPACKAGE = False
    # Columns to load for context if available (lat/lon for points)
    SEGMENT_GEO_COLS = ['latitud', 'longitud', 'nom'] # Add 'nom' for context

    print(f"\n--- Attempting to load {SEGMENT_FILE_PATH} ---")
    try:
        if IS_GEOPACKAGE:
            # This block is not used for aparcaments_sota_superficie.csv currently
            df_segments = gpd.read_file(SEGMENT_FILE_PATH)
        else:
            # Read as CSV, ensure all potential ID and geo columns are loaded
            cols_to_load = [SEGMENT_ID_COLUMN_NAME] + [col for col in SEGMENT_GEO_COLS if col != SEGMENT_ID_COLUMN_NAME]
            # Check if file exists before trying to read
            if not os.path.exists(SEGMENT_FILE_PATH):
                print(f"ERROR: Segment file not found: {SEGMENT_FILE_PATH}")
                return
            df_segments = pd.read_csv(SEGMENT_FILE_PATH, usecols=lambda c: c in cols_to_load)
        
        print(f"Successfully loaded {SEGMENT_FILE_PATH}")
        print(f"Shape: {df_segments.shape}")
        print(f"Columns found: {df_segments.columns.tolist()}")
        
        if SEGMENT_ID_COLUMN_NAME not in df_segments.columns:
            print(f"ERROR: Identified ID column '{SEGMENT_ID_COLUMN_NAME}' not found in {os.path.basename(SEGMENT_FILE_PATH)} columns.")
            return

        print(f"Using '{SEGMENT_ID_COLUMN_NAME}' as the segment identifier.")
        print(f"Data type of '{SEGMENT_ID_COLUMN_NAME}': {df_segments[SEGMENT_ID_COLUMN_NAME].dtype}")
        
        # Attempt to convert identified ID column to numeric, coercing errors to NaN
        df_segments['ID_compare_numeric'] = pd.to_numeric(df_segments[SEGMENT_ID_COLUMN_NAME], errors='coerce')
        num_non_numeric_segments = df_segments['ID_compare_numeric'].isnull().sum()
        
        if num_non_numeric_segments > 0:
            print(f"WARNING: Column '{SEGMENT_ID_COLUMN_NAME}' in {os.path.basename(SEGMENT_FILE_PATH)} contains {num_non_numeric_segments} non-numeric values that were coerced to NaN.")
            print(f"Sample non-numeric values: {df_segments[df_segments['ID_compare_numeric'].isnull()][SEGMENT_ID_COLUMN_NAME].unique()[:5]}")
            df_segments_cleaned_ids = df_segments.dropna(subset=['ID_compare_numeric']).copy()
            # Ensure it becomes int after dropping NaNs for comparison
            df_segments_cleaned_ids['ID_compare'] = df_segments_cleaned_ids['ID_compare_numeric'].astype(int)
        else:
            print(f"Column '{SEGMENT_ID_COLUMN_NAME}' in {os.path.basename(SEGMENT_FILE_PATH)} appears to be numeric or safely convertible.")
            df_segments_cleaned_ids = df_segments.copy()
            # Ensure it becomes int for comparison
            df_segments_cleaned_ids['ID_compare'] = df_segments_cleaned_ids[SEGMENT_ID_COLUMN_NAME].astype(int)

        print(f"First 5 unique ID_compare values from {os.path.basename(SEGMENT_FILE_PATH)}: {df_segments_cleaned_ids['ID_compare'].unique()[:5]}")
        print(f"Number of unique ID_compare values from {os.path.basename(SEGMENT_FILE_PATH)}: {df_segments_cleaned_ids['ID_compare'].nunique()}")
        print(f"Total rows in {os.path.basename(SEGMENT_FILE_PATH)} after potential cleaning: {len(df_segments_cleaned_ids)}")

    except Exception as e:
        print(f"ERROR loading or processing {SEGMENT_FILE_PATH}: {e}")
        return

    # --- Load Historical Data (parking_history_consolidated.parquet) ---
    history_path = "data/processed/parking_history_consolidated.parquet"
    try:
        df_history = pd.read_parquet(history_path, columns=['ID_TRAMO'])
        df_history['ID_TRAMO_numeric'] = pd.to_numeric(df_history['ID_TRAMO'], errors='coerce')
        df_history_cleaned_ids = df_history.dropna(subset=['ID_TRAMO_numeric']).copy()
        df_history_cleaned_ids['ID_TRAMO_compare'] = df_history_cleaned_ids['ID_TRAMO_numeric'].astype(int)
    except Exception as e:
        print(f"ERROR loading or processing {history_path}: {e}")
        return

    # --- Comparison: ID_TRAMO from history vs ID from current segment file ---
    print(f"\n--- Comparison (Historical ID_TRAMO vs {os.path.basename(SEGMENT_FILE_PATH)} '{SEGMENT_ID_COLUMN_NAME}') ---")
    
    unique_history_ids_to_check = pd.Series(df_history_cleaned_ids['ID_TRAMO_compare'].unique())
    current_segment_ids_set = set(df_segments_cleaned_ids['ID_compare'])
    
    missing_from_current_segments = unique_history_ids_to_check[~unique_history_ids_to_check.isin(current_segment_ids_set)]
    found_in_current_segments = unique_history_ids_to_check[unique_history_ids_to_check.isin(current_segment_ids_set)]

    print(f"Number of unique ID_TRAMO_compare values from history: {len(unique_history_ids_to_check)}")
    print(f"Number of unique '{SEGMENT_ID_COLUMN_NAME}' values in {os.path.basename(SEGMENT_FILE_PATH)}: {len(current_segment_ids_set)}")
    print(f"Number of historical ID_TRAMOs FOUND in {os.path.basename(SEGMENT_FILE_PATH)}: {len(found_in_current_segments)}")
    print(f"Number of historical ID_TRAMOs MISSING from {os.path.basename(SEGMENT_FILE_PATH)}: {len(missing_from_current_segments)}")
    
    if not missing_from_current_segments.empty:
        print(f"First 5 historical ID_TRAMOs MISSING from {os.path.basename(SEGMENT_FILE_PATH)}: {missing_from_current_segments.values[:5]}")
    if not found_in_current_segments.empty:
        print(f"First 5 historical ID_TRAMOs FOUND in {os.path.basename(SEGMENT_FILE_PATH)}: {found_in_current_segments.values[:5]}")

    # --- Analysis of Duplicate SEGMENT_ID_COLUMN_NAME in Cleaned Segment File ---
    print(f"\n--- Analysis of Duplicate '{SEGMENT_ID_COLUMN_NAME}' in (cleaned) {os.path.basename(SEGMENT_FILE_PATH)} ---")
    # Use ID_compare for duplicate checking as it's the cleaned numeric version
    duplicates_in_current_segments = df_segments_cleaned_ids[df_segments_cleaned_ids.duplicated(subset=['ID_compare'], keep=False)]
    num_unique_duplicated_ids = duplicates_in_current_segments['ID_compare'].nunique()
    
    print(f"Number of rows in {os.path.basename(SEGMENT_FILE_PATH)} involved in '{SEGMENT_ID_COLUMN_NAME}' duplicates (after cleaning to numeric): {len(duplicates_in_current_segments)}")
    print(f"Number of unique '{SEGMENT_ID_COLUMN_NAME}' values that are duplicated: {num_unique_duplicated_ids}")
    
    if not duplicates_in_current_segments.empty:
        print(f"Sample of duplicated '{SEGMENT_ID_COLUMN_NAME}' (ID_compare) and their original values:")
        # Show original SEGMENT_ID_COLUMN_NAME and the numeric ID_compare for context
        print(duplicates_in_current_segments[[SEGMENT_ID_COLUMN_NAME, 'ID_compare'] + [col for col in SEGMENT_GEO_COLS if col in duplicates_in_current_segments.columns and col != SEGMENT_ID_COLUMN_NAME]].sort_values(by='ID_compare').head(10).to_string())

    print("\nDiagnosis complete.")

if __name__ == '__main__':
    diagnose_segment_ids() 