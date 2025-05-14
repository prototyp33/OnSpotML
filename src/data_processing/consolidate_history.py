# src/data_processing/consolidate_history.py

import pandas as pd
import os
import glob
import logging

# --- Configuration ---
INPUT_DIR = "data/raw/traffic_history/"
OUTPUT_PATH = "data/interim/parking_history_consolidated.parquet"
FILE_PATTERN = "*_TRAMS_TRAMS.csv" # Pattern to match the monthly files
# Exclude potential future/placeholder files if needed
# Example: FILE_PATTERN = "202[2-3]_*_TRAMS_TRAMS.csv"

# Define columns to keep and rename mapping
COLUMNS_TO_KEEP = ['idTram', 'data', 'estatActual']
COLUMN_RENAME_MAP = {
    'idTram': 'ID_TRAMO',
    'data': 'timestamp',
    'estatActual': 'actual_state' # Assuming this maps to our target concept
}
TIMESTAMP_COL = 'timestamp' # Name after renaming

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("consolidate_history")

# --- Main Processing ---
def main():
    logger.info(f"Starting consolidation of history files from: {INPUT_DIR}")
    csv_files = glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN))

    if not csv_files:
        logger.error(f"No CSV files found matching pattern '{FILE_PATTERN}' in {INPUT_DIR}. Exiting.")
        return

    logger.info(f"Found {len(csv_files)} files to process.")
    all_data = []

    for file_path in sorted(csv_files): # Sort to process chronologically (optional)
        logger.info(f"Processing file: {os.path.basename(file_path)}...")
        try:
            # Read CSV, selecting only necessary columns
            # Consider adding dtype specification if memory becomes an issue
            df_chunk = pd.read_csv(file_path, usecols=COLUMNS_TO_KEEP)

            # Rename columns
            df_chunk = df_chunk.rename(columns=COLUMN_RENAME_MAP)

            # Parse timestamp column
            try:
                # Let pandas infer format first. Add format='...' if needed.
                # df_chunk[TIMESTAMP_COL] = pd.to_datetime(df_chunk[TIMESTAMP_COL])
                # Explicitly define the format based on inspection
                df_chunk[TIMESTAMP_COL] = pd.to_datetime(df_chunk[TIMESTAMP_COL], format='%Y%m%d%H%M%S', errors='coerce')

                # Check if parsing resulted in NaT (Not a Time) for all rows
                if df_chunk[TIMESTAMP_COL].isnull().all():
                     raise ValueError("Timestamp parsing resulted in all NaT values. Check format.")
            except Exception as date_err:
                logger.error(f"Error parsing timestamp column in {os.path.basename(file_path)}: {date_err}. Skipping file.")
                continue # Skip this file if date parsing fails

            # Optional: Filter out rows with NaT timestamps if only some failed
            initial_rows = len(df_chunk)
            df_chunk = df_chunk.dropna(subset=[TIMESTAMP_COL])
            if len(df_chunk) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(df_chunk)} rows with invalid timestamps in {os.path.basename(file_path)}.")

            if df_chunk.empty:
                 logger.warning(f"No valid data remaining in {os.path.basename(file_path)} after timestamp processing.")
                 continue

            all_data.append(df_chunk)
            logger.info(f"Successfully processed {os.path.basename(file_path)} ({len(df_chunk)} rows).")

        except FileNotFoundError:
            logger.error(f"File not found during processing: {file_path}. Skipping.")
        except ValueError as ve: # Catch errors like missing columns
             logger.error(f"ValueError processing {os.path.basename(file_path)}: {ve}. Skipping.")
        except Exception as e:
            logger.error(f"Unexpected error processing {os.path.basename(file_path)}: {e}. Skipping.")

    if not all_data:
        logger.error("No data was successfully processed from any file. Cannot create consolidated output.")
        return

    # Concatenate all processed DataFrames
    logger.info("Concatenating data from all processed files...")
    df_consolidated = pd.concat(all_data, ignore_index=True)
    logger.info(f"Consolidated data has {len(df_consolidated)} total records.")

    # --- Final Checks & Saving ---
    # Sort by timestamp (important for time series)
    df_consolidated = df_consolidated.sort_values(by=[TIMESTAMP_COL, 'ID_TRAMO'])

    # Check time range
    min_ts = df_consolidated[TIMESTAMP_COL].min()
    max_ts = df_consolidated[TIMESTAMP_COL].max()
    logger.info(f"Consolidated data time range (before filtering): {min_ts} to {max_ts}")

    # Filter out data after 2023-12-31 23:59:59
    cutoff_date = pd.Timestamp("2023-12-31 23:59:59")
    initial_rows_filter = len(df_consolidated)
    df_consolidated = df_consolidated[df_consolidated[TIMESTAMP_COL] <= cutoff_date]
    filtered_rows = initial_rows_filter - len(df_consolidated)
    if filtered_rows > 0:
        logger.info(f"Filtered out {filtered_rows} records with timestamps after {cutoff_date}.")
    else:
        logger.info(f"No records found with timestamps after {cutoff_date}.")

    # Check time range again after filtering
    min_ts_filtered = df_consolidated[TIMESTAMP_COL].min()
    max_ts_filtered = df_consolidated[TIMESTAMP_COL].max()
    logger.info(f"Consolidated data time range (after filtering): {min_ts_filtered} to {max_ts_filtered}")

    # Check data types (optional but good practice)
    logger.info("Final DataFrame info:")
    df_consolidated.info(memory_usage='deep')

    # Check for duplicates (optional, depends on expected data granularity)
    # duplicates = df_consolidated.duplicated(subset=['ID_TRAMO', TIMESTAMP_COL]).sum()
    # if duplicates > 0:
    #     logger.warning(f"Found {duplicates} duplicate entries for ID_TRAMO and timestamp. Consider dropping.")
        # df_consolidated = df_consolidated.drop_duplicates(subset=['ID_TRAMO', TIMESTAMP_COL], keep='last')

    # Save to Parquet
    try:
        output_dir = os.path.dirname(OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)
        df_consolidated.to_parquet(OUTPUT_PATH, index=False)
        logger.info(f"Successfully saved consolidated data to: {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Error saving consolidated data to Parquet: {e}")

if __name__ == "__main__":
    main() 