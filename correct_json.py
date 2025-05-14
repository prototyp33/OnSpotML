import json
import os

# --- Configuration ---
# !! UPDATE THESE PATHS !!
INPUT_JSON_PATH = '/Users/adrianiraeguialvear/Desktop/OnSpotML_v2/data/raw/Information of the B-SM User Information System (SIU) of the parking prediction in the blue area in the city of Barcelona for two consecutive days.json'
OUTPUT_JSON_PATH = '/Users/adrianiraeguialvear/Desktop/OnSpotML_v2/data/raw/parking_predictions_corrected.json' # New file name
# The key under which the list of records is nested
DATA_KEY = 'OPENDATA_PSIU_APPARKB'
# --- End Configuration ---

print(f"Attempting to correct JSON structure from: {INPUT_JSON_PATH}")
print(f"Will write corrected data to: {OUTPUT_JSON_PATH}")

corrected_data_list = []
raw_data_full = None

try:
    # 1. Load the original JSON data
    print("Loading original JSON...")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f: # Added encoding
        raw_data_full = json.load(f)
    print("Original JSON loaded successfully.")

    # 2. Check structure and get the list of records
    if not isinstance(raw_data_full, dict) or DATA_KEY not in raw_data_full:
         raise ValueError(f"Expected a dictionary with top-level key '{DATA_KEY}'. Found type: {type(raw_data_full)}")

    original_data_list = raw_data_full[DATA_KEY]

    if not isinstance(original_data_list, list):
        raise ValueError(f"Expected '{DATA_KEY}' to contain a list. Found type: {type(original_data_list)}")

    print(f"Found {len(original_data_list)} items under '{DATA_KEY}'. Processing...")

    # 3. Iterate and attempt to combine pairs
    i = 0
    while i < len(original_data_list):
        record1 = original_data_list[i]
        # Check if this looks like the FH_INICIO record and if there's a next record
        if 'FH_INICIO' in record1 and 'TRAMOS' not in record1 and (i + 1) < len(original_data_list):
            record2 = original_data_list[i+1]
            # Check if the next record looks like the TRAMOS record
            if 'TRAMOS' in record2 and 'FH_INICIO' not in record2:
                print(f"  Combining record {i} (FH_INICIO: {record1.get('FH_INICIO')}) and record {i+1} (TRAMOS).")
                # Combine them
                combined_record = {
                    'FH_INICIO': record1['FH_INICIO'],
                    'TRAMOS': record2['TRAMOS'] # Take the whole TRAMOS structure
                }
                corrected_data_list.append(combined_record)
                i += 2 # Move past both records
            else:
                # Record 1 had FH_INICIO, but Record 2 didn't fit the pattern
                print(f"  WARN: Record {i} has FH_INICIO, but Record {i+1} doesn't look like a matching TRAMOS record. Skipping Record {i}.")
                # Decide how to handle this - skip record1 for now
                i += 1
        elif 'FH_INICIO' in record1 and 'TRAMOS' in record1:
             # This record already looks correct, add it directly
             print(f"  Record {i} appears correctly structured. Adding directly.")
             corrected_data_list.append(record1)
             i += 1
        else:
            # Record doesn't fit the expected pattern(s)
            print(f"  WARN: Record {i} doesn't match expected patterns (FH_INICIO only, TRAMOS only, or combined). Skipping.")
            # print(f"    Record content: {record1}") # Uncomment for more detail
            i += 1

    print(f"Finished processing. Found {len(corrected_data_list)} potentially corrected records.")

    # 4. Prepare the final structure for output
    final_corrected_structure = {
        DATA_KEY: corrected_data_list
    }

    # 5. Write the corrected data to the new file
    print(f"Writing corrected JSON to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_corrected_structure, f, indent=2, ensure_ascii=False) # Indent for readability, handle encoding
    print("Successfully wrote corrected JSON file.")

except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_JSON_PATH}")
except MemoryError:
    print("ERROR: MemoryError occurred. The JSON file is likely too large to load into memory with this script.")
    print("Consider using a streaming JSON parser like 'ijson'.")
except json.JSONDecodeError as e:
    print(f"ERROR: Failed to decode JSON from input file. Invalid JSON format: {e}")
except ValueError as e:
    print(f"ERROR: Data structure validation failed: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 