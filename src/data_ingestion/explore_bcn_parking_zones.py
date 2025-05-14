import requests
import pandas as pd
import os
import re # Import regex module

# Set pandas display options for wider columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# --- Configuration ---
TARGET_DIR = "data/raw/traffic_history" # Save history to a subfolder
# DATASET_ID_TARIFES = 'c401d753-4e78-4760-9943-fff34d90e3d0'
# DATASET_ID_SCHEDULES = 'a9f5eb1e-5aeb-40a2-8f66-c27d2a40c816'

# KNOWN_DATASET_ID = '8319c2b1-4dab-4831-9942-7547c4e9facb' # trams (Traffic Status)
SEARCH_TERM_OCCUPANCY = 'name:trams' # Search specifically by name
# FILTER_QUERY = 'groups:transports' # Commented out group filter

# Ensure the target directory exists
os.makedirs(TARGET_DIR, exist_ok=True) # Updated TARGET_DIR

# --- Function to download a resource ---
def download_resource(resource, target_dir):
    """Downloads a resource dictionary to the target directory."""
    url = resource.get('url')
    name = resource.get('name', url.split('/')[-1]) # Get name or derive from URL
    format = resource.get('format', '').lower()
    resource_id = resource.get('id')

    if not url:
        print(f"  Skipping resource {resource_id} ('{name}') - No URL found.")
        return None

    # Construct filename (use resource name if available and reasonable)
    if name and len(name) > 4 and name.lower().endswith(f'.{format}'):
         filename = name
    else:
        # Fallback to ID + format extension
        filename = f"{resource_id}.{format}" 
        print(f"  Resource name '{name}' seems invalid or missing extension, using ID: {filename}")

    filepath = os.path.join(target_dir, filename)

    print(f"  Downloading resource '{name}' (ID: {resource_id}) from: {url}")
    print(f"  Saving to: {filepath}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Successfully downloaded {filename}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {filename}: {e}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during download: {e}")
        return None

# --- Step 1: Search for the traffic dataset by name ---
print(f"Searching for dataset with query: '{SEARCH_TERM_OCCUPANCY}'...")
search_url = 'https://opendata-ajuntament.barcelona.cat/data/api/3/action/package_search'
search_params = {
    'q': SEARCH_TERM_OCCUPANCY,
    # 'fq': FILTER_QUERY, # Ensure group filter is commented out
    'rows': 1 # Expecting only one result for name search
}

found_datasets = []
selected_dataset_id = None # Reset this

try:
    response = requests.get(search_url, params=search_params)
    response.raise_for_status()
    search_results = response.json()

    if search_results.get('success'):
        datasets = search_results.get('result', {}).get('results', [])
        count = search_results.get('result', {}).get('count', 0)
        print(f"Found {count} dataset(s) matching the query.")
        if datasets:
            # Extract the ID from the first (and hopefully only) result
            ds = datasets[0]
            selected_dataset_id = ds.get('id')
            print(f"Found dataset: Name='{ds.get('name')}', ID='{selected_dataset_id}', Title='{ds.get('title')}'")
            # Store the found dataset info (optional but good practice)
            found_datasets.append(ds) 
        else:
            print("Dataset 'trams' not found via name search.")
            selected_dataset_id = None # Ensure it's None if not found
    else:
        print(f"Dataset search failed: {search_results.get('error', 'Unknown error')}")
        selected_dataset_id = None

except requests.exceptions.RequestException as e:
    print(f"Error during dataset search: {e}")
    selected_dataset_id = None
except Exception as e:
    print(f"An unexpected error occurred during search: {e}")
    selected_dataset_id = None


# --- Step 2: Get Metadata and Select Historical Resource ---
# This step will now use the selected_dataset_id found in Step 1
historical_csv_resource = None # Store the specific resource dict for download

if selected_dataset_id: # Check the ID found from the search
    print(f"\nFetching metadata for Traffic Status dataset ID: {selected_dataset_id}...")
    show_url = 'https://opendata-ajuntament.barcelona.cat/data/api/3/action/package_show'
    show_params = {'id': selected_dataset_id} # Use the ID found from search
    try:
        response = requests.get(show_url, params=show_params)
        response.raise_for_status()
        metadata = response.json()

        if metadata.get('success'):
            selected_dataset = metadata.get('result', {})
            # Use dataset_name from the result, falling back to the ID
            dataset_name = selected_dataset.get('name', selected_dataset_id)
            print(f"Successfully fetched metadata for dataset: '{dataset_name}'\n")
            resources_in_dataset = selected_dataset.get('resources', [])
            print(f"--- Resources in dataset '{dataset_name}' ---")
            if not resources_in_dataset:
                print("  No resources found in this dataset.")
            else:
                # --- Find and list specific historical CSVs --- 
                historical_files_to_download = []
                hist_pattern = re.compile(r"^(\d{4})_(\d{2})_.*?_TRAMS_TRAMS\.csv$", re.IGNORECASE)
                
                start_year, start_month = 2022, 1
                end_year, end_month = 2023, 12
                print(f"\nScanning resources for historical traffic data ({start_year:04d}-{start_month:02d} to {end_year:04d}-{end_month:02d})...")
                
                for res in resources_in_dataset:
                    res_name = res.get('name', '')
                    res_format = res.get('format', '').upper()
                    res_url = res.get('url')
                    
                    if res_format == 'CSV' and res_url: # Ensure it's CSV and has a URL
                        match = hist_pattern.match(res_name)
                        if match:
                            year = int(match.group(1))
                            month = int(match.group(2))
                             
                            # Filter for the specific date range
                            if (year > start_year or (year == start_year and month >= start_month)) and \
                               (year < end_year or (year == end_year and month <= end_month)):
                                historical_files_to_download.append(res) # Store the full resource dict
                                # print(f"    * Found relevant historical file: {res_name}") # Optional verbose logging

                # Sort the found files chronologically (optional, but good practice)
                historical_files_to_download.sort(key=lambda x: (int(re.match(hist_pattern, x['name']).group(1)), int(re.match(hist_pattern, x['name']).group(2))))
                
                print(f"\n--- Identified {len(historical_files_to_download)} historical traffic CSV files for download --- ")
                for res_info in historical_files_to_download:
                    print(f"  - {res_info['name']} (ID: {res_info['id']})")
                print("---------------------------------------------------------------------")

                # Note: The list 'historical_files_to_download' now holds the target resources for Step 3

        else:
            print(f"Failed to fetch metadata for dataset ID '{selected_dataset_id}': {metadata.get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for dataset ID '{selected_dataset_id}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred fetching metadata: {e}")
else:
    print("\nSkipping metadata fetch because no dataset was selected from search.") # Restored message


# --- Step 3: Download ALL Selected Historical Resources ---
download_count = 0
if historical_files_to_download: # Check the list identified in Step 2
    print(f"\n--- Attempting to download {len(historical_files_to_download)} historical files --- ")
    for resource_to_download in historical_files_to_download: # Loop through the identified list
        print(f"\nDownloading: {resource_to_download.get('name')}...")
        filepath = download_resource(resource_to_download, TARGET_DIR) # Use specific TARGET_DIR
        if filepath:
            print(f"Successfully downloaded to: {filepath}")
            download_count += 1
        else:
            print(f"Download FAILED for resource: {resource_to_download.get('name')}")
    print(f"\n--- Download process complete. Successfully downloaded {download_count} / {len(historical_files_to_download)} files. ---")
else:
    print("\nNo historical CSV resource files identified for download.")


# --- Step 4: (Removed File Inspection) ---
# (Code for loading/inspecting single file should be removed here)

print("\n--- Traffic History Download Complete --- ")

# Reminder for next steps
print("\n---")
print("Next steps:")
print(f"1. Check the '{TARGET_DIR}' folder for the {download_count} downloaded CSV files (covering Jan 2022 - Dec 2023). Verify the count matches the number identified ({len(historical_files_to_download)}). ")
print("2. Create a NEW script or notebook (e.g., notebooks/06-combine-traffic-history.ipynb).")
print("3. In the new script/notebook, load each downloaded CSV.")
print("4. Perform necessary processing:")
print("   - Assign correct column names.")
print("   - Convert 'DataHoraLectura' to datetime objects.")
print("   - Resample data to a fixed 5-minute frequency, forward-filling 'EstatActual' and 'PrevisioActual' to handle invariability gaps.")
print("5. Concatenate the processed monthly DataFrames into one large DataFrame.")
print("6. Save the final combined and processed traffic dataset (e.g., as a Parquet or Feather file for efficiency).")
print("---")

