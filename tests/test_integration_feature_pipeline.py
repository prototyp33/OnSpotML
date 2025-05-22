import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
import shutil # For cleaning up test output

# Assuming build_features.py is in src/features/
# Adjust the import path if your project structure is different
from src.features.build_features import build_all_features, load_data, create_temporal_features, create_weather_features, create_event_features, create_spatial_cluster_features, create_gtfs_features # Ensure this is imported
import numpy as np # For np.pi if used directly or indirectly by create_temporal_features
from shapely.geometry import Point # Required for test_create_spatial_cluster_features_logic
# No need to import DBSCAN directly for the test, but it's a dependency of the function

# --- Test Configuration ---
# Path to the root of the test sample data
TEST_SAMPLE_DATA_ROOT = Path(__file__).parent / "sample_data" / "feature_pipeline"
TEST_RAW_DATA_DIR = TEST_SAMPLE_DATA_ROOT / "raw_test_data" # Mocked RAW_DATA_DIR
TEST_PROCESSED_DATA_DIR = TEST_SAMPLE_DATA_ROOT / "processed_test_data" # Mocked PROCESSED_DATA_DIR
TEST_GTFS_DIR = TEST_RAW_DATA_DIR / "transport" # Mocked GTFS_DIR within raw

# --- Helper Functions / Fixtures ---

@pytest.fixture(scope="function") # Use "function" scope to ensure clean state for each test
def setup_test_environment(monkeypatch, tmp_path):
    '''
    Prepares the test environment before each test function execution.
    - Mocks global path configurations in build_features.py
    - Creates temporary directories for raw and processed data.
    - Copies sample data to the temporary raw data directory.
    '''
    
    # 1. Create temporary directories for mocked data paths
    # These will be subdirectories within pytest's tmp_path for easy cleanup
    mock_raw_dir = tmp_path / "raw"
    mock_processed_dir = tmp_path / "processed"
    mock_gtfs_dir_within_raw = mock_raw_dir / "transport"
    mock_weather_dir_within_raw = mock_raw_dir / "weather"
    mock_events_dir_within_raw = mock_raw_dir / "events"

    mock_raw_dir.mkdir(parents=True, exist_ok=True)
    mock_processed_dir.mkdir(parents=True, exist_ok=True)
    mock_gtfs_dir_within_raw.mkdir(parents=True, exist_ok=True)
    mock_weather_dir_within_raw.mkdir(parents=True, exist_ok=True)
    mock_events_dir_within_raw.mkdir(parents=True, exist_ok=True)

    # 2. Monkeypatch the global path variables in build_features.py
    # Ensure these paths match what build_features.py expects for its constants
    monkeypatch.setattr("src.features.build_features.BASE_DIR", tmp_path) # Or some other sensible base like tmp_path itself
    monkeypatch.setattr("src.features.build_features.RAW_DATA_DIR", mock_raw_dir)
    monkeypatch.setattr("src.features.build_features.PROCESSED_DATA_DIR", mock_processed_dir)
    monkeypatch.setattr("src.features.build_features.GTFS_DIR", mock_gtfs_dir_within_raw)
    
    # 3. Copy sample files to the mocked directories
    # Source: tests/sample_data/feature_pipeline/
    # Destination: tmp_path/raw/ or tmp_path/processed/ (as per build_features logic)

    # Parking data (goes to 'processed' in build_features logic as input, but for setup, it's like a 'raw' source)
    # build_all_features loads "parking_predictions_with_pois.parquet" from PROCESSED_DATA_DIR
    # So, we place our sample file there for the test.
    sample_parking_data = TEST_SAMPLE_DATA_ROOT / "parking_predictions_with_pois.parquet"
    shutil.copy(sample_parking_data, mock_processed_dir / "parking_predictions_with_pois.parquet")

    # Weather data
    sample_weather_data = TEST_SAMPLE_DATA_ROOT / "realtime_weather.json"
    shutil.copy(sample_weather_data, mock_weather_dir_within_raw / "realtime_weather.json")

    # Events data
    sample_events_data = TEST_SAMPLE_DATA_ROOT / "cultural_events.csv"
    shutil.copy(sample_events_data, mock_events_dir_within_raw / "cultural_events.csv")

    # GTFS data (stops.txt)
    sample_gtfs_stops = TEST_SAMPLE_DATA_ROOT / "gtfs" / "stops.txt"
    shutil.copy(sample_gtfs_stops, mock_gtfs_dir_within_raw / "stops.txt")

    yield mock_raw_dir, mock_processed_dir # Provide these paths to the test if needed

    # Teardown (handled by tmp_path fixture automatically)
    # print(f"Test environment cleaned up: {tmp_path}")


# --- Test Cases ---

def test_build_all_features_end_to_end(setup_test_environment):
    '''
    Test the successful end-to-end execution of build_all_features.
    '''
    mock_raw_dir, mock_processed_dir = setup_test_environment

    # Run the main feature building function
    # It will use the monkeypatched paths for input and output
    master_df = build_all_features()

    # 1. Check if the function returned a DataFrame
    assert master_df is not None, "build_all_features() should return a DataFrame."
    assert isinstance(master_df, pd.DataFrame), "Output should be a pandas DataFrame."

    # 2. Check if the main output file was created
    expected_output_file = mock_processed_dir / "features_master_table.parquet"
    assert expected_output_file.exists(), f"Output file {expected_output_file} was not created."

    # 3. Load the output file and verify its content
    loaded_master_df = pd.read_parquet(expected_output_file)
    assert not loaded_master_df.empty, "The saved master feature table should not be empty."
    
    # Compare shape or row count with the input sample parking data
    # (Assuming parking_predictions_with_pois.parquet has 4 rows from our sample data script)
    # This might need adjustment based on how build_all_features handles data.
    # For now, let's assume it processes all 4 rows from the sample.
    # The number of columns will be original + new features.
    assert len(loaded_master_df) == 4, "Number of rows in output should match sample input."

    # 4. Check for presence of key columns from different feature groups
    expected_columns = [
        'ID_TRAMO', 'timestamp', 'geometry', # Base columns from input
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_public_holiday', # Temporal
        'temp_c', 'precip_mm', 'wind_kph', 'humidity', # Weather
        'is_event_ongoing', # Events
        'cluster_label', # Spatial (ensure this is robust, DBSCAN can be sensitive)
        'distance_nearest_stop', 'bus_stop_density_500m' # GTFS
    ]
    for col in expected_columns:
        assert col in loaded_master_df.columns, f"Expected column '{col}' not found in output."
        
    # 5. Check data types for a few representative columns (optional, but good practice)
    assert pd.api.types.is_integer_dtype(loaded_master_df['hour']), "'hour' should be integer."
    assert pd.api.types.is_float_dtype(loaded_master_df['temp_c']), "'temp_c' should be float."
    assert pd.api.types.is_integer_dtype(loaded_master_df['is_event_ongoing']), "'is_event_ongoing' should be integer (0 or 1)."
    if 'geometry' in loaded_master_df.columns and isinstance(loaded_master_df, gpd.GeoDataFrame):
         assert loaded_master_df['geometry'].geom_type.isin(['Point', 'MultiPoint']).all() # Or whatever geometry type is expected


    # 6. Check if intermediate feature set files were created
    intermediate_files = [
        "temporal_features_set.parquet",
        "weather_features_set.parquet",
        "event_features_set.parquet",
        "spatial_cluster_features_set.parquet",
        "gtfs_features_set.parquet"
    ]
    for fname in intermediate_files:
        assert (mock_processed_dir / fname).exists(), f"Intermediate file {fname} was not created."

    print("test_build_all_features_end_to_end PASSED")

# Placeholder for more tests - to be added in subsequent steps
# def test_create_temporal_features_logic():
#     pass

def test_create_temporal_features_logic():
    '''
    Tests the logic of create_temporal_features directly.
    Verifies correct generation of time-based features from a sample DataFrame.
    '''
    # 1. Prepare sample input DataFrame
    sample_data = {
        'timestamp': pd.to_datetime([
            '2023-01-01 00:00:00', # Sunday, New Year's Day (a holiday)
            '2023-03-15 10:30:00', # Wednesday
            '2023-07-04 14:00:00', # Tuesday, US Independence Day (test with ES holidays, should not be holiday)
            '2023-12-24 23:59:59', # Sunday, Christmas Eve
            '2024-02-29 06:15:00'  # Thursday, Leap Year
        ])
    }
    input_df = pd.DataFrame(sample_data)

    # 2. Call the function
    # The function expects the input DataFrame to have the timestamp column, 
    # and it returns only the new temporal features, indexed like the input.
    temporal_features_df = create_temporal_features(input_df, timestamp_col='timestamp')

    # 3. Assertions
    assert not temporal_features_df.empty
    assert len(temporal_features_df) == len(input_df)

    # Check for expected columns
    expected_temporal_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'year',
        'is_weekend', 'is_weekday',
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'day_of_year', 'week_of_year', 'quarter',
        'time_of_day_segment', 'is_public_holiday'
    ]
    for col in expected_temporal_cols:
        assert col in temporal_features_df.columns, f"Column '{col}' missing in temporal features."

    # Check specific values for the first row (2023-01-01 00:00:00, Sunday, New Year)
    assert temporal_features_df.loc[0, 'hour'] == 0
    assert temporal_features_df.loc[0, 'day_of_week'] == 6 # Sunday is 6 if Monday is 0
    assert temporal_features_df.loc[0, 'day_of_month'] == 1
    assert temporal_features_df.loc[0, 'month'] == 1
    assert temporal_features_df.loc[0, 'year'] == 2023
    assert temporal_features_df.loc[0, 'is_weekend'] == 1
    assert temporal_features_df.loc[0, 'is_weekday'] == 0
    assert temporal_features_df.loc[0, 'time_of_day_segment'] == 'night'
    # New Year's Day is a public holiday in Spain ('ES')
    assert temporal_features_df.loc[0, 'is_public_holiday'] == 1 

    # Check specific values for the second row (2023-03-15 10:30:00, Wednesday)
    assert temporal_features_df.loc[1, 'hour'] == 10
    assert temporal_features_df.loc[1, 'day_of_week'] == 2 # Wednesday is 2
    assert temporal_features_df.loc[1, 'month'] == 3
    assert temporal_features_df.loc[1, 'is_weekend'] == 0
    assert temporal_features_df.loc[1, 'time_of_day_segment'] == 'morning'
    # March 15th is generally not a national holiday in Spain
    assert temporal_features_df.loc[1, 'is_public_holiday'] == 0

    # Check cyclical features (rough check for sin/cos presence and plausible range)
    assert -1 <= temporal_features_df.loc[0, 'hour_sin'] <= 1
    assert -1 <= temporal_features_df.loc[0, 'hour_cos'] <= 1
    
    # Check leap year handling for day_of_year
    # 2024-02-29 is the 31+29 = 60th day of the year
    assert temporal_features_df.loc[4, 'year'] == 2024
    assert temporal_features_df.loc[4, 'day_of_year'] == 60
    assert temporal_features_df.loc[4, 'is_public_holiday'] == 0 # Feb 29 is not a holiday

    print("test_create_temporal_features_logic PASSED")

def test_create_weather_features_logic():
    '''
    Tests the logic of create_weather_features directly.
    Verifies correct extraction and alignment of weather features.
    '''
    # 1. Prepare sample input data
    sample_weather_data_dict = {
        "forecast": {
            "forecastday": [
                {
                    "date": "2023-03-15",
                    "hour": [
                        {"time_epoch": 1678863600, "time": "2023-03-15 08:00", "temp_c": 12.0, "precip_mm": 0.1, "wind_kph": 5.0, "humidity": 70, "cloud": 10, "condition": {"text": "Light rain"}},
                        {"time_epoch": 1678867200, "time": "2023-03-15 09:00", "temp_c": 14.0, "precip_mm": 0.0, "wind_kph": 6.0, "humidity": 65, "cloud": 5, "condition": {"text": "Sunny"}},
                        {"time_epoch": 1678870800, "time": "2023-03-15 10:00", "temp_c": 16.0, "precip_mm": 0.0, "wind_kph": 7.0, "humidity": 60, "cloud": 20, "condition": {"text": "Partly cloudy"}}
                    ]
                }
            ]
        }
    }
    
    # Timestamps to align weather data against
    # Note: Timestamps in build_features.py's create_weather_features are localized to Europe/Madrid
    target_timestamps_data = {
        'timestamp': pd.to_datetime([
            '2023-03-15 08:05:00', # Nearest to 08:00 weather
            '2023-03-15 08:55:00', # Nearest to 09:00 weather
            '2023-03-15 10:10:00', # Nearest to 10:00 weather
            '2023-03-15 11:00:00'  # No weather data for 11:00, should take nearest (10:00)
        ])
        # The function itself handles timezone localization if naive, or conversion if different.
        # So, we can pass naive timestamps here to test that part of its logic.
    }
    target_df = pd.DataFrame(target_timestamps_data)

    # 2. Call the function
    # create_weather_features(weather_data: dict, target_timestamps_df: pd.DataFrame, timestamp_col:str = 'timestamp')
    weather_features_df = create_weather_features(sample_weather_data_dict, target_df, timestamp_col='timestamp')

    # 3. Assertions
    assert not weather_features_df.empty
    assert len(weather_features_df) == len(target_df)

    expected_weather_cols = ['temp_c', 'precip_mm', 'wind_kph', 'humidity', 'cloud_cover', 'condition_text']
    for col in expected_weather_cols:
        assert col in weather_features_df.columns, f"Column '{col}' missing in weather features."

    # Check values based on 'nearest' merge_asof logic
    # Target '2023-03-15 08:05:00' should get weather from '2023-03-15 08:00'
    assert weather_features_df.loc[0, 'temp_c'] == 12.0
    assert weather_features_df.loc[0, 'condition_text'] == "Light rain"

    # Target '2023-03-15 08:55:00' should get weather from '2023-03-15 09:00'
    assert weather_features_df.loc[1, 'temp_c'] == 14.0
    assert weather_features_df.loc[1, 'condition_text'] == "Sunny"
    
    # Target '2023-03-15 10:10:00' should get weather from '2023-03-15 10:00'
    assert weather_features_df.loc[2, 'temp_c'] == 16.0
    assert weather_features_df.loc[2, 'condition_text'] == "Partly cloudy"

    # Target '2023-03-15 11:00:00' should also get weather from '2023-03-15 10:00' (nearest)
    assert weather_features_df.loc[3, 'temp_c'] == 16.0
    assert weather_features_df.loc[3, 'condition_text'] == "Partly cloudy"
    
    # Test with empty weather data input
    empty_weather_data = {}
    empty_weather_features_df = create_weather_features(empty_weather_data, target_df, timestamp_col='timestamp')
    assert empty_weather_features_df.empty or empty_weather_features_df.isnull().all().all()
    # Depending on implementation, it might return an empty DF or DF with all NaNs but correct columns/index.
    # The current build_features.py implementation returns an empty DF if weather_data is empty.
    # If it returned a DF with NaNs, we'd check:
    # assert len(empty_weather_features_df) == len(target_df)
    # assert empty_weather_features_df['temp_c'].isnull().all()


    # Test with target_df that has no timestamp_col
    target_df_no_ts = target_df.rename(columns={'timestamp': 'some_other_time'})
    weather_features_no_ts_col = create_weather_features(sample_weather_data_dict, target_df_no_ts, timestamp_col='timestamp')
    assert weather_features_no_ts_col.empty # Should return empty as per build_features.py logic

    print("test_create_weather_features_logic PASSED")

def test_create_event_features_logic():
    '''
    Tests the logic of create_event_features directly.
    Verifies correct identification of ongoing events based on sample event data and target timestamps.
    '''
    # 1. Prepare sample input data
    sample_events_list = [
        # Event 1: Specific start and end time
        {'DataInici': '2023-03-15T10:00:00', 'DataFi': '2023-03-15T18:00:00', 'Nom': 'Concert A'},
        # Event 2: Spans multiple full days
        {'DataInici': '2023-03-20T00:00:00', 'DataFi': '2023-03-22T23:59:59', 'Nom': 'Festival B'},
        # Event 3: Single full day (DataFi might be just date, function should handle EOD)
        {'DataInici': '2023-03-25', 'DataFi': '2023-03-25', 'Nom': 'Market C'},
        # Event 4: Event with time component crossing midnight
        {'DataInici': '2023-03-28T22:00:00', 'DataFi': '2023-03-29T02:00:00', 'Nom': 'Night Event D'}
    ]
    events_df = pd.DataFrame(sample_events_list)
    # Convert to datetime as they would be after initial loading and basic processing
    # create_event_features itself also does pd.to_datetime
    events_df['DataInici'] = pd.to_datetime(events_df['DataInici'])
    events_df['DataFi'] = pd.to_datetime(events_df['DataFi'])


    target_timestamps_list = [
        '2023-03-15T09:00:00', # Before Event 1
        '2023-03-15T12:00:00', # During Event 1
        '2023-03-15T18:00:00', # Exactly at Event 1 end
        '2023-03-15T19:00:00', # After Event 1
        '2023-03-21T12:00:00', # During Festival B
        '2023-03-25T15:00:00', # During Market C (assuming DataFi '2023-03-25' means end of day)
        '2023-03-28T23:00:00', # During Night Event D (first day part)
        '2023-03-29T01:00:00', # During Night Event D (second day part)
        '2023-04-01T10:00:00'  # No event
    ]
    # The function expects a Series for target_timestamps_series
    target_timestamps_series = pd.Series(pd.to_datetime(target_timestamps_list))
    # The function handles timezone localization, so naive is fine here.

    # 2. Call the function
    event_features_df = create_event_features(events_df, target_timestamps_series)

    # 3. Assertions
    assert not event_features_df.empty
    assert len(event_features_df) == len(target_timestamps_series)
    assert 'is_event_ongoing' in event_features_df.columns
    assert event_features_df['is_event_ongoing'].dtype == 'int'

    # Expected outcomes for 'is_event_ongoing' (0 or 1)
    # Index corresponds to target_timestamps_list
    expected_values = [
        0, # Before Event 1
        1, # During Event 1
        1, # Exactly at Event 1 end (inclusive end)
        0, # After Event 1
        1, # During Festival B
        1, # During Market C
        1, # During Night Event D (first day part)
        1, # During Night Event D (second day part)
        0  # No event
    ]
    
    assert event_features_df['is_event_ongoing'].tolist() == expected_values

    # Test with empty events_df
    empty_events_df = pd.DataFrame(columns=['DataInici', 'DataFi', 'Nom'])
    empty_event_features = create_event_features(empty_events_df, target_timestamps_series)
    assert len(empty_event_features) == len(target_timestamps_series)
    assert 'is_event_ongoing' in empty_event_features.columns
    assert empty_event_features['is_event_ongoing'].sum() == 0 # Should all be 0

    # Test with events_df missing required columns (should return all 0s and log error)
    malformed_events_df = pd.DataFrame({'Name': ['Event X']})
    malformed_event_features = create_event_features(malformed_events_df, target_timestamps_series)
    assert len(malformed_event_features) == len(target_timestamps_series)
    assert 'is_event_ongoing' in malformed_event_features.columns
    assert malformed_event_features['is_event_ongoing'].sum() == 0

    print("test_create_event_features_logic PASSED")

def test_create_spatial_cluster_features_logic():
    '''
    Tests the logic of create_spatial_cluster_features directly.
    Verifies correct spatial clustering of sample GeoDataFrame.
    '''
    # 1. Prepare sample input GeoDataFrame
    # Create points that should form a few clusters and some noise points
    # Using simple integer coordinates for easy understanding; will be projected in function
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        # Cluster 1 (Points 1, 2, 3) - using WGS84-like coordinates
        'latitude':  [40.7128, 40.7129, 40.7130, 
        # Cluster 2 (Points 4, 5)
                       40.7580, 40.7581,
        # Noise Points (Points 6, 7, 8) - further apart
                       40.8000, 40.8500, 34.0522], 
        'longitude': [-74.0060, -74.0061, -74.0059,
                       -73.9850, -73.9851,
                       -73.9000, -73.8000, -118.2437] 
    }
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
        crs="EPSG:4326" # Set initial CRS to WGS84
    )

    # 2. Call the function
    # create_spatial_cluster_features(gdf, geometry_col, id_col, target_crs, eps, min_samples)
    # Using a relatively large eps for WGS84 projected to a meter-based CRS.
    # The function projects to EPSG:25831 (UTM zone 31N, meter-based) by default.
    # eps is in the units of the target_crs. 100 meters for this example.
    # min_samples=2 means two points can form a cluster.
    cluster_features_df = create_spatial_cluster_features(
        gdf, 
        geometry_col='geometry', 
        id_col='id',
        target_crs="EPSG:25831", # Default, but explicit for clarity
        eps=1000, # Adjusted eps (in meters) for points that are ~0.0001 degrees apart
                  # (approx 10m, so 1km eps should group them if close)
        min_samples=2 
    )
    
    # 3. Assertions
    assert not cluster_features_df.empty
    assert 'id' in cluster_features_df.columns
    assert 'cluster_label' in cluster_features_df.columns
    assert len(cluster_features_df) == len(gdf) # Should return labels for all input points

    # Check cluster labels. This can be a bit tricky as exact labels can vary.
    # We expect points 1,2,3 to be in one cluster, and 4,5 in another.
    # Noise points 6,7,8 should ideally be -1 or in separate small clusters if eps is too large.
    
    # Get labels for specific groups of points
    labels_group1 = cluster_features_df[cluster_features_df['id'].isin([1, 2, 3])]['cluster_label']
    labels_group2 = cluster_features_df[cluster_features_df['id'].isin([4, 5])]['cluster_label']
    labels_noise = cluster_features_df[cluster_features_df['id'].isin([6, 7, 8])]['cluster_label']

    # Assert all points in group1 have the same, non-noise label
    assert len(labels_group1.unique()) == 1, "Group 1 points should be in the same cluster."
    assert labels_group1.iloc[0] != -1, "Group 1 cluster label should not be noise."

    # Assert all points in group2 have the same, non-noise label
    assert len(labels_group2.unique()) == 1, "Group 2 points should be in the same cluster."
    assert labels_group2.iloc[0] != -1, "Group 2 cluster label should not be noise."

    # Assert group1 and group2 are in different clusters
    assert labels_group1.iloc[0] != labels_group2.iloc[0], "Group 1 and Group 2 should be in different clusters."

    # Assert noise points are marked as noise (-1) or are not part of the main clusters
    # With min_samples=2, individual points far away should be -1.
    # If eps is very large, they might get grouped. The key is they are different from cluster 1 and 2.
    for label in labels_noise:
        assert label == -1 or (label != labels_group1.iloc[0] and label != labels_group2.iloc[0]),             f"Noise point has label {label}, conflicting with main clusters or not being -1."
            
    # Test with empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame({'id': [], 'geometry': []}, crs="EPSG:4326")
    empty_cluster_df = create_spatial_cluster_features(empty_gdf, id_col='id', eps=100, min_samples=2)
    assert empty_cluster_df.empty or 'cluster_label' not in empty_cluster_df # Depending on implementation for empty

    # Test with insufficient samples for min_samples
    # (e.g., 1 point when min_samples=2, should result in noise label -1)
    single_point_data = {'id': [10], 'latitude': [40.0], 'longitude': [-74.0]}
    single_point_gdf = gpd.GeoDataFrame(single_point_data, geometry=gpd.points_from_xy(single_point_data['longitude'], single_point_data['latitude']), crs="EPSG:4326")
    single_point_clusters = create_spatial_cluster_features(single_point_gdf, id_col='id', eps=100, min_samples=2)
    assert len(single_point_clusters) == 1
    assert single_point_clusters.loc[0, 'cluster_label'] == -1 # Should be noise

    print("test_create_spatial_cluster_features_logic PASSED")

def test_create_gtfs_features_logic(setup_test_environment):
    '''
    Tests the logic of create_gtfs_features.
    Verifies correct calculation of distance to nearest stop and stop density.
    Uses the setup_test_environment fixture to ensure GTFS data is available at the mocked path.
    '''
    mock_raw_dir, mock_processed_dir = setup_test_environment
    # The GTFS_DIR in build_features.py is monkeypatched to: tmp_path / "raw" / "transport"
    # And setup_test_environment copies sample_data/feature_pipeline/gtfs/stops.txt there.
    
    # 1. Prepare sample input parking GeoDataFrame
    # Points are in WGS84 (EPSG:4326). The function will reproject.
    # Stop S1: 41.390205, -73.984472 (from sample stops.txt, roughly)
    # Stop S2: 41.391205, -73.985472
    # Stop S3: 40.758896, -73.985130
    parking_data = {
        'parking_id': [1, 2, 3],
        # Point near Stop S1
        'latitude': [41.390200, 
        # Point between Stop S1 and S2, but closer to S2
                     41.390800, 
        # Point far from S1/S2, but potentially near S3 if S3 was closer. For this test, it's far from S1/S2.
                     40.750000], 
        'longitude': [-73.984400, 
                      -73.985000, 
                      -73.980000]  
    }
    parking_gdf = gpd.GeoDataFrame(
        parking_data,
        geometry=gpd.points_from_xy(parking_data['longitude'], parking_data['latitude']),
        crs="EPSG:4326"
    )

    # 2. Call the function
    # create_gtfs_features(parking_gdf: gpd.GeoDataFrame, gtfs_dir: Path, geometry_col: str = 'geometry')
    # gtfs_dir will be taken from the monkeypatched src.features.build_features.GTFS_DIR
    # The monkeypatched GTFS_DIR is (tmp_path / "raw" / "transport")
    gtfs_features_df = create_gtfs_features(parking_gdf, Path(mock_raw_dir / "transport"))

    # 3. Assertions
    assert not gtfs_features_df.empty
    assert len(gtfs_features_df) == len(parking_gdf)
    
    expected_gtfs_cols = ['distance_nearest_stop', 'bus_stop_density_500m']
    for col in expected_gtfs_cols:
        assert col in gtfs_features_df.columns, f"Column '{col}' missing in GTFS features."

    # Check values (these are highly dependent on the exact coordinates and CRS used for calculations)
    # Distances are in meters as the function projects to EPSG:25831.
    # Point 1 (41.390200, -73.984400) vs Stop S1 (41.390205, -73.984472) ~ very close
    # Point 2 (41.390800, -73.985000) vs Stop S2 (41.391205, -73.985472) ~ some distance, closer to S2 than S1
    # Point 3 (40.750000, -73.980000) vs any of S1, S2, S3 ~ very far

    # Rough checks for distances:
    assert gtfs_features_df.loc[0, 'distance_nearest_stop'] < 100  # Point 1 should be very close to S1
    assert gtfs_features_df.loc[1, 'distance_nearest_stop'] < gtfs_features_df.loc[0, 'distance_nearest_stop'] + 1000 # P2 is further than P1 from S1, but maybe closer to S2. This is a loose check.
    # A more precise check would require calculating expected distances manually or with a GIS tool.
    # For now, ensure values are populated and are plausible (not NaN unless expected).
    assert pd.notna(gtfs_features_df.loc[0, 'distance_nearest_stop'])
    assert pd.notna(gtfs_features_df.loc[1, 'distance_nearest_stop'])
    assert pd.notna(gtfs_features_df.loc[2, 'distance_nearest_stop'])

    # Check stop density (number of stops within 500m)
    # Based on sample stops.txt: S1, S2 are close. S3 is far.
    # Parking point 1 is near S1. Parking point 2 is near S1 & S2. Parking point 3 is far from all.
    # This depends heavily on the exact locations and the 500m buffer.
    # Stop S1: 41.390205, -73.984472
    # Stop S2: 41.391205, -73.985472 (approx 130m from S1)
    # Parking 1 (near S1): should find at least S1. If S2 is within 500m of Parking 1, then 2.
    # Parking 2 (between S1 and S2): should find S1 and S2.
    # Parking 3 (far): should find 0.
    
    # Given the coordinates, S1 and S2 are very close (<< 500m apart).
    # Parking 1 is very close to S1. So S1 is in buffer. S2 might also be.
    # Parking 2 is between S1 and S2. So both S1 and S2 should be in buffer.
    # Parking 3 is very far. So 0 stops in buffer.
    
    # For point 1, S1 is very close. S2 is ~130m from S1. So S2 should also be in the 500m buffer of point 1.
    assert gtfs_features_df.loc[0, 'bus_stop_density_500m'] >= 1 # At least S1, likely S2 too
    # For point 2, which is between S1 and S2.
    assert gtfs_features_df.loc[1, 'bus_stop_density_500m'] >= 2 # Should find both S1 and S2
    assert gtfs_features_df.loc[2, 'bus_stop_density_500m'] == 0 # Far from S1, S2, S3

    # Test with parking_gdf having no geometry
    parking_no_geom_df = pd.DataFrame({'parking_id': [1]})
    gtfs_no_geom = create_gtfs_features(parking_no_geom_df, Path(mock_raw_dir / "transport"))
    assert gtfs_no_geom['distance_nearest_stop'].isnull().all() or gtfs_no_geom['distance_nearest_stop'].empty # Handled by check for geometry col
    assert gtfs_no_geom['bus_stop_density_500m'].eq(0).all() or gtfs_no_geom['bus_stop_density_500m'].empty


    # Test with non-existent GTFS directory (or missing stops.txt)
    # Create a new empty temp dir for this specific case
    with pytest.raises(Exception): # Or check for logged error and default output
        # The function currently logs an error and returns default (NaN/0) features if stops.txt not found.
        # It doesn't raise an exception itself, but an internal operation might if path is totally invalid.
        # For a more robust test, check logs or ensure default output.
        # For now, let's assume it produces default output as per its error handling.
        invalid_gtfs_dir = mock_raw_dir / "non_existent_gtfs"
        invalid_gtfs_dir.mkdir() # Create an empty dir
        gtfs_bad_path_df = create_gtfs_features(parking_gdf, invalid_gtfs_dir)
        assert gtfs_bad_path_df['distance_nearest_stop'].isnull().all()
        assert gtfs_bad_path_df['bus_stop_density_500m'].eq(0).all()


    print("test_create_gtfs_features_logic PASSED")

# ... and so on for other feature functions
