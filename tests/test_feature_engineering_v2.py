"""
Tests for the FeatureEngineeringPipeline class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yaml
from shapely.geometry import Point
import geopandas as gpd
from src.modeling.feature_engineering_v2 import FeatureEngineeringPipeline

# Test data fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='5min')
    parking_ids = ['P1', 'P2']
    
    data = []
    for date in dates:
        for parking_id in parking_ids:
            data.append({
                'timestamp': date,
                'parking_id': parking_id,
                'available_spaces': np.random.randint(0, 100),
                'total_spaces': 100
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_poi_data():
    """Create sample POI data for testing."""
    pois = []
    for i in range(100):
        pois.append({
            'poi_id': f'POI_{i}',
            'name': f'POI {i}',
            'category': np.random.choice(['restaurant', 'shop', 'entertainment', 'transport', 'other']),
            'latitude': np.random.uniform(40.0, 41.0),
            'longitude': np.random.uniform(-74.0, -73.0),
            'importance': np.random.choice([1, 2, 3])
        })
    return pd.DataFrame(pois)

@pytest.fixture
def config():
    """Create test configuration."""
    return {
        'feature_engineering': {
            'temporal_features': {
                'enabled': True,
                'cyclic_encoding': True,
                'holiday_features': True
            },
            'lag_features': {
                'enabled': True,
                'windows': [5, 15, 30, 60],  # minutes
                'features': ['available_spaces', 'occupancy_rate']
            },
            'poi_features': {
                'enabled': True,
                'radii': [100, 200, 500],  # meters
                'categories': ['restaurant', 'shop', 'entertainment', 'transport'],
                'cache_dir': 'tests/cache'
            },
            'facility_features': {
                'enabled': True,
                'static_features': True,
                'dynamic_features': True
            },
            'weather_features': {
                'enabled': True,
                'features': ['temperature', 'precipitation', 'wind_speed']
            },
            'event_features': {
                'enabled': True,
                'features': ['is_event', 'event_type', 'event_importance']
            }
        }
    }

@pytest.fixture
def pipeline():
    """Create a FeatureEngineeringPipeline instance."""
    return FeatureEngineeringPipeline()

def test_temporal_features(pipeline, sample_data):
    """Test temporal feature creation."""
    df = pipeline.create_temporal_features(sample_data)
    
    # Check if temporal features are created
    assert 'hour' in df.columns
    assert 'dayofweek' in df.columns
    assert 'month' in df.columns
    assert 'hour_sin' in df.columns
    assert 'hour_cos' in df.columns
    assert 'is_weekend' in df.columns
    assert 'is_public_holiday' in df.columns
    
    # Check feature ranges
    assert df['hour'].min() >= 0
    assert df['hour'].max() <= 23
    assert df['dayofweek'].min() >= 0
    assert df['dayofweek'].max() <= 6
    assert df['month'].min() >= 1
    assert df['month'].max() <= 12
    assert df['is_weekend'].isin([0, 1]).all()
    assert df['is_public_holiday'].isin([0, 1]).all()

def test_lag_features(pipeline, sample_data):
    """Test lag feature creation."""
    df = pipeline.create_lag_features(sample_data)
    
    # Check if lag features are created
    for lag in [1, 2, 3, 6, 12, 24]:
        assert f'occupancy_rate_lag_{lag}' in df.columns
        assert f'available_spaces_lag_{lag}' in df.columns
    
    # Check if occupancy rate is calculated
    assert 'occupancy_rate' in df.columns
    assert df['occupancy_rate'].between(0, 1).all()

def test_poi_features(pipeline, sample_data, sample_poi_data, tmp_path):
    """Test POI feature creation."""
    print("\n=== Debug: test_poi_features ===")
    print(f"Initial sample_data shape: {sample_data.shape}")
    print(f"Initial sample_poi_data shape: {sample_poi_data.shape}")
    
    # Add required columns to sample data
    sample_data['latitude'] = np.random.uniform(40.0, 41.0, len(sample_data))
    sample_data['longitude'] = np.random.uniform(-74.0, -73.0, len(sample_data))
    print(f"Sample data columns after adding location: {sample_data.columns.tolist()}")
    
    # Update config with temporary cache directory
    test_config = {
        'feature_engineering': {
            'poi_features': {
                'radii': [100, 200, 500],
                'categories': ['restaurant', 'shop', 'entertainment', 'transport'],
                'cache_dir': str(tmp_path)
            }
        }
    }
    print(f"Test config: {test_config}")
    
    pipeline = FeatureEngineeringPipeline(test_config)
    print(f"Pipeline config: {pipeline.config}")
    
    try:
        df = pipeline.create_poi_features(sample_data, sample_poi_data)
        print(f"Output DataFrame shape: {df.shape}")
        print(f"Output DataFrame columns: {df.columns.tolist()}")
        
        # Check if POI features are created for each radius
        for radius in test_config['feature_engineering']['poi_features']['radii']:
            for category in test_config['feature_engineering']['poi_features']['categories']:
                feature_name = f'poi_{category}_count_{radius}m'
                print(f"Checking feature: {feature_name}")
                assert feature_name in df.columns, f"Missing feature: {feature_name}"
                print(f"Feature {feature_name} exists with values: {df[feature_name].describe()}")
        
        # Check value ranges
        for col in df.columns:
            if col.startswith('poi_'):
                print(f"Checking value range for {col}")
                print(f"Min: {df[col].min()}, Max: {df[col].max()}")
                assert df[col].min() >= 0, f"Negative values found in {col}"
                if 'density' in col:
                    assert df[col].max() <= 1, f"Values > 1 found in {col}"
        
        # Test without POI data
        print("\nTesting without POI data...")
        df_no_poi = pipeline.create_poi_features(sample_data)
        print(f"Shape without POI: {df_no_poi.shape}")
        print(f"Columns without POI: {df_no_poi.columns.tolist()}")
        assert len(df_no_poi) == len(sample_data)
        assert all(col not in df_no_poi.columns for col in df.columns if col.startswith('poi_'))
        
        # Test with missing location columns
        print("\nTesting with missing location columns...")
        df_no_loc = sample_data.drop(['latitude', 'longitude'], axis=1)
        print(f"Shape without location: {df_no_loc.shape}")
        print(f"Columns without location: {df_no_loc.columns.tolist()}")
        df_no_loc = pipeline.create_poi_features(df_no_loc, sample_poi_data)
        assert len(df_no_loc) == len(sample_data)
        assert all(col not in df_no_loc.columns for col in df.columns if col.startswith('poi_'))
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        raise

def test_facility_features(pipeline, sample_data):
    """Test facility feature creation."""
    df = pipeline.create_facility_features(sample_data)
    
    # Check if facility features are created
    assert 'is_open_now' in df.columns
    assert 'hours_until_close' in df.columns
    assert 'hours_since_open' in df.columns

def test_edge_cases(pipeline):
    """Test edge cases in feature creation."""
    print("\n=== Debug: test_edge_cases ===")
    
    # Test with empty DataFrame
    print("\nTesting empty DataFrame...")
    empty_df = pd.DataFrame({
        'timestamp': pd.to_datetime([]),
        'parking_id': [],
        'available_spaces': [],
        'total_spaces': []
    })
    print(f"Empty DataFrame shape: {empty_df.shape}")
    print(f"Empty DataFrame columns: {empty_df.columns.tolist()}")
    df = pipeline.create_features(empty_df)
    print(f"Result shape: {df.shape}")
    assert len(df) == 0
    
    # Test with missing columns
    print("\nTesting missing columns...")
    missing_cols_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01']),
        'parking_id': ['P1'],
        'available_spaces': [50],
        'total_spaces': [100]
    })
    missing_cols_df = missing_cols_df.drop('available_spaces', axis=1)
    print(f"Missing columns DataFrame shape: {missing_cols_df.shape}")
    print(f"Missing columns DataFrame columns: {missing_cols_df.columns.tolist()}")
    try:
        pipeline.create_features(missing_cols_df)
    except ValueError as e:
        print(f"Expected ValueError: {str(e)}")
        raise
    
    # Test with invalid data types
    print("\nTesting invalid data types...")
    invalid_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01']),
        'parking_id': ['P1'],
        'available_spaces': ['invalid'],  # Invalid type
        'total_spaces': ['invalid']  # Invalid type
    })
    print(f"Invalid types DataFrame:\n{invalid_df.dtypes}")
    try:
        pipeline.create_features(invalid_df)
    except ValueError as e:
        print(f"Expected ValueError: {str(e)}")
        raise
    
    # Test with invalid value ranges
    print("\nTesting invalid value ranges...")
    invalid_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01']),
        'parking_id': ['P1'],
        'available_spaces': [-1],  # Negative value
        'total_spaces': [100]
    })
    print(f"Invalid values DataFrame:\n{invalid_df}")
    try:
        pipeline.create_features(invalid_df)
    except ValueError as e:
        print(f"Expected ValueError: {str(e)}")
        raise
    
    # Test with available_spaces > total_spaces
    print("\nTesting available_spaces > total_spaces...")
    invalid_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01']),
        'parking_id': ['P1'],
        'available_spaces': [200],  # Greater than total_spaces
        'total_spaces': [100]
    })
    print(f"Invalid ratio DataFrame:\n{invalid_df}")
    try:
        pipeline.create_features(invalid_df)
    except ValueError as e:
        print(f"Expected ValueError: {str(e)}")
        raise

def test_data_validation(pipeline, sample_data):
    """Test data validation in feature creation."""
    # Test with negative values
    invalid_df = sample_data.copy()
    invalid_df.loc[0, 'available_spaces'] = -1
    with pytest.raises(ValueError):
        pipeline.create_features(invalid_df)
    
    # Test with available_spaces > total_spaces
    invalid_df = sample_data.copy()
    invalid_df.loc[0, 'available_spaces'] = 200
    with pytest.raises(ValueError):
        pipeline.create_features(invalid_df)

def test_feature_engineering_pipeline(pipeline, sample_data):
    """Test the complete feature engineering pipeline."""
    df = pipeline.create_features(sample_data)
    
    # Check if all feature groups are created
    assert 'hour_sin' in df.columns  # temporal
    assert 'occupancy_rate_lag_1' in df.columns  # lag
    assert 'is_open_now' in df.columns  # facility
    
    # Check if data is not modified in place
    assert len(sample_data) == len(df)
    assert 'hour_sin' not in sample_data.columns

def test_performance(pipeline, sample_data):
    """Test performance of feature creation."""
    import time
    
    # Measure time for feature creation
    start_time = time.time()
    df = pipeline.create_features(sample_data)
    end_time = time.time()
    
    # Check if feature creation is reasonably fast
    assert end_time - start_time < 1.0  # Should take less than 1 second
    
    # Check memory usage
    assert df.memory_usage().sum() < 1e6  # Should use less than 1MB 