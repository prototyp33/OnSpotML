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
    dates = pd.date_range(start='2024-01-01', periods=24, freq='5min')
    parking_ids = ['P1', 'P2']
    
    data = []
    for parking_id in parking_ids:
        for date in dates:
            data.append({
                'timestamp': date,
                'parking_id': parking_id,
                'available_spaces': np.random.randint(0, 100),
                'total_spaces': 100,
                'latitude': 41.3851 + np.random.uniform(-0.01, 0.01),
                'longitude': 2.1734 + np.random.uniform(-0.01, 0.01)
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_poi_data():
    """Create sample POI data for testing."""
    return pd.DataFrame({
        'latitude': [41.3851, 41.3852, 41.3853],
        'longitude': [2.1734, 2.1735, 2.1736],
        'category': ['restaurant', 'shop', 'entertainment'],
        'importance': [0.8, 0.6, 0.7]
    })

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
    """Create a FeatureEngineeringPipeline instance with test configuration."""
    config = {
        'feature_engineering': {
            'poi_features': {
                'radii': [100, 200],
                'categories': ['restaurant', 'shop', 'entertainment']
            }
        },
        'parallel_processing': {
            'enabled': False
        },
        'scale_features': True
    }
    return FeatureEngineeringPipeline(config)

@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=24, freq='5min')
    data = []
    
    for date in dates:
        data.append({
            'timestamp': date,
            'temperature': np.random.uniform(10, 30),
            'feels_like': np.random.uniform(10, 30),
            'humidity': np.random.uniform(0, 100),
            'pressure': np.random.uniform(1000, 1020),
            'wind_speed': np.random.uniform(0, 30),
            'wind_deg': np.random.uniform(0, 360),
            'clouds': np.random.uniform(0, 100),
            'rain_1h': np.random.uniform(0, 10),
            'snow_1h': 0,
            'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain']),
            'weather_description': 'test description',
            'is_raining': np.random.choice([0, 1]),
            'is_snowing': 0,
            'precipitation': np.random.uniform(0, 10),
            'wind_sin': np.sin(2 * np.pi * np.random.uniform(0, 360) / 360),
            'wind_cos': np.cos(2 * np.pi * np.random.uniform(0, 360) / 360)
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def pipeline_with_weather():
    """Create a FeatureEngineeringPipeline instance with weather configuration."""
    config = {
        'weather': {
            'api_key': 'test_api_key',
            'cache_dir': 'tests/cache/weather'
        },
        'feature_engineering': {
            'weather_features': {
                'enabled': True,
                'features': [
                    'temperature', 'feels_like', 'humidity', 'pressure',
                    'wind_speed', 'wind_sin', 'wind_cos', 'clouds',
                    'is_raining', 'is_snowing', 'precipitation'
                ]
            }
        }
    }
    return FeatureEngineeringPipeline(config)

@pytest.fixture
def pipeline_with_events():
    """Create a FeatureEngineeringPipeline instance with event configuration."""
    config = {
        'events': {
            'api_key': 'test_api_key',
            'cache_dir': 'tests/cache/events',
            'radius': 5000
        },
        'feature_engineering': {
            'event_features': {
                'enabled': True,
                'radii': [500, 1000, 2000]
            }
        }
    }
    return FeatureEngineeringPipeline(config)

@pytest.fixture
def sample_event_data():
    """Create sample event data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=24, freq='h')
    data = []
    
    for date in dates:
        data.append({
            'event_id': f'event_{len(data)}',
            'timestamp': date,
            'end_time': date + timedelta(hours=2),
            'name': f'Test Event {len(data)}',
            'category': np.random.choice(['concert', 'sports', 'festival', 'exhibition', 'conference', 'theater', 'other']),
            'venue': f'Venue {len(data)}',
            'latitude': 41.3851 + np.random.uniform(-0.01, 0.01),
            'longitude': 2.1734 + np.random.uniform(-0.01, 0.01),
            'capacity': np.random.randint(100, 10000),
            'ticket_price': np.random.uniform(0, 100),
            'is_free': np.random.choice([True, False]),
            'description': 'Test event description',
            'importance': np.random.uniform(0, 1),
            'is_major_event': np.random.choice([0, 1])
        })
    
    return pd.DataFrame(data)

def test_validate_data(pipeline, sample_data):
    """Test data validation."""
    # Test valid data
    pipeline.validate_data(sample_data)
    
    # Test missing required columns
    invalid_data = sample_data.drop('timestamp', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        pipeline.validate_data(invalid_data)
    
    # Test invalid data types
    invalid_data = sample_data.copy()
    invalid_data['timestamp'] = 'invalid'
    with pytest.raises(ValueError, match="timestamp column must be datetime type"):
        pipeline.validate_data(invalid_data)
    
    # Test invalid value ranges
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'available_spaces'] = -1
    with pytest.raises(ValueError, match="available_spaces cannot be negative"):
        pipeline.validate_data(invalid_data)
    
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'available_spaces'] = 101
    with pytest.raises(ValueError, match="available_spaces cannot be greater than total_spaces"):
        pipeline.validate_data(invalid_data)

def test_create_temporal_features(pipeline, sample_data):
    """Test temporal feature creation."""
    result = pipeline.create_temporal_features(sample_data)
    
    # Check if all temporal features are created
    expected_features = [
        'hour', 'dayofweek', 'month',
        'hour_sin', 'hour_cos',
        'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos',
        'is_weekend', 'is_public_holiday'
    ]
    
    for feature in expected_features:
        assert feature in result.columns
    
    # Check value ranges
    assert result['hour'].between(0, 23).all()
    assert result['dayofweek'].between(0, 6).all()
    assert result['month'].between(1, 12).all()
    assert result['is_weekend'].isin([0, 1]).all()
    assert result['is_public_holiday'].isin([0, 1]).all()

def test_create_lag_features(pipeline, sample_data):
    """Test lag feature creation."""
    result = pipeline.create_lag_features(sample_data)
    
    # Check if lag features are created
    expected_lags = [1, 2, 3, 6, 12, 24]
    for lag in expected_lags:
        assert f'occupancy_rate_lag_{lag}' in result.columns
        assert f'available_spaces_lag_{lag}' in result.columns
    
    # Check if occupancy rate is calculated correctly
    assert 'occupancy_rate' in result.columns
    expected_occupancy = 1 - (sample_data['available_spaces'] / sample_data['total_spaces'])
    pd.testing.assert_series_equal(
        result['occupancy_rate'].reset_index(drop=True),
        expected_occupancy.reset_index(drop=True),
        check_names=False
    )

def test_create_poi_features(pipeline, sample_data, sample_poi_data):
    """Test POI feature creation."""
    result = pipeline.create_poi_features(sample_data, sample_poi_data)
    
    # Check if POI features are created
    expected_features = []
    for radius in [100, 200]:
        for category in ['restaurant', 'shop', 'entertainment']:
            expected_features.extend([
                f'poi_{category}_count_{radius}m',
                f'poi_{category}_density_{radius}m'
            ])
    
    for feature in expected_features:
        assert feature in result.columns
    
    # Check if values are non-negative
    for feature in expected_features:
        assert (result[feature] >= 0).all()

def test_create_features(pipeline, sample_data, sample_poi_data):
    """Test the complete feature creation pipeline."""
    result = pipeline.create_features(sample_data)
    
    # Check if all temporal features are created
    temporal_features = [
        'hour', 'dayofweek', 'month',
        'hour_sin', 'hour_cos',
        'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos',
        'is_weekend', 'is_public_holiday'
    ]
    for feature in temporal_features:
        assert feature in result.columns, f"Missing temporal feature: {feature}"
    
    # Check if lag and facility features are present
    lag_features = [col for col in result.columns if 'lag' in col]
    facility_features = [col for col in result.columns if 'facility' in col or 'is_open_now' in col]
    assert len(lag_features) > 0, "No lag features found"
    assert len(facility_features) > 0, "No facility features found"
    
    # Optional feature groups that depend on data availability
    optional_groups = [
        'weather', 'events', 'transport', 'poi'
    ]
    for group in optional_groups:
        group_features = [col for col in result.columns if col.startswith(group)]
        if len(group_features) > 0:
            print(f"Found {len(group_features)} features for optional group: {group}")

def test_fit_transform(pipeline, sample_data):
    """Test fit_transform method."""
    result = pipeline.fit_transform(sample_data)
    
    # Check if scalers are created
    assert len(pipeline.scalers) > 0
    
    # Check if all features are scaled
    for feature, scaler in pipeline.scalers.items():
        assert feature in result.columns
        # Check that most values are within reasonable range
        scaled_values = result[feature].dropna()
        if len(scaled_values) > 0:
            assert abs(scaled_values.mean()) < 1.0  # Mean should be close to 0
            assert scaled_values.std() < 2.0  # Std should be close to 1

def test_transform(pipeline, sample_data):
    """Test transform method."""
    # First fit the pipeline
    pipeline.fit_transform(sample_data)
    
    # Then transform new data
    new_data = sample_data.copy()
    result = pipeline.transform(new_data)
    
    # Check if all features are present and scaled
    assert len(result.columns) >= len(sample_data.columns)
    for feature, scaler in pipeline.scalers.items():
        assert feature in result.columns
        # Check that most values are within reasonable range
        scaled_values = result[feature].dropna()
        if len(scaled_values) > 0:
            assert abs(scaled_values.mean()) < 1.0  # Mean should be close to 0
            assert scaled_values.std() < 2.0  # Std should be close to 1

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
    
    # Test that validation raises the expected error
    with pytest.raises(ValueError, match="Missing required columns: \['available_spaces'\]"):
        pipeline.create_features(missing_cols_df)
    
    # Test with invalid data types
    print("\nTesting invalid data types...")
    invalid_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01']),
        'parking_id': ['P1'],
        'available_spaces': ['invalid'],  # Invalid type
        'total_spaces': ['invalid']  # Invalid type
    })
    print(f"Invalid types DataFrame:\n{invalid_df.dtypes}")
    
    # Test that validation raises the expected error
    with pytest.raises(ValueError, match="available_spaces must be numeric"):
        pipeline.create_features(invalid_df)
    
    # Test with invalid value ranges
    print("\nTesting invalid value ranges...")
    invalid_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01']),
        'parking_id': ['P1'],
        'available_spaces': [-1],  # Negative value
        'total_spaces': [100]
    })
    print(f"Invalid values DataFrame:\n{invalid_df}")
    
    # Test that validation raises the expected error
    with pytest.raises(ValueError, match="available_spaces cannot be negative"):
        pipeline.create_features(invalid_df)
    
    # Test with available_spaces > total_spaces
    print("\nTesting available_spaces > total_spaces...")
    invalid_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01']),
        'parking_id': ['P1'],
        'available_spaces': [200],  # Greater than total_spaces
        'total_spaces': [100]
    })
    print(f"Invalid ratio DataFrame:\n{invalid_df}")
    
    # Test that validation raises the expected error
    with pytest.raises(ValueError, match="available_spaces cannot be greater than total_spaces"):
        pipeline.create_features(invalid_df)

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

def test_create_weather_features(pipeline_with_weather, sample_data, sample_weather_data, monkeypatch):
    """Test weather feature creation."""
    # Mock the weather data fetcher
    def mock_fetch_weather_data(*args, **kwargs):
        return sample_weather_data
    
    monkeypatch.setattr(pipeline_with_weather.weather_fetcher, 'fetch_weather_data', mock_fetch_weather_data)
    
    # Create features
    result = pipeline_with_weather.create_weather_features(sample_data)
    
    # Check if weather features are created
    weather_features = [
        'temperature', 'feels_like', 'humidity', 'pressure',
        'wind_speed', 'wind_sin', 'wind_cos', 'clouds',
        'is_raining', 'is_snowing', 'precipitation'
    ]
    
    for feature in weather_features:
        assert feature in result.columns
        
        # Check lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            assert f'{feature}_lag_{lag}' in result.columns
        
        # Check rolling statistics
        for window in [6, 12, 24]:
            assert f'{feature}_rolling_mean_{window}' in result.columns
            assert f'{feature}_rolling_std_{window}' in result.columns
    
    # Check value ranges
    for feature in weather_features:
        if feature in ['is_raining', 'is_snowing']:
            assert result[feature].isin([0, 1]).all()
        elif feature in ['wind_sin', 'wind_cos']:
            assert result[feature].between(-1, 1).all()
        else:
            assert not result[feature].isna().all()

def test_create_weather_features_no_data(pipeline_with_weather, sample_data, monkeypatch):
    """Test weather feature creation with no weather data."""
    # Mock the weather data fetcher to return empty DataFrame
    def mock_fetch_weather_data(*args, **kwargs):
        return pd.DataFrame()
    
    monkeypatch.setattr(pipeline_with_weather.weather_fetcher, 'fetch_weather_data', mock_fetch_weather_data)
    
    # Create features
    result = pipeline_with_weather.create_weather_features(sample_data)
    
    # Check that input data is unchanged
    assert len(result) == len(sample_data)
    assert all(col in result.columns for col in sample_data.columns)
    assert not any(col.startswith('temperature_') for col in result.columns)

def test_create_weather_features_merge(pipeline_with_weather, sample_data, sample_weather_data, monkeypatch):
    """Test weather data merging with parking data."""
    # Mock the weather data fetcher
    def mock_fetch_weather_data(*args, **kwargs):
        return sample_weather_data
    
    monkeypatch.setattr(pipeline_with_weather.weather_fetcher, 'fetch_weather_data', mock_fetch_weather_data)
    
    # Create features
    result = pipeline_with_weather.create_weather_features(sample_data)
    
    # Check that all parking data is preserved
    assert len(result) == len(sample_data)
    assert all(col in result.columns for col in sample_data.columns)
    
    # Check that weather data is properly merged
    weather_timestamps = set(sample_weather_data['timestamp'])
    result_timestamps = set(result['timestamp'])
    assert weather_timestamps.issubset(result_timestamps)

def test_create_event_features(pipeline_with_events, sample_data, sample_event_data, monkeypatch):
    """Test event feature creation."""
    # Mock the event data fetcher
    def mock_fetch_event_data(*args, **kwargs):
        return sample_event_data
    
    monkeypatch.setattr(pipeline_with_events.event_fetcher, 'fetch_event_data', mock_fetch_event_data)
    
    # Create features
    result = pipeline_with_events.create_event_features(sample_data)
    
    # Check if event features are created
    expected_features = []
    for radius in [500, 1000, 2000]:
        expected_features.extend([
            f'event_count_{radius}m',
            f'event_importance_{radius}m',
            f'major_event_count_{radius}m',
            f'event_density_{radius}m',
            f'weighted_event_importance_{radius}m'
        ])
    
    expected_features.extend([
        'hours_until_next_event',
        'hours_since_last_event',
        'is_event_time',
        'is_major_event_time'
    ])
    
    for feature in expected_features:
        assert feature in result.columns
    
    # Check value ranges
    for radius in [500, 1000, 2000]:
        assert (result[f'event_count_{radius}m'] >= 0).all()
        assert (result[f'event_importance_{radius}m'] >= 0).all()
        assert (result[f'major_event_count_{radius}m'] >= 0).all()
        assert (result[f'event_density_{radius}m'] >= 0).all()
        assert (result[f'weighted_event_importance_{radius}m'] >= 0).all()
    
    assert result['is_event_time'].isin([True, False]).all()
    assert result['is_major_event_time'].isin([True, False]).all()

def test_create_event_features_no_data(pipeline_with_events, sample_data, monkeypatch):
    """Test event feature creation with no event data."""
    # Mock the event data fetcher to return empty DataFrame
    def mock_fetch_event_data(*args, **kwargs):
        return pd.DataFrame()
    
    monkeypatch.setattr(pipeline_with_events.event_fetcher, 'fetch_event_data', mock_fetch_event_data)
    
    # Create features
    result = pipeline_with_events.create_event_features(sample_data)
    
    # Check that input data is unchanged
    assert len(result) == len(sample_data)
    assert all(col in result.columns for col in sample_data.columns)
    assert not any(col.startswith('event_') for col in result.columns)

def test_create_event_features_merge(pipeline_with_events, sample_data, sample_event_data, monkeypatch):
    """Test event data merging with parking data."""
    # Mock the event data fetcher
    def mock_fetch_event_data(*args, **kwargs):
        return sample_event_data
    
    monkeypatch.setattr(pipeline_with_events.event_fetcher, 'fetch_event_data', mock_fetch_event_data)
    
    # Create features
    result = pipeline_with_events.create_event_features(sample_data)
    
    # Check that all parking data is preserved
    assert len(result) == len(sample_data)
    assert all(col in result.columns for col in sample_data.columns)
    
    # Check that event features are properly calculated
    assert 'event_count_500m' in result.columns
    assert 'event_importance_500m' in result.columns
    assert 'major_event_count_500m' in result.columns

def test_event_time_features(pipeline_with_events, sample_data, sample_event_data, monkeypatch):
    """Test event time-based features."""
    # Mock the event data fetcher
    def mock_fetch_event_data(*args, **kwargs):
        return sample_event_data
    
    monkeypatch.setattr(pipeline_with_events.event_fetcher, 'fetch_event_data', mock_fetch_event_data)
    
    # Create features
    result = pipeline_with_events.create_event_features(sample_data)
    
    # Check time-based features
    assert 'hours_until_next_event' in result.columns
    assert 'hours_since_last_event' in result.columns
    assert 'is_event_time' in result.columns
    assert 'is_major_event_time' in result.columns
    
    # Check value ranges
    assert result['hours_until_next_event'].between(0, float('inf')).all()
    assert result['hours_since_last_event'].between(0, float('inf')).all()
    assert result['is_event_time'].isin([True, False]).all()
    assert result['is_major_event_time'].isin([True, False]).all() 