"""
Tests for the WeatherDataFetcher class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from src.data_ingestion.weather_data_fetcher import WeatherDataFetcher

@pytest.fixture
def config():
    """Create test configuration."""
    return {
        'api_key': 'test_api_key',
        'cache_dir': 'tests/cache/weather'
    }

@pytest.fixture
def weather_fetcher(config):
    """Create a WeatherDataFetcher instance."""
    return WeatherDataFetcher(config)

@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
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
            'weather_description': 'test description'
        })
    
    return pd.DataFrame(data)

def test_initialization(weather_fetcher, config):
    """Test WeatherDataFetcher initialization."""
    assert weather_fetcher.api_key == config['api_key']
    assert weather_fetcher.lat == 41.3851
    assert weather_fetcher.lon == 2.1734
    assert weather_fetcher.cache_dir == Path(config['cache_dir'])

def test_get_cache_file(weather_fetcher):
    """Test cache file path generation."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    cache_file = weather_fetcher._get_cache_file(start_date, end_date)
    expected_path = Path('tests/cache/weather/weather_20240101_20240102.parquet')
    
    assert cache_file == expected_path

def test_process_weather_data(weather_fetcher):
    """Test weather data processing."""
    # Create sample API response
    api_data = [{
        'hourly': [{
            'dt': int(datetime(2024, 1, 1).timestamp()),
            'temp': 20.0,
            'feels_like': 22.0,
            'humidity': 60,
            'pressure': 1015,
            'wind_speed': 5.0,
            'wind_deg': 180,
            'clouds': 20,
            'rain': {'1h': 0.5},
            'snow': {'1h': 0},
            'weather': [{'main': 'Rain', 'description': 'light rain'}]
        }]
    }]
    
    df = weather_fetcher._process_weather_data(api_data)
    
    # Check if all expected columns are present
    expected_columns = [
        'timestamp', 'temperature', 'feels_like', 'humidity', 'pressure',
        'wind_speed', 'wind_deg', 'clouds', 'rain_1h', 'snow_1h',
        'weather_main', 'weather_description', 'is_raining', 'is_snowing',
        'precipitation', 'wind_sin', 'wind_cos'
    ]
    
    assert all(col in df.columns for col in expected_columns)
    
    # Check value ranges
    assert df['temperature'].between(10, 30).all()
    assert df['humidity'].between(0, 100).all()
    assert df['pressure'].between(1000, 1020).all()
    assert df['wind_speed'].between(0, 30).all()
    assert df['wind_deg'].between(0, 360).all()
    assert df['clouds'].between(0, 100).all()
    assert df['is_raining'].isin([0, 1]).all()
    assert df['is_snowing'].isin([0, 1]).all()
    assert df['precipitation'] >= 0

def test_fetch_weather_data_caching(weather_fetcher, sample_weather_data, tmp_path):
    """Test weather data fetching and caching."""
    # Set up test cache directory
    weather_fetcher.cache_dir = tmp_path
    
    # Save sample data to cache
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    cache_file = weather_fetcher._get_cache_file(start_date, end_date)
    sample_weather_data.to_parquet(cache_file)
    
    # Fetch data (should use cache)
    df = weather_fetcher.fetch_weather_data(start_date, end_date)
    
    # Check if data matches
    pd.testing.assert_frame_equal(df, sample_weather_data)

def test_fetch_weather_data_no_api_key(weather_fetcher):
    """Test weather data fetching without API key."""
    weather_fetcher.api_key = None
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    df = weather_fetcher.fetch_weather_data(start_date, end_date)
    assert df.empty 