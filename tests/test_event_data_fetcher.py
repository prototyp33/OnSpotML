"""
Tests for the EventDataFetcher class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from src.data_ingestion.event_data_fetcher import EventDataFetcher

@pytest.fixture
def config():
    """Create test configuration."""
    return {
        'api_key': 'test_api_key',
        'cache_dir': 'tests/cache/events',
        'radius': 5000
    }

@pytest.fixture
def event_fetcher(config):
    """Create an EventDataFetcher instance."""
    return EventDataFetcher(config)

@pytest.fixture
def sample_event_data():
    """Create sample event data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
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
            'description': 'Test event description'
        })
    
    return pd.DataFrame(data)

def test_initialization(event_fetcher, config):
    """Test EventDataFetcher initialization."""
    assert event_fetcher.api_key == config['api_key']
    assert event_fetcher.lat == 41.3851
    assert event_fetcher.lon == 2.1734
    assert event_fetcher.radius == config['radius']
    assert event_fetcher.cache_dir == Path(config['cache_dir'])

def test_get_cache_file(event_fetcher):
    """Test cache file path generation."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    cache_file = event_fetcher._get_cache_file(start_date, end_date)
    expected_path = Path('tests/cache/events/events_20240101_20240102.parquet')
    
    assert cache_file == expected_path

def test_process_event_data(event_fetcher):
    """Test event data processing."""
    # Create sample API response
    api_data = [{
        'id': 'event_1',
        'start_time': '2024-01-01T10:00:00',
        'end_time': '2024-01-01T12:00:00',
        'name': 'Test Concert',
        'category': 'concert',
        'venue': {
            'name': 'Test Venue',
            'latitude': 41.3851,
            'longitude': 2.1734,
            'capacity': 1000
        },
        'ticket_price': 50.0,
        'is_free': False,
        'description': 'Test concert description'
    }]
    
    df = event_fetcher._process_event_data(api_data)
    
    # Check if all expected columns are present
    expected_columns = [
        'event_id', 'timestamp', 'end_time', 'name', 'category',
        'venue', 'latitude', 'longitude', 'capacity', 'ticket_price',
        'is_free', 'description', 'importance', 'duration_hours',
        'is_major_event', 'is_free_event', 'hour', 'dayofweek', 'is_weekend'
    ]
    
    assert all(col in df.columns for col in expected_columns)
    
    # Check value ranges
    assert df['importance'].between(0, 1).all()
    assert df['duration_hours'].between(0, 24).all()
    assert df['is_major_event'].isin([0, 1]).all()
    assert df['is_free_event'].isin([0, 1]).all()
    assert df['hour'].between(0, 23).all()
    assert df['dayofweek'].between(0, 6).all()
    assert df['is_weekend'].isin([0, 1]).all()

def test_fetch_event_data_caching(event_fetcher, sample_event_data, tmp_path):
    """Test event data fetching and caching."""
    # Set up test cache directory
    event_fetcher.cache_dir = tmp_path
    
    # Save sample data to cache
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    cache_file = event_fetcher._get_cache_file(start_date, end_date)
    sample_event_data.to_parquet(cache_file)
    
    # Fetch data (should use cache)
    df = event_fetcher.fetch_event_data(start_date, end_date)
    
    # Check if data matches
    pd.testing.assert_frame_equal(df, sample_event_data)

def test_fetch_event_data_no_api_key(event_fetcher):
    """Test event data fetching without API key."""
    event_fetcher.api_key = None
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    df = event_fetcher.fetch_event_data(start_date, end_date)
    assert df.empty 