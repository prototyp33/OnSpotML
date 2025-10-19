"""
Event data fetcher for Barcelona parking prediction.

This module provides functionality to fetch and process event data for Barcelona.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import requests
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventDataFetcher:
    """Class to fetch and process event data for Barcelona."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the event data fetcher.
        
        Args:
            config: Optional configuration dictionary containing API keys and parameters
        """
        self.config = config or {}
        load_dotenv()  # Load environment variables
        
        # Get API key from environment or config
        self.api_key = os.getenv('EVENT_API_KEY') or self.config.get('api_key')
        if not self.api_key:
            logger.warning("No Event API key found. Event data fetching will be limited.")
        
        # Barcelona coordinates
        self.lat = 41.3851
        self.lon = 2.1734
        self.radius = self.config.get('radius', 5000)  # Default 5km radius
        
        # Cache directory
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache/events'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Event categories and their importance weights
        self.event_categories = {
            'concert': 1.0,
            'sports': 0.9,
            'festival': 0.8,
            'exhibition': 0.7,
            'conference': 0.6,
            'theater': 0.5,
            'other': 0.3
        }
    
    def fetch_event_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch event data for a given date range.
        
        Args:
            start_date: Start date for event data
            end_date: End date for event data
            
        Returns:
            DataFrame containing event data
        """
        # Check cache first
        cache_file = self._get_cache_file(start_date, end_date)
        if cache_file.exists():
            logger.info(f"Loading event data from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        
        if not self.api_key:
            logger.warning("No API key available, returning empty DataFrame")
            return pd.DataFrame()
        
        # Fetch data from Event API
        try:
            data = self._fetch_from_api(start_date, end_date)
            df = self._process_event_data(data)
            
            # Cache the results
            df.to_parquet(cache_file)
            logger.info(f"Cached event data to {cache_file}")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching event data: {e}")
            return pd.DataFrame()
    
    def _get_cache_file(self, start_date: datetime, end_date: datetime) -> Path:
        """Get the cache file path for a date range."""
        return self.cache_dir / f"events_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    
    def _fetch_from_api(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch event data from Event API."""
        # Event API endpoint
        base_url = "https://api.events.example.com/v1/events"  # Replace with actual API endpoint
        
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'radius': self.radius,
                'date': current_date.strftime('%Y-%m-%d'),
                'api_key': self.api_key
            }
            
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data.extend(response.json()['events'])
                current_date += timedelta(days=1)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching event data for {current_date}: {e}")
                break
        
        return data
    
    def _process_event_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process raw event data into a DataFrame."""
        processed_data = []
        
        for event in data:
            # Extract event details
            event_data = {
                'event_id': event.get('id'),
                'timestamp': pd.to_datetime(event.get('start_time')),
                'end_time': pd.to_datetime(event.get('end_time')),
                'name': event.get('name'),
                'category': event.get('category', 'other').lower(),
                'venue': event.get('venue', {}).get('name'),
                'latitude': event.get('venue', {}).get('latitude'),
                'longitude': event.get('venue', {}).get('longitude'),
                'capacity': event.get('venue', {}).get('capacity', 0),
                'ticket_price': event.get('ticket_price', 0),
                'is_free': event.get('is_free', False),
                'description': event.get('description', '')
            }
            
            # Calculate event importance
            category_weight = self.event_categories.get(event_data['category'], 0.3)
            capacity_factor = min(event_data['capacity'] / 1000, 1.0)  # Normalize capacity
            price_factor = 1.0 if event_data['is_free'] else min(event_data['ticket_price'] / 100, 1.0)
            
            event_data['importance'] = category_weight * (0.4 * capacity_factor + 0.6 * price_factor)
            
            processed_data.append(event_data)
        
        df = pd.DataFrame(processed_data)
        
        if not df.empty:
            # Add derived features
            df['duration_hours'] = (df['end_time'] - df['timestamp']).dt.total_seconds() / 3600
            df['is_major_event'] = (df['importance'] > 0.7).astype(int)
            df['is_free_event'] = df['is_free'].astype(int)
            
            # Create time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['dayofweek'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
        
        return df 