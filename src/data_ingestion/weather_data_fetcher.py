"""
Weather data fetcher for Barcelona parking prediction.

This module provides functionality to fetch and process weather data for Barcelona.
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

class WeatherDataFetcher:
    """Class to fetch and process weather data for Barcelona."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the weather data fetcher.
        
        Args:
            config: Optional configuration dictionary containing API keys and parameters
        """
        self.config = config or {}
        load_dotenv()  # Load environment variables
        
        # Get API key from environment or config
        self.api_key = os.getenv('OPENWEATHER_API_KEY') or self.config.get('api_key')
        if not self.api_key:
            logger.warning("No OpenWeather API key found. Weather data fetching will be limited.")
        
        # Barcelona coordinates
        self.lat = 41.3851
        self.lon = 2.1734
        
        # Cache directory
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache/weather'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch weather data for a given date range.
        
        Args:
            start_date: Start date for weather data
            end_date: End date for weather data
            
        Returns:
            DataFrame containing weather data
        """
        # Check cache first
        cache_file = self._get_cache_file(start_date, end_date)
        if cache_file.exists():
            logger.info(f"Loading weather data from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        
        if not self.api_key:
            logger.warning("No API key available, returning empty DataFrame")
            return pd.DataFrame()
        
        # Fetch data from OpenWeather API
        try:
            data = self._fetch_from_api(start_date, end_date)
            df = self._process_weather_data(data)
            
            # Cache the results
            df.to_parquet(cache_file)
            logger.info(f"Cached weather data to {cache_file}")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()
    
    def _get_cache_file(self, start_date: datetime, end_date: datetime) -> Path:
        """Get the cache file path for a date range."""
        return self.cache_dir / f"weather_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    
    def _fetch_from_api(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch weather data from OpenWeather API."""
        # OpenWeather API endpoint for historical data
        base_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
        
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            timestamp = int(current_date.timestamp())
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'dt': timestamp,
                'appid': self.api_key,
                'units': 'metric'  # Use metric units
            }
            
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data.append(response.json())
                current_date += timedelta(hours=1)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching weather data for {current_date}: {e}")
                break
        
        return data
    
    def _process_weather_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process raw weather data into a DataFrame."""
        processed_data = []
        
        for entry in data:
            for hour_data in entry.get('hourly', []):
                processed_data.append({
                    'timestamp': datetime.fromtimestamp(hour_data['dt']),
                    'temperature': hour_data['temp'],
                    'feels_like': hour_data['feels_like'],
                    'humidity': hour_data['humidity'],
                    'pressure': hour_data['pressure'],
                    'wind_speed': hour_data['wind_speed'],
                    'wind_deg': hour_data['wind_deg'],
                    'clouds': hour_data['clouds'],
                    'rain_1h': hour_data.get('rain', {}).get('1h', 0),
                    'snow_1h': hour_data.get('snow', {}).get('1h', 0),
                    'weather_main': hour_data['weather'][0]['main'],
                    'weather_description': hour_data['weather'][0]['description']
                })
        
        df = pd.DataFrame(processed_data)
        
        # Add derived features
        df['is_raining'] = (df['rain_1h'] > 0).astype(int)
        df['is_snowing'] = (df['snow_1h'] > 0).astype(int)
        df['precipitation'] = df['rain_1h'] + df['snow_1h']
        
        # Create cyclical features for wind direction
        df['wind_sin'] = np.sin(2 * np.pi * df['wind_deg'] / 360)
        df['wind_cos'] = np.cos(2 * np.pi * df['wind_deg'] / 360)
        
        return df 