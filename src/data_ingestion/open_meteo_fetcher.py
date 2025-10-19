"""
Open-Meteo Weather Data Fetcher
Free weather API with no API key required
Perfect for Barcelona parking prediction system
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json


class OpenMeteoFetcher:
    """
    Fetches weather data from Open-Meteo API (free, no API key required)
    Provides current, forecast, and historical weather data for Barcelona
    """
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.archive_url = "https://archive-api.open-meteo.com/v1"  # Separate URL for historical data
        self.barcelona_coords = {
            "latitude": 41.3851,
            "longitude": 2.1734
        }
        self.logger = logging.getLogger(__name__)
        
    def get_current_weather(self) -> Optional[Dict[str, Any]]:
        """
        Get current weather conditions for Barcelona
        
        Returns:
            Dict with current weather data or None if failed
        """
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "latitude": self.barcelona_coords["latitude"],
                "longitude": self.barcelona_coords["longitude"],
                "current": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m", 
                    "apparent_temperature",
                    "is_day",
                    "precipitation",
                    "rain",
                    "showers",
                    "snowfall",
                    "weather_code",
                    "cloud_cover",
                    "pressure_msl",
                    "surface_pressure",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m",
                    "sunshine_duration"
                ]),
                "timezone": "Europe/Madrid",
                "forecast_days": 1
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract and format current weather
            current = data.get("current", {})
            
            weather_data = {
                "timestamp": current.get("time"),
                "temperature_c": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "apparent_temperature_c": current.get("apparent_temperature"),
                "precipitation_mm": current.get("precipitation"),
                "rain_mm": current.get("rain"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "wind_direction": current.get("wind_direction_10m"),
                "pressure_hpa": current.get("pressure_msl"),
                "cloud_cover": current.get("cloud_cover"),
                "weather_code": current.get("weather_code"),
                "is_day": current.get("is_day"),
                "source": "open-meteo"
            }
            
            self.logger.info(f"Successfully fetched current weather: {weather_data['temperature_c']}°C")
            return weather_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch current weather: {e}")
            return None
    
    def get_historical_weather(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical weather data for Barcelona
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical weather data or None if failed
        """
        try:
            url = f"{self.archive_url}/archive"  # Use archive URL
            params = {
                "latitude": self.barcelona_coords["latitude"],
                "longitude": self.barcelona_coords["longitude"],
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join([  # Join parameters with commas
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "rain",
                    "snowfall",
                    "pressure_msl",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "weather_code"
                ]),
                "timezone": "Europe/Madrid"
            }
            
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            hourly_data = data.get("hourly", {})
            
            # Convert to DataFrame
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly_data.get("time", [])),
                "temperature_c": hourly_data.get("temperature_2m", []),
                "humidity": hourly_data.get("relative_humidity_2m", []),
                "apparent_temperature_c": hourly_data.get("apparent_temperature", []),
                "precipitation_mm": hourly_data.get("precipitation", []),
                "rain_mm": hourly_data.get("rain", []),
                "pressure_hpa": hourly_data.get("pressure_msl", []),
                "cloud_cover": hourly_data.get("cloud_cover", []),
                "wind_speed_kmh": hourly_data.get("wind_speed_10m", []),
                "wind_direction": hourly_data.get("wind_direction_10m", []),
                "weather_code": hourly_data.get("weather_code", [])
            })
            
            df["source"] = "open-meteo"
            
            self.logger.info(f"Successfully fetched {len(df)} hours of historical weather data")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch historical weather: {e}")
            return None
    
    def get_forecast_weather(self, hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Get weather forecast for Barcelona
        
        Args:
            hours: Number of hours to forecast (max 168 for free tier)
            
        Returns:
            DataFrame with hourly weather forecast or None if failed
        """
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "latitude": self.barcelona_coords["latitude"],
                "longitude": self.barcelona_coords["longitude"],
                "hourly": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "weather_code",
                    "cloud_cover",
                    "pressure_msl", 
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "sunshine_duration"  # Added sunshine duration
                ]),
                "timezone": "Europe/Madrid"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse hourly data with error handling for array length mismatches
            hourly_data = data.get("hourly", {})
            
            # Get the base time array length
            times = hourly_data.get("time", [])
            if not times:
                self.logger.error("No time data in forecast response")
                return None
                
            base_length = len(times)
            self.logger.info(f"Processing {base_length} hourly forecast records")
            
            # Helper function to ensure array length matches
            def ensure_length(arr, target_length, fill_value=None):
                if len(arr) == target_length:
                    return arr
                elif len(arr) < target_length:
                    # Pad with fill_value
                    return arr + [fill_value] * (target_length - len(arr))
                else:
                    # Truncate
                    return arr[:target_length]
            
            df = pd.DataFrame({
                "datetime": times,
                "temperature_c": ensure_length(hourly_data.get("temperature_2m", []), base_length, 20.0),
                "humidity": ensure_length(hourly_data.get("relative_humidity_2m", []), base_length, 50),
                "precipitation_mm": ensure_length(hourly_data.get("precipitation", []), base_length, 0.0),
                "cloud_cover": ensure_length(hourly_data.get("cloud_cover", []), base_length, 0),
                "pressure_hpa": ensure_length(hourly_data.get("pressure_msl", []), base_length, 1013.25),
                "wind_speed_kmh": ensure_length(hourly_data.get("wind_speed_10m", []), base_length, 0.0),
                "wind_direction": ensure_length(hourly_data.get("wind_direction_10m", []), base_length, 0),
                "weather_code": ensure_length(hourly_data.get("weather_code", []), base_length, 0),
                "sunshine_duration": ensure_length(hourly_data.get("sunshine_duration", []), base_length, 0.0)
            })
            
            # Convert datetime column
            df["datetime"] = pd.to_datetime(df["datetime"])
            
            df["source"] = "open-meteo"
            df["forecast_type"] = "hourly"
            
            self.logger.info(f"Successfully fetched {hours}-hour weather forecast ({len(df)} hours)")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch weather forecast: {e}")
            return None
    
    def save_weather_data(self, data, filepath: str, data_type: str = "current"):
        """
        Save weather data to file
        
        Args:
            data: Weather data (dict or DataFrame)
            filepath: Output file path
            data_type: Type of data being saved
        """
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
                self.logger.info(f"Saved {len(data)} {data_type} weather records to {filepath}")
            else:
                # Convert Timestamp objects to strings for JSON serialization
                if isinstance(data, dict):
                    data_copy = data.copy()
                    for key, value in data_copy.items():
                        if hasattr(value, 'to_pydatetime'):  # Pandas Timestamp
                            data_copy[key] = value.isoformat()
                        elif isinstance(value, list):
                            # Handle lists that might contain timestamps
                            data_copy[key] = [
                                item.isoformat() if hasattr(item, 'to_pydatetime') else item 
                                for item in value
                            ]
                        elif isinstance(value, dict):
                            # Handle nested dictionaries
                            for nested_key, nested_value in value.items():
                                if hasattr(nested_value, 'to_pydatetime'):
                                    data_copy[key][nested_key] = nested_value.isoformat()
                    data = data_copy
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)  # Add default=str as fallback
                self.logger.info(f"Saved {data_type} weather data to {filepath}")
                
        except Exception as e:
            self.logger.error(f"Failed to save {data_type} weather data: {e}")
    
    def get_weather_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive weather summary for Barcelona
        
        Returns:
            Dictionary with current, forecast, and recent historical data
        """
        summary = {
            "location": "Barcelona, Spain",
            "coordinates": self.barcelona_coords,
            "timestamp": datetime.now().isoformat(),
            "current": None,
            "forecast_24h": None,
            "recent_history": None
        }
        
        # Current weather
        current = self.get_current_weather()
        if current:
            summary["current"] = current
        
        # 24-hour forecast
        forecast = self.get_forecast_weather(hours=24)
        if forecast is not None:
            summary["forecast_24h"] = forecast.head(24).to_dict('records')
        
        # Last 7 days historical
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        historical = self.get_historical_weather(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if historical is not None:
            summary["recent_history"] = {
                "avg_temperature": historical["temperature_c"].mean(),
                "total_precipitation": historical["precipitation_mm"].sum(),
                "avg_humidity": historical["humidity"].mean()
            }
        
        return summary


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fetcher = OpenMeteoFetcher()
    
    # Test current weather
    current = fetcher.get_current_weather()
    print("Current Weather:", current)
    
    # Test forecast
    forecast = fetcher.get_forecast_weather(hours=24)
    if forecast is not None:
        print(f"Forecast: {len(forecast)} hours")
    
    # Test historical (last week)
    end_date = datetime.now().date() - timedelta(days=7)  # Use past dates
    start_date = end_date - timedelta(days=7)
    historical = fetcher.get_historical_weather(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    if historical is not None:
        print(f"Historical: {len(historical)} hours") 