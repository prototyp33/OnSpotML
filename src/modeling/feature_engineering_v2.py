"""
Feature Engineering Pipeline for parking occupancy prediction.

This module provides a comprehensive pipeline for creating and managing features
used in the parking occupancy prediction model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import holidays
import os
from shapely.geometry import Point
import geopandas as gpd
import ray
from src.data_ingestion.weather_data_fetcher import WeatherDataFetcher
from src.data_ingestion.event_data_fetcher import EventDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """Pipeline for creating and managing features for parking occupancy prediction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            config: Optional configuration dictionary containing pipeline parameters
        """
        self.config = config or {}
        self.scalers = {}
        self.feature_groups = {
            'temporal': self.create_temporal_features,
            'lag': self.create_lag_features,
            'weather': self.create_weather_features,
            'events': self.create_event_features,
            'transport': self.create_transport_features,
            'facility': self.create_facility_features,
            'poi': self.create_poi_features,
            'spatial': self.create_spatial_features
        }
        
        # Initialize data fetchers
        self.weather_fetcher = WeatherDataFetcher(self.config.get('weather', {}))
        self.event_fetcher = EventDataFetcher(self.config.get('events', {}))
        
        # Target CRS for reprojection
        self.target_crs = self.config.get('spatial', {}).get('target_crs', "EPSG:25831")
        
        # Initialize Ray if parallel processing is enabled
        if self.config.get('parallel_processing', {}).get('enabled', False):
            try:
                ray.init(ignore_reinit_error=True)
                logger.info("Ray initialized for parallel processing")
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {e}")
    
    def __del__(self):
        """Cleanup when the pipeline is destroyed."""
        if self.config.get('parallel_processing', {}).get('enabled', False):
            try:
                ray.shutdown()
                logger.info("Ray shutdown completed")
            except Exception as e:
                logger.warning(f"Failed to shutdown Ray: {e}")
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            ValueError: If data validation fails
        """
        # Check required columns
        base_required_columns = ['timestamp', 'parking_id']
        optional_raw_columns = ['available_spaces', 'total_spaces']
        
        missing_base = [col for col in base_required_columns if col not in df.columns]
        if missing_base:
            raise ValueError(f"Missing base required columns: {missing_base}")

        missing_optional_raw = [col for col in optional_raw_columns if col not in df.columns]
        if missing_optional_raw:
            logger.warning(f"Optional raw columns {missing_optional_raw} are missing. Some features depending on them might be skipped or behave differently.")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError("timestamp column must be datetime type")
        
        # Conditional checks for optional raw columns
        if 'available_spaces' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['available_spaces']):
                raise ValueError("available_spaces must be numeric if present")
            if (df['available_spaces'] < 0).any():
                raise ValueError("available_spaces cannot be negative if present")
            if 'total_spaces' in df.columns and (df['available_spaces'] > df['total_spaces']).any():
                raise ValueError("available_spaces cannot be greater than total_spaces if both are present")
        
        if 'total_spaces' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['total_spaces']):
                raise ValueError("total_spaces must be numeric if present")
            if (df['total_spaces'] <= 0).any():
                raise ValueError("total_spaces must be positive if present")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            df: Input DataFrame containing raw data
            
        Returns:
            DataFrame with all features created
        """
        logger.info("Creating features...")
        
        # Validate input data
        self.validate_data(df)
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Create features for each group
        for group_name, feature_func in self.feature_groups.items():
            try:
                df = feature_func(df)
                logger.info(f"Created {group_name} features")
            except Exception as e:
                logger.warning(f"Error creating {group_name} features: {e}")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamp."""
        df = df.copy()
        
        # Extract basic temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Create cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Create binary features
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Add time of day bins
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Add peak hours indicators
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # Add lunch hours indicator
        df['is_lunch_hours'] = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
        
        # Add holiday features
        es_holidays = holidays.ES()
        df['is_public_holiday'] = df['timestamp'].dt.date.apply(lambda x: x in es_holidays).astype(int)
        
        # Add holiday proximity features
        df['days_to_holiday'] = df['timestamp'].dt.date.apply(
            lambda x: min(abs((x - h).days) for h in es_holidays.keys())
        )
        df['is_near_holiday'] = (df['days_to_holiday'] <= 2).astype(int)
        
        # Add one-hot encoding for time of day
        time_of_day_dummies = pd.get_dummies(df['time_of_day'], prefix='time_of_day')
        df = pd.concat([df, time_of_day_dummies], axis=1)
        
        # Drop the original categorical column
        df = df.drop('time_of_day', axis=1)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series analysis."""
        df = df.copy()
        
        # Sort by parking_id and timestamp
        df = df.sort_values(['parking_id', 'timestamp'])
        
        # Calculate occupancy rate if not present and raw columns are available
        if 'occupancy_rate' not in df.columns:
            if 'available_spaces' in df.columns and 'total_spaces' in df.columns:
                # Ensure total_spaces is not zero to prevent division by zero errors
                df['occupancy_rate'] = 1 - (df['available_spaces'] / df['total_spaces'].replace(0, np.nan))
                df['occupancy_rate'] = np.nan_to_num(df['occupancy_rate']) # Handle NaN from division by zero if any
                logger.info("Calculated 'occupancy_rate' from available_spaces and total_spaces.")
            else:
                logger.warning("'occupancy_rate' not found and cannot be calculated as 'available_spaces' or 'total_spaces' are missing. Lag features for occupancy_rate might be affected.")
        
        # Create lag features for occupancy if occupancy_rate exists
        if 'occupancy_rate' in df.columns:
            lag_periods = [1, 2, 3, 6, 12, 24]  # 5min intervals
            for lag in lag_periods:
                df[f'occupancy_rate_lag_{lag}'] = df.groupby('parking_id')['occupancy_rate'].shift(lag)
            
            # Add rolling statistics for different time windows
            windows = [6, 12, 24, 48]  # 30min, 1h, 2h, 4h
            for window in windows:
                df[f'occupancy_rate_rolling_mean_{window}'] = df.groupby('parking_id')['occupancy_rate'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'occupancy_rate_rolling_std_{window}'] = df.groupby('parking_id')['occupancy_rate'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                df[f'occupancy_rate_rolling_min_{window}'] = df.groupby('parking_id')['occupancy_rate'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                df[f'occupancy_rate_rolling_max_{window}'] = df.groupby('parking_id')['occupancy_rate'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
            
            # Add rate of change features for occupancy_rate
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'occupancy_rate_roc_{lag}'] = df.groupby('parking_id')['occupancy_rate'].transform(
                    lambda x: x.pct_change(periods=lag)
                )
                df[f'occupancy_rate_roc_{lag}'] = np.nan_to_num(df[f'occupancy_rate_roc_{lag}'])

            # Add exponential weighted moving averages for occupancy_rate
            for span in [6, 12, 24]:
                df[f'occupancy_rate_ewm_{span}'] = df.groupby('parking_id')['occupancy_rate'].transform(
                    lambda x: x.ewm(span=span, min_periods=1).mean()
                )
            
            # Add same time yesterday/week features for occupancy_rate
            if not df['timestamp'].dt.tz:
                df['occupancy_rate_same_time_yesterday'] = df.groupby(['parking_id', df['timestamp'].dt.hour, df['timestamp'].dt.minute])['occupancy_rate'].transform(lambda x: x.shift(1*24*12)) # Assumes 5-min interval data, 288 periods for 1 day
                df['occupancy_rate_same_time_last_week'] = df.groupby(['parking_id', df['timestamp'].dt.dayofweek, df['timestamp'].dt.hour, df['timestamp'].dt.minute])['occupancy_rate'].transform(lambda x: x.shift(1*7*24*12)) # 2016 periods for 1 week
            else:
                logger.warning("Timestamp is timezone aware, skipping same time yesterday/week features for occupancy rate due to potential complexity with TZ.")

        # Lag features for available_spaces (if present)
        if 'available_spaces' in df.columns:
            lag_periods = [1, 2, 3, 6, 12, 24]
            for lag in lag_periods:
                df[f'available_spaces_lag_{lag}'] = df.groupby('parking_id')['available_spaces'].shift(lag)
            # Add rate of change features for available_spaces
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'available_spaces_roc_{lag}'] = df.groupby('parking_id')['available_spaces'].transform(
                    lambda x: x.pct_change(periods=lag)
                )
                df[f'available_spaces_roc_{lag}'] = np.nan_to_num(df[f'available_spaces_roc_{lag}'])
        else:
            logger.warning("'available_spaces' not found. Lag features for available_spaces will be skipped.")

        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features.
        
        Args:
            df: Input DataFrame containing parking data
            
        Returns:
            DataFrame with weather features added
        """
        df = df.copy()
        
        # Get date range from input data
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        # Fetch weather data
        weather_data = self.weather_fetcher.fetch_weather_data(start_date, end_date)
        
        if weather_data.empty:
            logger.warning("No weather data available, skipping weather feature creation")
            return df
        
        # Merge weather data with input data
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            weather_data.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Add weather-related features
        weather_features = [
            'temperature', 'feels_like', 'humidity', 'pressure',
            'wind_speed', 'wind_sin', 'wind_cos', 'clouds',
            'is_raining', 'is_snowing', 'precipitation'
        ]
        
        # Create lag features for weather
        for feature in weather_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 6, 12, 24]:  # 5min intervals
                    df[f'{feature}_lag_{lag}'] = df.groupby('parking_id')[feature].shift(lag)
        
        # Create rolling statistics for weather features
        for feature in weather_features:
            if feature in df.columns:
                for window in [6, 12, 24]:  # 30min, 1h, 2h
                    df[f'{feature}_rolling_mean_{window}'] = df.groupby('parking_id')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df[f'{feature}_rolling_std_{window}'] = df.groupby('parking_id')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
        
        return df
    
    def create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create event-related features.
        
        Args:
            df: Input DataFrame containing parking data
            
        Returns:
            DataFrame with event features added
        """
        df = df.copy()
        
        # Get date range from input data
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        # Fetch event data
        event_data = self.event_fetcher.fetch_event_data(start_date, end_date)
        
        if event_data.empty:
            logger.warning("No event data available, skipping event feature creation")
            return df
        
        # Create GeoDataFrames for spatial operations
        parking_points = gpd.GeoDataFrame(
            df, 
            geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
            crs="EPSG:4326"
        )
        
        event_points = gpd.GeoDataFrame(
            event_data,
            geometry=[Point(xy) for xy in zip(event_data['longitude'], event_data['latitude'])],
            crs="EPSG:4326"
        )
        
        # Reproject to target CRS for accurate distance calculations
        if self.target_crs:
            parking_points = parking_points.to_crs(self.target_crs)
            event_points = event_points.to_crs(self.target_crs)
        
        # Get event configuration
        event_config = self.config.get('feature_engineering', {}).get('event_features', {})
        radii = event_config.get('radii', [500, 1000, 2000])  # meters
        
        # Create event features for each radius
        for radius in radii:
            # Count events within radius
            df[f'event_count_{radius}m'] = parking_points.geometry.apply(
                lambda x: len(event_points[event_points.geometry.distance(x) <= radius])
            )
            
            # Calculate event importance within radius
            df[f'event_importance_{radius}m'] = parking_points.geometry.apply(
                lambda x: event_points[event_points.geometry.distance(x) <= radius]['importance'].sum()
            )
            
            # Count major events within radius
            df[f'major_event_count_{radius}m'] = parking_points.geometry.apply(
                lambda x: len(event_points[
                    (event_points.geometry.distance(x) <= radius) & 
                    (event_points['is_major_event'] == 1)
                ])
            )
            
            # Calculate event density
            area = np.pi * (radius ** 2)  # Area in square meters
            df[f'event_density_{radius}m'] = df[f'event_count_{radius}m'] / area
            
            # Calculate weighted event importance
            df[f'weighted_event_importance_{radius}m'] = df[f'event_importance_{radius}m'] / area
        
        # Create time-based event features
        df['hours_until_next_event'] = df.apply(
            lambda row: self._get_hours_until_next_event(row['timestamp'], event_data), axis=1
        )
        
        df['hours_since_last_event'] = df.apply(
            lambda row: self._get_hours_since_last_event(row['timestamp'], event_data), axis=1
        )
        
        # Create event impact features
        df['is_event_time'] = df['hours_until_next_event'] <= 2  # Within 2 hours of an event
        df['is_major_event_time'] = df.apply(
            lambda row: self._is_major_event_time(row['timestamp'], event_data), axis=1
        )
        
        return df
    
    def _get_hours_until_next_event(self, timestamp: datetime, event_data: pd.DataFrame) -> float:
        """Calculate hours until the next event."""
        future_events = event_data[event_data['timestamp'] > timestamp]
        if future_events.empty:
            return float('inf')
        return (future_events.iloc[0]['timestamp'] - timestamp).total_seconds() / 3600
    
    def _get_hours_since_last_event(self, timestamp: datetime, event_data: pd.DataFrame) -> float:
        """Calculate hours since the last event."""
        past_events = event_data[event_data['timestamp'] < timestamp]
        if past_events.empty:
            return float('inf')
        return (timestamp - past_events.iloc[-1]['timestamp']).total_seconds() / 3600
    
    def _is_major_event_time(self, timestamp: datetime, event_data: pd.DataFrame) -> bool:
        """Check if the timestamp is within 2 hours of a major event."""
        time_window = timedelta(hours=2)
        nearby_events = event_data[
            (event_data['timestamp'] >= timestamp - time_window) &
            (event_data['timestamp'] <= timestamp + time_window) &
            (event_data['is_major_event'] == 1)
        ]
        return len(nearby_events) > 0
    
    def create_transport_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transport-related features."""
        df = df.copy()
        # This is a placeholder - implement actual transport feature creation
        return df
    
    def create_facility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create facility-related features."""
        df = df.copy()
        
        # Add facility features
        df['is_open_now'] = 1  # Placeholder - implement actual logic
        df['hours_until_close'] = 0  # Placeholder - implement actual logic
        df['hours_since_open'] = 0  # Placeholder - implement actual logic
        
        return df
    
    def create_poi_features(self, df: pd.DataFrame, poi_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create POI-related features.
        
        Args:
            df: Input DataFrame containing parking data
            poi_data: Optional DataFrame containing POI data
            
        Returns:
            DataFrame with POI features added
        """
        df = df.copy()
        
        if poi_data is None:
            logger.warning("No POI data provided, skipping POI feature creation")
            return df
        
        # Ensure required columns exist
        required_cols = ['latitude', 'longitude']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for POI features, skipping")
            return df
        
        # Create GeoDataFrames
        parking_points = gpd.GeoDataFrame(
            df, 
            geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
            crs="EPSG:4326"
        )
        
        poi_points = gpd.GeoDataFrame(
            poi_data,
            geometry=[Point(xy) for xy in zip(poi_data['longitude'], poi_data['latitude'])],
            crs="EPSG:4326"
        )
        
        # Reproject to target CRS for accurate distance calculations
        if self.target_crs:
            parking_points = parking_points.to_crs(self.target_crs)
            poi_points = poi_points.to_crs(self.target_crs)
            
        # Get POI configuration
        poi_config = self.config.get('feature_engineering', {}).get('poi_features', {})
        radii = poi_config.get('radii', [100, 200, 500])
        categories = poi_config.get('categories', ['restaurant', 'shop', 'entertainment', 'transport'])
        
        # Create POI features for each radius and category
        for radius in radii:
            for category in categories:
                # Filter POIs by category
                category_pois = poi_points[poi_points['category'] == category]
                
                if len(category_pois) > 0:
                    # Count POIs within radius
                    df[f'poi_{category}_count_{radius}m'] = parking_points.geometry.apply(
                        lambda x: len(category_pois[category_pois.geometry.distance(x) <= radius])
                    )
                    
                    # Calculate POI density
                    area = np.pi * (radius ** 2)  # Area in square meters
                    df[f'poi_{category}_density_{radius}m'] = df[f'poi_{category}_count_{radius}m'] / area
                    
                    # Calculate weighted importance
                    df[f'poi_{category}_importance_{radius}m'] = parking_points.geometry.apply(
                        lambda x: category_pois[category_pois.geometry.distance(x) <= radius]['importance'].sum()
                    )
        
        return df
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features for parking locations."""
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['latitude', 'longitude', 'parking_id']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for spatial features, skipping")
            return df
        
        # Create GeoDataFrame
        parking_points_orig_crs = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
            crs="EPSG:4326"
        )

        parking_points = parking_points_orig_crs.copy()
        if self.target_crs:
            parking_points = parking_points.to_crs(self.target_crs)

        # Calculate distances between parking locations
        parking_locations = parking_points[['parking_id', 'geometry']].drop_duplicates()
        distance_matrix = np.zeros((len(parking_locations), len(parking_locations)))
        
        for i, loc1 in enumerate(parking_locations.itertuples()):
            for j, loc2 in enumerate(parking_locations.itertuples()):
                if i != j:
                    distance_matrix[i, j] = loc1.geometry.distance(loc2.geometry)
        
        # Create distance-based features
        for i, loc in enumerate(parking_locations.itertuples()):
            # Find nearest 3 parking locations
            distances = distance_matrix[i]
            nearest_indices = np.argsort(distances)[1:4]  # Exclude self
            nearest_parking_ids = parking_locations.iloc[nearest_indices]['parking_id'].values
            
            # Add features for each nearest location
            for j, near_id in enumerate(nearest_parking_ids):
                df[f'nearest_{j+1}_parking_distance'] = np.where(
                    df['parking_id'] == loc.parking_id,
                    distances[nearest_indices[j]],
                    0
                )
        
        # Add zone-based features
        # Create a simple grid-based zone system
        df['zone_lat'] = np.floor(df['latitude'] * 100) / 100
        df['zone_lon'] = np.floor(df['longitude'] * 100) / 100
        df['zone_id'] = df['zone_lat'].astype(str) + '_' + df['zone_lon'].astype(str)
        
        # Calculate zone-level statistics
        if 'occupancy_rate' not in df.columns:
             if 'available_spaces' in df.columns and 'total_spaces' in df.columns:
                 df['occupancy_rate'] = 1 - (df['available_spaces'] / df['total_spaces'].replace(0, np.nan))
                 df['occupancy_rate'] = np.nan_to_num(df['occupancy_rate'])
             else:
                 # If essential columns for occupancy_rate are missing, log and skip zone stats
                 logger.warning("Cannot calculate zone_avg_occupancy as 'available_spaces' or 'total_spaces' are missing for occupancy_rate calculation.")
                 # Return df as is, or with fewer spatial features if others can still be made
                 return df


        zone_stats_cols_to_create = ['zone_avg_occupancy', 'zone_occupancy_std', 'zone_parking_count']
        for col in zone_stats_cols_to_create:
            if col in df.columns:
                df = df.drop(columns=[col])

        zone_stats = df.groupby('zone_id').agg({
            'occupancy_rate': ['mean', 'std', 'count']
        }).reset_index()
        
        zone_stats.columns = ['zone_id'] + zone_stats_cols_to_create
        
        # Merge zone statistics back to main dataframe
        df = df.merge(zone_stats, on='zone_id', how='left')
        
        # Add interaction features
        # Replace 0 in denominator with NaN to avoid division by zero, then handle NaN/inf
        df['zone_occupancy_ratio'] = df['occupancy_rate'] / df['zone_avg_occupancy'].replace(0, np.nan)
        df['zone_occupancy_ratio'] = np.nan_to_num(df['zone_occupancy_ratio'])
        
        df['zone_occupancy_diff'] = df['occupancy_rate'] - df['zone_avg_occupancy']
        df['zone_occupancy_diff'] = np.nan_to_num(df['zone_occupancy_diff'])
        
        # Add distance to city center (assuming Barcelona center coordinates)
        bcn_center = Point(2.1734, 41.3851)  # Barcelona center coordinates
        
        # Create a GeoSeries for the center point with the original CRS, then reproject
        center_gs = gpd.GeoSeries([bcn_center], crs="EPSG:4326")
        if self.target_crs:
            center_gs = center_gs.to_crs(self.target_crs)
        reprojected_center = center_gs[0]
        
        df['distance_to_center'] = parking_points.geometry.distance(reprojected_center)
        
        # Add distance-based features
        # Note: These thresholds might need adjustment after CRS change as distances are now in meters
        df['is_central'] = (df['distance_to_center'] <= 2000).astype(int)  # Within 2km of center
        df['is_peripheral'] = (df['distance_to_center'] > 5000).astype(int)  # More than 5km from center
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline and transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        # Create features
        df = self.create_features(df)
        
        # Scale numerical features if needed
        if self.config.get('scale_features', False):
            df = self._scale_features(df)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.fit_transform(df)
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Scale each numerical column
        for col in numerical_cols:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                df[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                df[col] = self.scalers[col].transform(df[[col]])
        
        return df 