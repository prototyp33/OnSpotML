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
            'poi': self.create_poi_features
        }
        
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
        required_columns = ['timestamp', 'parking_id', 'available_spaces', 'total_spaces']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError("timestamp column must be datetime type")
        
        # Check value ranges
        if (df['available_spaces'] < 0).any():
            raise ValueError("available_spaces cannot be negative")
        
        if (df['available_spaces'] > df['total_spaces']).any():
            raise ValueError("available_spaces cannot be greater than total_spaces")
        
        if (df['total_spaces'] <= 0).any():
            raise ValueError("total_spaces must be positive")
    
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
        
        # Add holiday features
        es_holidays = holidays.ES()
        df['is_public_holiday'] = df['timestamp'].dt.date.apply(lambda x: x in es_holidays).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series analysis."""
        df = df.copy()
        
        # Sort by parking_id and timestamp
        df = df.sort_values(['parking_id', 'timestamp'])
        
        # Calculate occupancy rate if not present
        if 'occupancy_rate' not in df.columns:
            df['occupancy_rate'] = 1 - (df['available_spaces'] / df['total_spaces'])
        
        # Create lag features for occupancy
        for lag in [1, 2, 3, 6, 12, 24]:  # 5min intervals
            df[f'occupancy_rate_lag_{lag}'] = df.groupby('parking_id')['occupancy_rate'].shift(lag)
            df[f'available_spaces_lag_{lag}'] = df.groupby('parking_id')['available_spaces'].shift(lag)
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-related features."""
        df = df.copy()
        # This is a placeholder - implement actual weather feature creation
        return df
    
    def create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create event-related features."""
        df = df.copy()
        # This is a placeholder - implement actual event feature creation
        return df
    
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