import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .gtfs_features import GTFSFeatureEngineer
from .meteorological_features import MeteorologicalFeatureEngineer
from .temporal_features import TemporalFeatureEngineer
from .spatial_features import SpatialFeatureEngineer

class FeatureEngineeringPipeline:
    def __init__(self, config):
        """Initialize feature engineering pipeline with configuration."""
        self.config = config
        self.gtfs_engineer = GTFSFeatureEngineer(config['gtfs_path'])
        self.meteo_engineer = MeteorologicalFeatureEngineer(config['meteo_path'])
        self.temporal_engineer = TemporalFeatureEngineer()
        self.spatial_engineer = SpatialFeatureEngineer(config['spatial_data_path'])
        
    def create_features(self, df):
        """Create comprehensive feature set for parking prediction."""
        # Create temporal features
        temporal_features = self.temporal_engineer.create_temporal_features(df)
        
        # Create meteorological features
        meteo_features = self.meteo_engineer.create_meteorological_features(df)
        
        # Create spatial features
        spatial_features = self.spatial_engineer.create_spatial_features(df)
        
        # Create GTFS features
        coordinates = list(zip(df['longitude'], df['latitude']))
        gtfs_features = self.gtfs_engineer.create_gtfs_features(coordinates)
        
        # Combine all features
        features = pd.concat([
            temporal_features,
            meteo_features,
            spatial_features,
            gtfs_features
        ], axis=1)
        
        return features
    
    def get_feature_names(self):
        """Get names of all features created by the pipeline."""
        temporal_names = self.temporal_engineer.get_feature_names()
        meteo_names = self.meteo_engineer.get_feature_names()
        spatial_names = self.spatial_engineer.get_feature_names()
        gtfs_names = [
            'stop_density_500m',
            'route_coverage_500m',
            'nearest_stop_distance',
            'nearest_stop_service_frequency'
        ]
        
        return temporal_names + meteo_names + spatial_names + gtfs_names 