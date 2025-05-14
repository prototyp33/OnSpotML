import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from datetime import datetime, timedelta

class GTFSFeatureEngineer:
    def __init__(self, gtfs_path):
        """Initialize GTFS feature engineer with path to GTFS files."""
        self.gtfs_path = gtfs_path
        self.stops = None
        self.stop_times = None
        self.routes = None
        self.trips = None
        self.calendar = None
        
    def load_gtfs_data(self):
        """Load required GTFS files."""
        try:
            self.stops = pd.read_csv(f"{self.gtfs_path}/stops.txt")
            self.stop_times = pd.read_csv(f"{self.gtfs_path}/stop_times.txt")
            self.routes = pd.read_csv(f"{self.gtfs_path}/routes.txt")
            self.trips = pd.read_csv(f"{self.gtfs_path}/trips.txt")
            self.calendar = pd.read_csv(f"{self.gtfs_path}/calendar.txt")
            
            # Convert stops to GeoDataFrame
            self.stops = gpd.GeoDataFrame(
                self.stops,
                geometry=gpd.points_from_xy(self.stops.stop_lon, self.stops.stop_lat)
            )
            
        except Exception as e:
            print(f"Error loading GTFS data: {e}")
            raise
    
    def calculate_stop_density(self, point, radius_meters=500):
        """Calculate number of stops within a given radius."""
        if not isinstance(self.stops, gpd.GeoDataFrame):
            self.load_gtfs_data()
            
        point = Point(point)
        stops_within_radius = self.stops[self.stops.geometry.distance(point) <= radius_meters/111000]
        return len(stops_within_radius)
    
    def calculate_service_frequency(self, stop_id, time_window='1H'):
        """Calculate average service frequency for a stop."""
        if self.stop_times is None:
            self.load_gtfs_data()
            
        # Convert arrival_time to datetime
        stop_times = self.stop_times.copy()
        stop_times['arrival_time'] = pd.to_datetime(stop_times['arrival_time'], format='%H:%M:%S')
        
        # Filter for specific stop
        stop_times = stop_times[stop_times['stop_id'] == stop_id]
        
        # Calculate frequency
        if time_window == '1H':
            return len(stop_times) / 24  # Average trips per hour
        elif time_window == '1D':
            return len(stop_times)  # Total trips per day
    
    def get_route_coverage(self, point, radius_meters=500):
        """Get number of unique routes serving stops within radius."""
        if not isinstance(self.stops, gpd.GeoDataFrame):
            self.load_gtfs_data()
            
        point = Point(point)
        stops_within_radius = self.stops[self.stops.geometry.distance(point) <= radius_meters/111000]
        
        # Get unique routes for these stops
        stop_times_filtered = self.stop_times[self.stop_times['stop_id'].isin(stops_within_radius['stop_id'])]
        trip_ids = stop_times_filtered['trip_id'].unique()
        routes = self.trips[self.trips['trip_id'].isin(trip_ids)]['route_id'].unique()
        
        return len(routes)
    
    def create_gtfs_features(self, coordinates):
        """Create comprehensive GTFS features for given coordinates."""
        features = []
        
        for coord in coordinates:
            stop_density = self.calculate_stop_density(coord)
            route_coverage = self.get_route_coverage(coord)
            
            # Find nearest stop
            point = Point(coord)
            distances = self.stops.geometry.distance(point)
            nearest_stop = self.stops.iloc[distances.idxmin()]
            nearest_stop_id = nearest_stop['stop_id']
            
            service_frequency = self.calculate_service_frequency(nearest_stop_id)
            
            features.append({
                'stop_density_500m': stop_density,
                'route_coverage_500m': route_coverage,
                'nearest_stop_distance': distances.min() * 111000,  # Convert to meters
                'nearest_stop_service_frequency': service_frequency
            })
        
        return pd.DataFrame(features) 