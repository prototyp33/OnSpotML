import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import osmnx as ox
from shapely.geometry import Point
import geopandas as gpd

logger = logging.getLogger(__name__)

def create_lagged_features(
    df: pd.DataFrame,
    lag_periods: List[int] = [1, 2, 3, 4, 6, 12, 24],  # Hours
    target_cols: List[str] = ['available_spaces', 'occupancy_rate']
) -> pd.DataFrame:
    """
    Create lagged features for target columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp and target columns
    lag_periods : List[int]
        List of lag periods in hours
    target_cols : List[str]
        List of columns to create lagged features for
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added lagged features
    """
    logger.info("Creating lagged features...")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by parking_id and timestamp
    df = df.sort_values(['parking_id', 'timestamp'])
    
    # Create lagged features for each target column
    for col in target_cols:
        for lag in lag_periods:
            # Calculate lag in minutes
            lag_minutes = lag * 60
            # Create lagged feature
            df[f'{col}_lag_{lag}h'] = df.groupby('parking_id')[col].shift(lag_minutes)
    
    return df

def create_facility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create static and dynamic facility features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with facility information
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added facility features
    """
    logger.info("Creating facility features...")
    
    # Static features (assuming these columns exist)
    static_features = [
        'parking_type',
        'pricing_tier',
        'is_24h',
        'has_security',
        'has_ev_charging',
        'is_underground',
        'is_public'
    ]
    
    # Dynamic features
    df['is_open_now'] = df.apply(
        lambda row: is_facility_open(
            row['timestamp'],
            row['opening_hours'],
            row['is_24h']
        ),
        axis=1
    )
    
    df['hours_until_close'] = df.apply(
        lambda row: hours_until_close(
            row['timestamp'],
            row['opening_hours'],
            row['is_24h']
        ),
        axis=1
    )
    
    df['hours_since_open'] = df.apply(
        lambda row: hours_since_open(
            row['timestamp'],
            row['opening_hours'],
            row['is_24h']
        ),
        axis=1
    )
    
    return df

def create_poi_features(
    df: pd.DataFrame,
    poi_types: List[str] = [
        'restaurant', 'cafe', 'bar', 'shop', 'supermarket',
        'school', 'university', 'hospital', 'office', 'attraction'
    ],
    radius: float = 500  # meters
) -> pd.DataFrame:
    """
    Create POI features for each parking facility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with parking facility locations
    poi_types : List[str]
        List of POI types to consider
    radius : float
        Search radius in meters
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added POI features
    """
    logger.info("Creating POI features...")
    
    # Convert coordinates to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
        crs="EPSG:4326"
    )
    
    # Create buffer for each point
    gdf['buffer'] = gdf.geometry.buffer(radius / 111000)  # Convert meters to degrees
    
    # Download POIs for each type
    for poi_type in poi_types:
        logger.info(f"Processing {poi_type} POIs...")
        
        # Download POIs
        pois = ox.geometries_from_place(
            'Barcelona, Spain',
            tags={'amenity': poi_type}
        )
        
        # Calculate features for each parking facility
        for idx, row in gdf.iterrows():
            # Count POIs in buffer
            poi_count = len(pois[pois.geometry.within(row['buffer'])])
            df.loc[idx, f'poi_{poi_type}_count'] = poi_count
            
            # Calculate density
            df.loc[idx, f'poi_{poi_type}_density'] = poi_count / (np.pi * (radius/1000)**2)
            
            # Calculate distance to nearest POI
            if poi_count > 0:
                nearest_poi = pois[pois.geometry.within(row['buffer'])].geometry.unary_union
                distance = row.geometry.distance(nearest_poi) * 111000  # Convert to meters
                df.loc[idx, f'poi_{poi_type}_nearest_distance'] = distance
            else:
                df.loc[idx, f'poi_{poi_type}_nearest_distance'] = radius
    
    return df

def is_facility_open(
    timestamp: datetime,
    opening_hours: str,
    is_24h: bool
) -> bool:
    """Check if facility is open at given timestamp."""
    if is_24h:
        return True
    
    # Parse opening hours (assuming format like "09:00-17:00")
    try:
        open_time, close_time = opening_hours.split('-')
        open_time = datetime.strptime(open_time.strip(), '%H:%M').time()
        close_time = datetime.strptime(close_time.strip(), '%H:%M').time()
        
        current_time = timestamp.time()
        return open_time <= current_time <= close_time
    except:
        return False

def hours_until_close(
    timestamp: datetime,
    opening_hours: str,
    is_24h: bool
) -> float:
    """Calculate hours until facility closes."""
    if is_24h:
        return 24.0
    
    try:
        _, close_time = opening_hours.split('-')
        close_time = datetime.strptime(close_time.strip(), '%H:%M').time()
        close_datetime = datetime.combine(timestamp.date(), close_time)
        
        if timestamp.time() > close_time:
            close_datetime += timedelta(days=1)
        
        return (close_datetime - timestamp).total_seconds() / 3600
    except:
        return 0.0

def hours_since_open(
    timestamp: datetime,
    opening_hours: str,
    is_24h: bool
) -> float:
    """Calculate hours since facility opened."""
    if is_24h:
        return 0.0
    
    try:
        open_time, _ = opening_hours.split('-')
        open_time = datetime.strptime(open_time.strip(), '%H:%M').time()
        open_datetime = datetime.combine(timestamp.date(), open_time)
        
        if timestamp.time() < open_time:
            open_datetime -= timedelta(days=1)
        
        return (timestamp - open_datetime).total_seconds() / 3600
    except:
        return 0.0 