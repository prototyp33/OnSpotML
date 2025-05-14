# Required imports
import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
from shapely.geometry import Point, LineString, box
import json
from datetime import datetime, time
import re

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure OSMnx
ox.config(use_cache=True, log_console=True)

# Constants
METERS_PER_PARKING_SPOT = 5.0  # Standard parallel parking spot length
POI_RADIUS = 100  # meters - radius to search for POIs around each segment
POI_TAGS = {
    'retail': ['shop'],
    'food': ['amenity=restaurant', 'amenity=cafe', 'amenity=bar'],
    'office': ['office'],
    'transport': ['highway=bus_stop', 'railway=station', 'railway=subway_entrance'],
    'leisure': ['leisure'],
    'tourism': ['tourism']
}

# File paths
INPUT_FILE = '../data/processed/parking_predictions_processed.parquet'
RAW_JSON_FILE = '../data/raw/parking_predictions_corrected.json'
OUTPUT_FILE = '../data/processed/parking_predictions_phase1_enriched.parquet'

def load_data():
    """Load and prepare the base data for feature engineering."""
    # Load the processed predictions
    df_pred = pd.read_parquet(INPUT_FILE)
    
    # Load the raw JSON to get geometry and zone information
    with open(RAW_JSON_FILE, 'r') as f:
        raw_data = json.load(f)
    
    # Extract features from the first record's TRAMOS
    features = raw_data['OPENDATA_PSIU_APPARKB'][0]['TRAMOS'][0]['features']
    
    # Create a GeoDataFrame with static information
    static_data = []
    for feature in features:
        props = feature['properties']
        geom = feature['geometry']
        static_data.append({
            'ID_TRAMO': props['ID_TRAMO'],
            'TRAMO': props['TRAMO'],
            'TIPO': props['TIPO'],
            'TARIFA': props['TARIFA'],
            'HORARIO': props['HORARIO'],
            'geometry': geom
        })
    
    gdf_static = gpd.GeoDataFrame(static_data)
    gdf_static['geometry'] = gdf_static['geometry'].apply(lambda x: LineString(x['coordinates']))
    gdf_static.set_crs(epsg=4326, inplace=True)
    
    return df_pred, gdf_static

def estimate_capacity(gdf):
    """Estimate parking capacity based on segment geometry length.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame with LineString geometries
        
    Returns:
        Series: Estimated number of parking spots per segment
    """
    # Convert to UTM for accurate measurements
    gdf_utm = gdf.to_crs(epsg=32631)  # UTM zone 31N for Barcelona
    
    # Calculate length in meters
    lengths = gdf_utm.geometry.length
    
    # Calculate capacity (round down to be conservative)
    capacity = (lengths / METERS_PER_PARKING_SPOT).apply(np.floor)
    
    return capacity.astype(int)

def fetch_pois(gdf):
    """Fetch POIs around each segment using OSMnx.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame with LineString geometries
        
    Returns:
        GeoDataFrame: Input GDF with added POI count columns
    """
    # Convert to UTM for buffer operation
    gdf_utm = gdf.to_crs(epsg=32631)
    
    # Get the total extent of all segments plus buffer
    total_bounds = gdf_utm.total_bounds
    bbox = box(*total_bounds).buffer(POI_RADIUS)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=gdf_utm.crs)
    bbox_wgs84 = bbox_gdf.to_crs(epsg=4326)
    
    # Initialize POI count columns
    for poi_type in POI_TAGS.keys():
        gdf[f'poi_{poi_type}_count'] = 0
    
    # Fetch POIs for each category
    for poi_type, tags in POI_TAGS.items():
        print(f"Fetching {poi_type} POIs...")
        
        # Get POIs within the bounding box
        pois = []
        for tag in tags:
            try:
                if '=' in tag:
                    key, value = tag.split('=')
                    poi_gdf = ox.features_from_polygon(
                        bbox_wgs84.iloc[0].geometry,
                        tags={key: value}
                    )
                else:
                    poi_gdf = ox.features_from_polygon(
                        bbox_wgs84.iloc[0].geometry,
                        tags={tag: True}
                    )
                if not poi_gdf.empty:
                    pois.append(poi_gdf)
            except Exception as e:
                print(f"Warning: Error fetching {tag} POIs: {e}")
                continue
        
        if pois:
            # Combine all POIs for this category
            poi_gdf = pd.concat(pois).pipe(gpd.GeoDataFrame)
            poi_gdf.set_crs(epsg=4326, inplace=True)
            poi_gdf_utm = poi_gdf.to_crs(epsg=32631)
            
            # Count POIs within radius of each segment
            for idx, row in gdf_utm.iterrows():
                buffer = row.geometry.buffer(POI_RADIUS)
                count = sum(poi_gdf_utm.geometry.intersects(buffer))
                gdf.at[idx, f'poi_{poi_type}_count'] = count
    
    return gdf

def parse_zone_properties(gdf):
    """Parse TARIFA and HORARIO strings to extract features.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame with TARIFA and HORARIO columns
        
    Returns:
        GeoDataFrame: Input GDF with added zone property columns
    """
    # Parse TARIFA
    def parse_tarifa(tarifa):
        if pd.isna(tarifa):
            return 0.0
        # Extract numeric value, assuming format like "2.50 EUR/h"
        match = re.search(r'(\d+(?:\.\d+)?)', str(tarifa))
        return float(match.group(1)) if match else 0.0
    
    # Parse HORARIO
    def parse_horario(horario):
        if pd.isna(horario):
            return {
                'start_hour': 8,
                'end_hour': 20,
                'total_hours': 12,
                'is_24h': False
            }
        
        horario = str(horario).upper()
        
        # Check for 24h
        if '24H' in horario:
            return {
                'start_hour': 0,
                'end_hour': 24,
                'total_hours': 24,
                'is_24h': True
            }
        
        # Extract hours, assuming format like "8:00-20:00"
        match = re.search(r'(\d{1,2})(?::\d{2})?\s*-\s*(\d{1,2})(?::\d{2})?', horario)
        if match:
            start_hour = int(match.group(1))
            end_hour = int(match.group(2))
            total_hours = end_hour - start_hour
            return {
                'start_hour': start_hour,
                'end_hour': end_hour,
                'total_hours': total_hours,
                'is_24h': False
            }
        
        # Default values if parsing fails
        return {
            'start_hour': 8,
            'end_hour': 20,
            'total_hours': 12,
            'is_24h': False
        }
    
    # Apply parsing functions
    print("Parsing zone properties...")
    gdf['tarifa_rate'] = gdf['TARIFA'].apply(parse_tarifa)
    
    horario_features = gdf['HORARIO'].apply(parse_horario)
    for feature in ['start_hour', 'end_hour', 'total_hours', 'is_24h']:
        gdf[f'horario_{feature}'] = horario_features.apply(lambda x: x[feature])
    
    return gdf

def main():
    # Load data
    print("Loading data...")
    df_predictions, gdf_static = load_data()
    print(f"Loaded {len(df_predictions):,} predictions and {len(gdf_static)} static records.")
    
    # Calculate capacity
    print("\nCalculating parking capacity...")
    gdf_static['estimated_capacity'] = estimate_capacity(gdf_static)
    print("Capacity statistics:")
    print(gdf_static['estimated_capacity'].describe())
    
    # Fetch POIs
    print("\nFetching POIs...")
    gdf_static = fetch_pois(gdf_static)
    print("\nPOI statistics:")
    poi_cols = [col for col in gdf_static.columns if col.startswith('poi_')]
    print(gdf_static[poi_cols].describe())
    
    # Parse zone properties
    gdf_static = parse_zone_properties(gdf_static)
    print("\nZone property statistics:")
    zone_cols = ['tarifa_rate'] + [col for col in gdf_static.columns if col.startswith('horario_')]
    print(gdf_static[zone_cols].describe())
    
    # Combine features with predictions
    print("\nCombining features with predictions...")
    
    # Select features to join
    feature_cols = [
        'ID_TRAMO',
        'estimated_capacity',
        'tarifa_rate',
        'horario_start_hour',
        'horario_end_hour',
        'horario_total_hours',
        'horario_is_24h'
    ] + [col for col in gdf_static.columns if col.startswith('poi_')]
    
    # Join features to predictions
    df_enriched = df_predictions.merge(
        gdf_static[feature_cols],
        on='ID_TRAMO',
        how='left'
    )
    
    # Add hour of day feature
    df_enriched['hour'] = df_enriched['timestamp'].dt.hour
    
    # Save enriched dataset
    print(f"\nSaving enriched dataset to {OUTPUT_FILE}")
    print(f"Final shape: {df_enriched.shape}")
    print("\nFeature summary:")
    print(df_enriched.info())
    
    df_enriched.to_parquet(OUTPUT_FILE, index=False)
    print("\nFeature engineering completed successfully!")

if __name__ == "__main__":
    main() 