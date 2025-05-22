import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# Define data
data = {
    'ID_TRAMO': [101, 102, 103, 104],
    'timestamp': pd.to_datetime([
        '2023-03-15 08:30:00',
        '2023-03-15 09:15:00',
        '2023-03-15 09:45:00',
        '2023-03-15 10:30:00'
    ]),
    'latitude': [41.3900, 41.3910, 40.7580, 40.7590], # Barcelona and NY area coordinates
    'longitude': [-73.9840, -73.9850, -73.9855, -73.9865],
    'some_numeric_feature': [10.5, 12.3, 15.1, 18.0],
    'categorical_feature': ['A', 'B', 'A', 'C']
}
df = pd.DataFrame(data)

# Create geometry
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326") # WGS84

# Define output path
output_dir = Path(__file__).parent
output_file = output_dir / "parking_predictions_with_pois.parquet"

# Save to GeoParquet
try:
    gdf.to_parquet(output_file)
    print(f"Successfully created GeoParquet file: {output_file}")
except Exception as e:
    print(f"Error creating GeoParquet file: {e}")
