import pandas as pd
import geopandas as gpd
from pathlib import Path

# Define the path to the Parquet file as a constant
PARQUET_FILE_PATH = Path("/Users/adrianiraeguialvear/Desktop/OnSpotML_v2/data/processed/features_master_table.parquet")

def load_geodataframe(file_path: Path) -> gpd.GeoDataFrame | pd.DataFrame | None:
    """
    Loads a Parquet file into a GeoDataFrame, with fallback to Pandas DataFrame.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None

    try:
        gdf = gpd.read_parquet(file_path)
        print(f"Successfully loaded '{file_path.name}' as GeoDataFrame.")
        print(f"CRS of loaded GDF: {gdf.crs}")
        if 'geometry' in gdf.columns and not gdf.empty:
            print(f"Geometry column type: {gdf.geometry.geom_type.unique()}")
        return gdf
    except Exception as e_gpd:
        print(f"Could not load directly as GeoDataFrame: {e_gpd}")
        print("Attempting to load as Pandas DataFrame first...")
        try:
            df = pd.read_parquet(file_path)
            print(f"Successfully loaded '{file_path.name}' as Pandas DataFrame.")
            
            if 'geometry' in df.columns and not df.empty:
                # Attempt to convert the 'geometry' column to actual geometry objects.
                # This is necessary if gpd.read_parquet failed but the column contains WKB bytes.
                try:
                    from shapely import wkb
                    # Check if the first geometry entry is bytes and not already a Shapely geometry
                    if isinstance(df['geometry'].iloc[0], bytes):
                        print("Attempting to convert WKB bytes in 'geometry' column.")
                        df['geometry'] = df['geometry'].apply(lambda geom_bytes: wkb.loads(geom_bytes) if geom_bytes else None)

                    gdf = gpd.GeoDataFrame(df, geometry='geometry')
                    # Note: CRS is ideally stored in Parquet metadata. If not, it might need manual setting here.
                    print(f"Converted Pandas DataFrame to GeoDataFrame. CRS: {gdf.crs}")
                    return gdf # Fix: Correctly return the newly created GeoDataFrame
                except Exception as e_conv:
                    print(f"Error converting 'geometry' column or creating GeoDataFrame: {e_conv}")
                    return df # Fallback to Pandas DataFrame
            return df # Return as Pandas DataFrame if no geometry column or conversion failed
        except Exception as e_pd:
            print(f"Error loading Parquet file with pandas: {e_pd}")
            return None

def inspect_dataframe(df: pd.DataFrame | gpd.GeoDataFrame):
    """
    Performs basic inspection and cleaning of the DataFrame.
    """
    print("\nFirst 5 rows of the loaded data:")
    print(df.head())

    print("\nData types (dtypes):")
    print(df.dtypes)

    # Convert timestamp if it's not already datetime
    if 'timestamp' in df.columns and df['timestamp'].dtype == 'int64':
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print("\nConverted 'timestamp' to datetime.")
        except Exception as e_ts:
            print(f"\nError converting timestamp: {e_ts}")

    # Example: Clean 'condition_text'
    if 'condition_text' in df.columns and df['condition_text'].dtype == 'object':
        df['condition_text'] = df['condition_text'].str.strip()
        print("\nStripped whitespace from 'condition_text'.")

    print("\nInfo after potential conversions:")
    df.info()

    # Further EDA and preprocessing would follow here...
    # For example, check unique values for 'TIPO', 'TARIFA', 'HORARIO'
    if 'TIPO' in df.columns: print(f"\nUnique TIPO values: {df['TIPO'].unique()}")
    if 'TARIFA' in df.columns: print(f"\nUnique TARIFA values: {df['TARIFA'].unique()}")
    if 'HORARIO' in df.columns: print(f"\nUnique HORARIO values: {df['HORARIO'].unique()}")
    if 'time_of_day_segment' in df.columns: print(f"\nUnique time_of_day_segment values: {df['time_of_day_segment'].unique()}")

if __name__ == "__main__":
    data = load_geodataframe(PARQUET_FILE_PATH)
    if data is not None:
        inspect_dataframe(data)
