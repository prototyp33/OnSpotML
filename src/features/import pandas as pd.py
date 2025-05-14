import pandas as pd
import geopandas as gpd

# Assuming your Parquet file is at this path
parquet_file_path = "/Users/adrianiraeguialvear/Desktop/OnSpotML_v2/data/processed/features_master_table.parquet"

# Load into a GeoDataFrame
try:
    gdf = gpd.read_parquet(parquet_file_path)
    print("Successfully loaded as GeoDataFrame.")
    print(f"CRS of loaded GDF: {gdf.crs}")
    print(f"Geometry column type: {gdf.geometry.geom_type.unique()}") 
except Exception as e_gpd:
    print(f"Could not load directly as GeoDataFrame: {e_gpd}")
    print("Attempting to load as Pandas DataFrame first...")
    try:
        df = pd.read_parquet(parquet_file_path)
        print("Successfully loaded as Pandas DataFrame.")
        # If 'geometry' is WKB, you might need to convert it
        if 'geometry' in df.columns:
            try:
                # Attempt to convert if it's WKB strings or bytes
                from shapely import wkb
                # Check if it's already shapely geometries (less likely if not loaded by gpd.read_parquet)
                if not hasattr(df['geometry'].iloc[0], 'geom_type'):
                     # If the geometry column from parquet is bytes (common for WKB)
                    if isinstance(df['geometry'].iloc[0], bytes):
                        df['geometry'] = df['geometry'].apply(wkb.loads)
                    # If it's a dict like in your JSON sample, that's unusual for Parquet.
                    # Parquet usually stores WKB as a byte array.
                    # The dict representation you showed is more typical of a JSON serialization of bytes.
                    # If it IS a dict of byte values in the DataFrame, it needs custom handling:
                    elif isinstance(df['geometry'].iloc[0], dict):
                        print("Geometry is a dict of byte values, requires custom deserialization.")
                        # Example for the dict structure you showed:
                        # def deserialize_geometry_from_dict(geom_dict):
                        #     if geom_dict is None: return None
                        #     byte_string = b''.join(bytes([v]) for k, v in sorted(geom_dict.items(), key=lambda item: int(item[0])))
                        #     return wkb.loads(byte_string)
                        # df['geometry'] = df['geometry'].apply(deserialize_geometry_from_dict)
                    else:
                         print(f"Geometry column is of unexpected type: {type(df['geometry'].iloc[0])}")

                gdf = gpd.GeoDataFrame(df, geometry='geometry')
                # Try to set CRS if known, e.g., from your logs it was EPSG:4326 before feature engineering
                # The CRS might be stored in the Parquet metadata if saved by GeoPandas
                if gdf.crs is None:
                    print("CRS not found, attempting to set to EPSG:4326 (WGS 84)")
                    gdf.set_crs("EPSG:4326", inplace=True)
                print(f"Converted to GeoDataFrame. CRS: {gdf.crs}")
            except Exception as e_conv:
                print(f"Error converting 'geometry' column or creating GeoDataFrame: {e_conv}")
                gdf = df # Fallback to DataFrame if GDF conversion fails
        else:
            gdf = df # No geometry column
    except Exception as e_pd:
        print(f"Error loading Parquet file with pandas: {e_pd}")
        gdf = None

if gdf is not None:
    print("\nFirst 5 rows of the loaded data:")
    print(gdf.head())

    print("\nData types (dtypes):")
    print(gdf.dtypes)

    # Convert timestamp if it's not already datetime
    if 'timestamp' in gdf.columns and gdf['timestamp'].dtype == 'int64':
        try:
            gdf['timestamp'] = pd.to_datetime(gdf['timestamp'], unit='ms')
            print("\nConverted 'timestamp' to datetime.")
        except Exception as e_ts:
            print(f"\nError converting timestamp: {e_ts}")

    # Example: Clean 'condition_text'
    if 'condition_text' in gdf.columns and gdf['condition_text'].dtype == 'object':
        gdf['condition_text'] = gdf['condition_text'].str.strip()
        print("\nStripped whitespace from 'condition_text'.")

    print("\nInfo after potential conversions:")
    gdf.info()

    # Further EDA and preprocessing would follow here...
    # For example, check unique values for 'TIPO', 'TARIFA', 'HORARIO'
    if 'TIPO' in gdf.columns: print(f"\nUnique TIPO values: {gdf['TIPO'].unique()}")
    if 'TARIFA' in gdf.columns: print(f"\nUnique TARIFA values: {gdf['TARIFA'].unique()}")
    if 'HORARIO' in gdf.columns: print(f"\nUnique HORARIO values: {gdf['HORARIO'].unique()}")
    if 'time_of_day_segment' in gdf.columns: print(f"\nUnique time_of_day_segment values: {gdf['time_of_day_segment'].unique()}")

