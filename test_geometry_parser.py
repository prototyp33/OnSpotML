import numpy as np
import logging
from shapely.geometry import LineString, MultiLineString
from shapely.validation import make_valid

# Configure basic logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Revised parse_geometry function with active debugging ---
def parse_geometry(geom_dict: dict):
    logger.debug(f"Attempting to parse input: {str(geom_dict)[:200]}...") # Log only part of potentially large dict
    if not isinstance(geom_dict, dict):
        logger.debug("Input is not a dictionary.")
        return None
    try:
        geom_type = geom_dict.get('type')
        coords = geom_dict.get('coordinates')

        if not geom_type:
            logger.debug("Geometry 'type' is missing.")
            return None
        if coords is None:
            logger.debug("Geometry 'coordinates' is None.")
            return None

        if geom_type == 'LineString':
            logger.debug(f"Processing LineString with coords: {coords}")
            if isinstance(coords, np.ndarray):
                processed_coords = [tuple(point) for point in coords if isinstance(point, (np.ndarray, list, tuple)) and len(point) == 2]
            elif isinstance(coords, list):
                processed_coords = [tuple(point) for point in coords if isinstance(point, (list, tuple)) and len(point) == 2]
            else:
                logger.debug(f"LineString coords type not list or ndarray: {type(coords)}")
                return None
            
            if len(processed_coords) < 2:
                logger.debug(f"LineString requires at least 2 valid points, got {len(processed_coords)}")
                return None
            geom = LineString(processed_coords)
            logger.debug(f"Successfully created LineString: {geom.wkt[:50]}...")

        elif geom_type == 'MultiLineString':
            logger.debug(f"Processing MultiLineString with coords: {coords}")
            if not isinstance(coords, np.ndarray) or coords.ndim == 0 :
                logger.debug(f"MultiLineString coords not a non-empty numpy array: type={type(coords)}, ndim={getattr(coords, 'ndim', 'N/A')}")
                return None
            
            if coords.shape[0] == 0 : 
                 logger.debug("MultiLineString coords array is empty (shape[0] == 0). Cannot extract list of lines.")
                 return None

            # Expecting coords[0] to be the array of lines due to observed nesting
            list_of_lines_np_arrays = coords[0]
            logger.debug(f"Extracted list_of_lines_np_arrays (expected array of line arrays): {list_of_lines_np_arrays}")

            if not isinstance(list_of_lines_np_arrays, np.ndarray):
                logger.debug(f"MultiLineString inner list_of_lines is not a numpy array: {type(list_of_lines_np_arrays)}")
                return None
            
            if list_of_lines_np_arrays.ndim == 0 or list_of_lines_np_arrays.shape[0] == 0: # Check if it's empty or not at least 1D
                 logger.debug(f"MultiLineString list_of_lines_np_arrays is empty or not an array of lines. Shape: {list_of_lines_np_arrays.shape}")
                 return None

            valid_shapely_lines = []
            for i, line_np_array_candidate in enumerate(list_of_lines_np_arrays):
                logger.debug(f"  Processing line candidate {i} from MultiLineString: {line_np_array_candidate}")
                current_line_points = []
                if isinstance(line_np_array_candidate, np.ndarray):
                    current_line_points = [tuple(point) for point in line_np_array_candidate if isinstance(point, (np.ndarray, list, tuple)) and len(point) == 2]
                elif isinstance(line_np_array_candidate, list): 
                    current_line_points = [tuple(point) for point in line_np_array_candidate if isinstance(point, (list, tuple)) and len(point) == 2]
                else:
                    logger.debug(f"  Skipping malformed line candidate (not ndarray or list): {type(line_np_array_candidate)}")
                    continue
                
                if len(current_line_points) >= 2:
                    valid_shapely_lines.append(LineString(current_line_points))
                    logger.debug(f"    Added valid LineString for line candidate {i}")
                else:
                    logger.debug(f"    Skipping line candidate {i}, not enough valid points ({len(current_line_points)})")
            
            if not valid_shapely_lines:
                logger.debug("No valid LineString objects created for MultiLineString.")
                return None
            geom = MultiLineString(valid_shapely_lines)
            logger.debug(f"Successfully created MultiLineString with {len(valid_shapely_lines)} lines.")
        else:
            logger.warning(f"Unsupported geometry type: {geom_type}")
            return None
        
        logger.debug(f"Attempting make_valid on {geom.wkt[:50]}...")
        valid_geom = make_valid(geom)
        logger.debug(f"make_valid returned: {valid_geom.wkt[:50]}...")
        return valid_geom
    
    except FloatingPointError as e: 
        logger.error(f"FloatingPointError parsing: {e} for dict: {str(geom_dict)[:200]}...", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"UNHANDLED Exception during parsing: {e} for dict: {str(geom_dict)[:200]}...", exc_info=True)
        raise # Re-raise to see the full traceback in this test script

# --- Test Cases ---
problematic_multilinestring = {
    'coordinates': np.array([ # Outermost array (coords)
        np.array([ # list_of_lines_np_arrays (coords[0])
            np.array([ # line_np_array_candidate
                np.array([2.17906, 41.39383]), np.array([2.17892, 41.39383]) 
            ])
            # Potentially add another line here for more complex test:
            # ,np.array([ 
            #     np.array([2.17900, 41.39380]), np.array([2.17880, 41.39380])
            # ])
        ], dtype=object) # This dtype might be for list_of_lines_np_arrays
    ], dtype=object), # This dtype is for the outermost 'coordinates' array
    'type': 'MultiLineString'
}

problematic_linestring = {
    'coordinates': np.array([
        np.array([2.0, 41.0]),
        np.array([2.1, 41.1])
    ], dtype=object), # The coordinates array itself can be an object array of points
    'type': 'LineString'
}

# A LineString where individual points are Python lists instead of np.arrays
list_coords_linestring = {
    'coordinates': [ # Python list of points
        [2.0, 41.0],
        [2.1, 41.1]
    ],
    'type': 'LineString'
}

# MultiLineString where outer 'coordinates' is a list, and inner is also list of lists
list_coords_multilinestring = {
    'coordinates': [ # Python list (instead of np.array for coords)
        [ # First line (list of points)
            [2.17906, 41.39383], [2.17892, 41.39383] 
        ],
        [ # Second line
            [2.17900, 41.39380], [2.17880, 41.39380]
        ]
    ],
    'type': 'MultiLineString'
}


# This structure aims to mimic the observed problematic one more closely if coords[0] is just one line
single_line_in_multilinestring_problematic = {
    'coordinates': np.array([ # Outermost 'coords'
        np.array([ # This is coords[0], the list_of_lines_np_arrays
             # Each element here is a line, represented as an array of points
            np.array([np.array([2.1, 41.1]), np.array([2.2, 41.2])]) 
        ], dtype=object) 
    ], dtype=object),
    'type': 'MultiLineString'
}


empty_coords_multilinestring_valid_outer = {
    'type': 'MultiLineString', 
    'coordinates': np.array([np.array([], dtype=object)], dtype=object) # Outer array has one empty array
}

empty_coords_multilinestring_truly_empty = {
    'type': 'MultiLineString', 
    'coordinates': np.array([], dtype=object) # Outer array is completely empty
}


malformed_point_linestring = {
    'type': 'LineString', 
    'coordinates': np.array([np.array([2.0,41.0]), np.array([2.1])], dtype=object) 
}

too_few_points_linestring = {
    'type': 'LineString',
    'coordinates': np.array([np.array([2.0, 41.0])], dtype=object)
}

non_dict_input = "This is not a dict"

test_cases = {
    "problematic_multilinestring": problematic_multilinestring,
    "problematic_linestring": problematic_linestring,
    "list_coords_linestring": list_coords_linestring,
    # "list_coords_multilinestring": list_coords_multilinestring, # This will fail, MultiLineString expects coords to be np.array for current logic
    "single_line_in_multilinestring_problematic": single_line_in_multilinestring_problematic,
    "empty_coords_multilinestring_valid_outer": empty_coords_multilinestring_valid_outer,
    "empty_coords_multilinestring_truly_empty": empty_coords_multilinestring_truly_empty,
    "malformed_point_linestring": malformed_point_linestring,
    "too_few_points_linestring": too_few_points_linestring,
    "non_dict_input": non_dict_input
}

if __name__ == "__main__":
    for name, geom_dict_test in test_cases.items():
        logger.info(f"--- Testing Case: {name} ---")
        result = parse_geometry(geom_dict_test)
        if result:
            logger.info(f"Result for {name}: Parsed WKT: {result.wkt[:100]}..., Valid: {result.is_valid}, Empty: {result.is_empty}")
        else:
            logger.info(f"Result for {name}: Parsed to None")

    # Example of how it might look in the original DataFrame
    # Assuming your DataFrame df has the 'geometry' column with these dicts
    # For this test, we are calling parse_geometry directly.
    # If you had a df, you would do:
    # df['parsed_geometry'] = df['geometry'].apply(parse_geometry) 