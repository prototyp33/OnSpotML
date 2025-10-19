# OnSpotML Data Integration Audit Report

## Executive Summary

🔴 **CRITICAL ISSUES FOUND** - The multi-source data integration pipeline has several significant issues that could lead to data quality problems, temporal misalignment, and model degradation.

## Critical Issues Identified

### 1. **SEVERE: Inconsistent Merge Key Validation** 
**Location**: `src/data_ingestion/barcelona_data_collector.py:618`
```python
# Merge parking segments with tariffs
if 'ID_TARIFA' not in parking_cleaned.columns or 'ID_TARIFA' not in tariffs_cleaned.columns:
    logger.error("Integration failed: Missing 'ID_TARIFA' merge key in one or both dataframes.")
    return None
```

**Problem**: 
- Only validates presence of merge keys but not their data quality
- No validation of key uniqueness or data types
- No handling of duplicate keys which could cause row inflation

**Impact**: HIGH - Silent data corruption during merging

### 2. **SEVERE: Timezone Handling Inconsistencies**
**Location**: Multiple files (`src/features/build_features.py`, `src/modeling/feature_engineering_v2.py`)

**Problems**:
- Inconsistent timezone localization across different data sources
- `ambiguous='infer'` could lead to incorrect timestamp assignments during DST transitions
- No validation that all data sources are aligned to same timezone before integration

**Code Examples**:
```python
# In weather features - Line 172
weather_df['time'] = weather_df['time'].dt.tz_localize('Europe/Madrid', ambiguous='infer', nonexistent='NaT')

# In event features - Line 265
ts_series_for_comparison = ts_series_for_comparison.dt.tz_localize('Europe/Madrid', ambiguous='infer', nonexistent='NaT')
```

**Impact**: HIGH - Temporal misalignment between data sources

### 3. **CRITICAL: Unsafe Index Management in Feature Building**
**Location**: `src/features/build_features.py:540-694`

**Problems**:
- Complex index manipulation without proper validation
- Risk of misaligned features when joining multiple DataFrames
- Fallback to creating `temp_join_id` from non-unique indices

**Code Example**:
```python
# Lines 582-588
if not main_id_column:
    logger.warning("No unique ID column found in base parking data. Creating 'temp_join_id' from index.")
    if parking_gdf.index.is_unique:
        parking_gdf = parking_gdf.reset_index().rename(columns={'index': 'temp_join_id'})
        main_id_column = 'temp_join_id'
    else:
        logger.error("Index is not unique, cannot create reliable 'temp_join_id'. Aborting.")
        return None
```

**Impact**: CRITICAL - Feature misalignment leading to data leakage

### 4. **HIGH: Missing Data Quality Validation**
**Location**: `src/data_ingestion/barcelona_data_collector.py:127`

**Problems**:
- Minimal CSV validation (only reads 5 rows)
- No schema validation against expected data types
- No range/sanity checks for numerical values
- No duplicate detection during ingestion

**Impact**: HIGH - Propagation of data quality issues throughout pipeline

### 5. **HIGH: Unsafe Spatial Operations**
**Location**: `src/features/build_features.py:308-388`

**Problems**:
- No CRS validation before spatial operations
- Assumptions about coordinate systems without verification
- Mixed handling of Point vs LineString geometries

**Code Example**:
```python
# Line 495-500
geometry = [Point(xy) for xy in zip(df[coord_cols[0]], df[coord_cols[1]])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326") # Assume WGS84 for raw coords
```

**Impact**: HIGH - Incorrect spatial relationships and distance calculations

### 6. **MEDIUM: Error Handling Deficiencies**
**Location**: Multiple files

**Problems**:
- Silent failures in data integration (returns None without detailed error context)
- Inconsistent error handling strategies
- Missing rollback mechanisms when integration fails

**Impact**: MEDIUM - Difficult debugging and unreliable pipeline execution

### 7. **MEDIUM: Insufficient Data Validation Coverage**
**Location**: `src/data_validation/` directory

**Problems**:
- Only one validation script (`check_tram_coverage.py`)
- No cross-source data consistency checks
- No automated validation in the integration pipeline

**Impact**: MEDIUM - Undetected data quality issues

## Detailed Technical Issues

### Weather Data Integration
```python:146:234:src/features/build_features.py
# Issue: merge_asof without proper validation of timestamp alignment
merged_features = pd.merge_asof(
    target_df_sorted,
    weather_df_sorted,
    left_on=timestamp_col,
    right_on='weather_timestamp',
    direction='nearest',  # Could match very distant timestamps
)
```

### Event Data Integration
```python:234:345:src/features/build_features.py
# Issue: Complex timezone logic without validation
# Issue: No validation of event date ranges (start > end dates)
# Issue: No handling of overlapping events
```

### Spatial Features Integration
```python:308:388:src/features/build_features.py
# Issue: DBSCAN clustering without parameter validation
# Issue: No handling of edge cases (empty clusters, all noise points)
```

## Recommended Fixes

### 1. **Implement Comprehensive Merge Key Validation**

Create `src/data_validation/merge_key_validator.py`:
```python
def validate_merge_keys(left_df, right_df, key_columns, operation="merge"):
    """Validate merge keys before joining operations."""
    issues = []
    
    for key in key_columns:
        # Check existence
        if key not in left_df.columns:
            issues.append(f"Key '{key}' missing in left DataFrame")
        if key not in right_df.columns:
            issues.append(f"Key '{key}' missing in right DataFrame")
            continue
        
        # Check data types
        if left_df[key].dtype != right_df[key].dtype:
            issues.append(f"Key '{key}' has mismatched types: {left_df[key].dtype} vs {right_df[key].dtype}")
        
        # Check for nulls
        left_nulls = left_df[key].isnull().sum()
        right_nulls = right_df[key].isnull().sum()
        if left_nulls > 0 or right_nulls > 0:
            issues.append(f"Key '{key}' has null values: left={left_nulls}, right={right_nulls}")
        
        # Check for duplicates
        left_dups = left_df[key].duplicated().sum()
        right_dups = right_df[key].duplicated().sum()
        if operation == "merge" and (left_dups > 0 or right_dups > 0):
            issues.append(f"Key '{key}' has duplicates: left={left_dups}, right={right_dups}")
    
    return issues
```

### 2. **Standardize Timezone Handling**

Create `src/utils/timezone_manager.py`:
```python
def standardize_timezone(series, target_tz="Europe/Madrid", validate=True):
    """Standardize timestamp series to target timezone with validation."""
    if validate:
        # Check for invalid timestamps
        invalid_count = series.isnull().sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid timestamps")
    
    if series.dt.tz is None:
        # Handle DST transitions more carefully
        return series.dt.tz_localize(target_tz, ambiguous='NaT', nonexistent='NaT')
    else:
        return series.dt.tz_convert(target_tz)
```

### 3. **Implement Safe Index Management**

```python
def safe_feature_join(base_df, feature_df, join_key, feature_name):
    """Safely join features with comprehensive validation."""
    # Validate indices alignment
    if not base_df.index.equals(feature_df.index):
        logger.error(f"Index mismatch for {feature_name} features")
        return None
    
    # Check for duplicate columns
    overlapping_cols = set(base_df.columns) & set(feature_df.columns)
    if overlapping_cols:
        logger.warning(f"Overlapping columns in {feature_name}: {overlapping_cols}")
    
    return base_df.join(feature_df, how='left', rsuffix=f'_{feature_name}')
```

### 4. **Add Comprehensive Data Validation**

Create `src/data_validation/integration_validator.py`:
```python
def validate_integrated_data(df, config):
    """Comprehensive validation of integrated dataset."""
    issues = []
    
    # Check required columns
    missing_cols = set(config['required_columns']) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check data types
    for col, expected_type in config['column_types'].items():
        if col in df.columns and not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
            issues.append(f"Column '{col}' has wrong type: {df[col].dtype} vs {expected_type}")
    
    # Check value ranges
    for col, (min_val, max_val) in config.get('value_ranges', {}).items():
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                issues.append(f"Column '{col}' has {out_of_range} values out of range [{min_val}, {max_val}]")
    
    return issues
```

### 5. **Implement Spatial Validation**

```python
def validate_spatial_data(gdf, expected_crs="EPSG:4326"):
    """Validate spatial data consistency."""
    issues = []
    
    if gdf.crs is None:
        issues.append("CRS is not set")
    elif str(gdf.crs) != expected_crs:
        issues.append(f"CRS mismatch: {gdf.crs} vs {expected_crs}")
    
    # Check for invalid geometries
    invalid_geoms = (~gdf.geometry.is_valid).sum()
    if invalid_geoms > 0:
        issues.append(f"{invalid_geoms} invalid geometries found")
    
    # Check for empty geometries
    empty_geoms = gdf.geometry.is_empty.sum()
    if empty_geoms > 0:
        issues.append(f"{empty_geoms} empty geometries found")
    
    return issues
```

## Implementation Priority

### High Priority (Immediate)
1. **Timezone standardization** - Implement consistent timezone handling
2. **Merge key validation** - Add validation before all join operations
3. **Index management** - Fix unsafe index operations in feature building

### Medium Priority (Next Sprint)
1. **Data validation framework** - Implement comprehensive validation
2. **Error handling** - Improve error handling and logging
3. **Spatial validation** - Add CRS and geometry validation

### Low Priority (Future)
1. **Performance optimization** - Optimize large data merges
2. **Monitoring** - Add data drift detection
3. **Documentation** - Update integration documentation

## Testing Recommendations

1. **Unit Tests**: Create tests for each integration function
2. **Integration Tests**: Test end-to-end data flow
3. **Data Quality Tests**: Automated validation in CI/CD
4. **Edge Case Tests**: Handle missing data, timezone edge cases, spatial edge cases

## Monitoring Recommendations

1. **Data Quality Metrics**: Track validation failures over time
2. **Integration Metrics**: Monitor merge success rates and row counts
3. **Temporal Consistency**: Alert on timestamp gaps or misalignments
4. **Spatial Consistency**: Monitor for CRS issues and invalid geometries

## Conclusion

The current multi-source data integration has several critical issues that require immediate attention. The highest risk issues are timezone inconsistencies and unsafe index management, which could lead to data leakage and model degradation. Implementing the recommended fixes will significantly improve data quality and pipeline reliability. 