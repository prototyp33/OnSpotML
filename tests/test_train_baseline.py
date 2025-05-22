import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yaml
import lightgbm as lgb
from src.modeling.train_baseline import (
    load_and_prepare_data,
    handle_duplicate_timestamps,
    validate_data_frequency,
    calculate_class_weights,
    calculate_metrics,
    train_model,
    analyze_features,
    calculate_vif,
    create_time_series_split,
    N_SPLITS,
    TEST_SIZE_DAYS,
    GAP_DAYS
)
from src.modeling.feature_engineering_v2 import FeatureEngineeringPipeline
from sklearn.model_selection import TimeSeriesSplit

# Test data fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='5min')
    parking_ids = ['P1', 'P2', 'P3']
    
    data = []
    for parking_id in parking_ids:
        for date in dates:
            data.append({
                'timestamp': date,
                'parking_id': parking_id,
                'available_spaces': np.random.randint(0, 100),
                'total_spaces': 100,
                'hour_sin': np.sin(date.hour * 2 * np.pi / 24),
                'hour_cos': np.cos(date.hour * 2 * np.pi / 24),
                'is_weekend': int(date.weekday() >= 5),
                'temp_c': np.random.normal(20, 5),
                'humidity': np.random.normal(60, 10),
                'wind_speed': np.random.normal(10, 3),
                'precipitation': np.random.normal(0, 2),
                'is_rainy': np.random.choice([0, 1], p=[0.8, 0.2]),
                'is_snowy': np.random.choice([0, 1], p=[0.9, 0.1]),
                'is_cloudy': np.random.choice([0, 1], p=[0.7, 0.3]),
                'is_clear': np.random.choice([0, 1], p=[0.6, 0.4]),
                'is_event_day': np.random.choice([0, 1], p=[0.9, 0.1]),
                'event_type': np.random.choice(['none', 'sports', 'concert', 'festival'], p=[0.9, 0.03, 0.03, 0.04]),
                'event_importance': np.random.choice([0, 1, 2, 3], p=[0.9, 0.03, 0.03, 0.04]),
                'public_transport_density': np.random.normal(50, 10),
                'public_transport_distance': np.random.normal(200, 50),
                'bus_stop_count': np.random.randint(0, 5),
                'metro_station_count': np.random.randint(0, 3),
                'total_spaces': 100,
                'parking_type': np.random.choice(['underground', 'surface', 'multi_level'], p=[0.3, 0.4, 0.3]),
                'pricing_tier': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.4, 0.3]),
                'is_24h': np.random.choice([0, 1], p=[0.7, 0.3]),
                'has_security': np.random.choice([0, 1], p=[0.6, 0.4]),
                'has_ev_charging': np.random.choice([0, 1], p=[0.8, 0.2]),
                'is_underground': np.random.choice([0, 1], p=[0.7, 0.3]),
                'is_public': np.random.choice([0, 1], p=[0.6, 0.4]),
                'is_open_now': np.random.choice([0, 1], p=[0.2, 0.8]),
                'hours_until_close': np.random.normal(8, 4),
                'hours_since_open': np.random.normal(8, 4)
            })
    
    df = pd.DataFrame(data)
    df['occupancy_class'] = (df['available_spaces'] / df['total_spaces']).apply(
        lambda x: 0 if x < 0.2 else (1 if x < 0.4 else (2 if x < 0.6 else (3 if x < 0.8 else 4)))
    )
    return df

@pytest.fixture
def config():
    """Create test configuration."""
    return {
        'feature_engineering': {
            'temporal_features': {'enabled': True},
            'lag_features': {'enabled': True},
            'poi_features': {'enabled': True},
            'facility_features': {'enabled': True}
        },
        'model': {
            'class_weight': 'balanced',
            'manual_weights': [1.0, 1.2, 1.0, 0.8, 0.6]
        }
    }

@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file for testing."""
    config = {
        'feature_engineering': {
            'temporal_features': True,
            'weather_features': True,
            'event_features': True,
            'transport_features': True,
            'poi_features': False
        },
        'model': {
            'class_weight': 'balanced',
            'manual_weights': [1.0, 1.2, 1.0, 0.8, 0.6]
        }
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)

def test_handle_duplicate_timestamps():
    """Test handling of duplicate timestamps."""
    print("\n=== Debug: test_handle_duplicate_timestamps ===")
    
    # Create test data with duplicates
    data = {
        'timestamp': pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:00:00', '2024-01-01 00:05:00']),
        'parking_id': ['P1', 'P1', 'P1'],
        'available_spaces': [50, 60, 70],
        'total_spaces': [100, 100, 100]
    }
    df = pd.DataFrame(data)
    print(f"Original DataFrame:\n{df}")
    print(f"Original shape: {df.shape}")
    
    # Handle duplicates
    df_cleaned = handle_duplicate_timestamps(df)
    print(f"\nCleaned DataFrame:\n{df_cleaned}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    
    # Verify results
    assert len(df_cleaned) == 2  # One duplicate removed
    print(f"\nVerifying results:")
    print(f"Expected length: 2, Actual length: {len(df_cleaned)}")
    print(f"First row available_spaces: {df_cleaned['available_spaces'].iloc[0]}")
    print(f"Second row available_spaces: {df_cleaned['available_spaces'].iloc[1]}")
    assert df_cleaned['available_spaces'].iloc[0] == 60  # Last instance kept
    assert df_cleaned['available_spaces'].iloc[1] == 70  # Non-duplicate kept

def test_validate_data_frequency():
    """Test data frequency validation."""
    print("\n=== Debug: test_validate_data_frequency ===")
    
    # Create test data with regular frequency
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='5min')
    data = {
        'timestamp': dates,
        'parking_id': ['P1'] * len(dates),
        'available_spaces': np.random.randint(0, 100, len(dates)),
        'total_spaces': [100] * len(dates)
    }
    df = pd.DataFrame(data)
    print(f"Test DataFrame shape: {df.shape}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Sample of timestamps:\n{df['timestamp'].head()}")
    
    # Validate frequency
    freq_df = validate_data_frequency(df)
    print(f"\nFrequency DataFrame:\n{freq_df}")
    print(f"Frequency DataFrame columns: {freq_df.columns.tolist()}")
    
    # Verify results
    assert isinstance(freq_df, pd.DataFrame)
    assert 'mean_diff' in freq_df.columns
    assert 'std_diff' in freq_df.columns
    assert 'deviation_rate' in freq_df.columns
    
    # Check frequency statistics
    print("\nChecking frequency statistics:")
    mean_diff = freq_df['mean_diff'].mean()
    std_diff = freq_df['std_diff'].mean()
    deviation_rate = freq_df['deviation_rate'].mean()
    print(f"Mean time difference: {mean_diff:.2f} minutes")
    print(f"Standard deviation: {std_diff:.2f} minutes")
    print(f"Deviation rate: {deviation_rate:.2%}")
    
    assert mean_diff == pytest.approx(5.0, rel=0.1)  # 5-minute frequency
    assert std_diff < 1.0  # Low standard deviation
    assert deviation_rate < 0.1  # Low deviation rate

def test_calculate_class_weights():
    """Test class weight calculation strategies."""
    print("\n=== Debug: test_calculate_class_weights ===")
    
    # Create test data with imbalanced classes
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 3])  # 4:2:3:1 ratio
    print(f"Test data shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Test different strategies
    strategies = ['none', 'balanced', 'custom', 'effective_samples', 'inverse_log', 'cost_sensitive']
    config = {'manual_weights': [1.0, 1.2, 1.0, 0.8]}
    print(f"\nTesting strategies: {strategies}")
    print(f"Config: {config}")
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        weights = calculate_class_weights(y, strategy, config)
        print(f"Calculated weights: {weights}")
        
        if strategy == 'none':
            assert weights is None
            print("Strategy 'none' returned None as expected")
        else:
            assert isinstance(weights, list)
            assert len(weights) == len(np.unique(y))
            assert all(w > 0 for w in weights)
            print(f"Number of weights: {len(weights)}")
            print(f"All weights positive: {all(w > 0 for w in weights)}")
            
            if strategy == 'balanced':
                # Check if weights are inversely proportional to class frequencies
                class_counts = np.bincount(y)
                expected_weights = len(y) / (len(np.unique(y)) * class_counts)
                print(f"Expected weights: {expected_weights}")
                print(f"Actual weights: {weights}")
                np.testing.assert_array_almost_equal(weights, expected_weights)
            
            elif strategy == 'cost_sensitive':
                # Check if manual weights are used
                print(f"Expected manual weights: {config['manual_weights']}")
                print(f"Actual weights: {weights}")
                np.testing.assert_array_almost_equal(weights, config['manual_weights'])

def test_calculate_metrics():
    """Test metric calculation."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])  # One error
    y_pred_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.2, 0.7, 0.1]
    ])
    
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Check if all expected metrics are present
    assert 'accuracy' in metrics
    assert 'f1_weighted' in metrics
    assert 'f1_macro' in metrics
    assert 'balanced_accuracy' in metrics
    
    # Check per-class metrics
    for cls in range(3):
        assert f'class_{cls}_precision' in metrics
        assert f'class_{cls}_recall' in metrics
        assert f'class_{cls}_f1' in metrics
        assert f'class_{cls}_auc' in metrics
        assert f'class_{cls}_ap' in metrics
        assert f'class_{cls}_brier' in metrics

def test_train_model(sample_data):
    """Test model training."""
    print("\n=== Debug: test_train_model ===")
    
    # Convert categorical columns to numeric
    categorical_cols = ['event_type', 'parking_type', 'pricing_tier']
    for col in categorical_cols:
        if col in sample_data.columns:
            # Convert to categorical codes
            sample_data[col] = pd.Categorical(sample_data[col]).codes
    
    print(f"Sample data columns: {sample_data.columns.tolist()}")
    print(f"Sample data dtypes:\n{sample_data.dtypes}")
    
    # Drop non-feature columns and ensure all remaining columns are numeric
    X = sample_data.drop(['occupancy_class', 'timestamp', 'parking_id'], axis=1)
    y = sample_data['occupancy_class']
    
    # Verify all columns are numeric
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]), f"Column {col} is not numeric: {X[col].dtype}"
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtypes:\n{X.dtypes}")
    
    model, params = train_model(X, y)
    
    # Verify model properties
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    assert isinstance(params, dict)
    
    # Test predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_prob.shape}")
    
    assert len(y_pred) == len(X)
    assert y_prob.shape[1] == len(np.unique(y))
    assert all(p >= 0 and p <= 1 for p in y_prob.flatten())

def test_analyze_features(sample_data, tmp_path):
    """Test feature analysis."""
    print("\n=== Debug: test_analyze_features ===")
    
    # Convert categorical columns to numeric
    categorical_cols = ['event_type', 'parking_type', 'pricing_tier']
    for col in categorical_cols:
        if col in sample_data.columns:
            sample_data[col] = pd.Categorical(sample_data[col]).codes
    
    print(f"Sample data columns: {sample_data.columns.tolist()}")
    print(f"Sample data dtypes:\n{sample_data.dtypes}")
    
    X = sample_data.drop(['occupancy_class', 'timestamp', 'parking_id'], axis=1)
    y = sample_data['occupancy_class']
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtypes:\n{X.dtypes}")
    
    feature_stats = analyze_features(X, y, {}, str(tmp_path))
    
    # Verify feature statistics
    assert isinstance(feature_stats, dict)
    assert 'permutation_importance' in feature_stats
    assert 'correlations' in feature_stats
    assert 'collinearity' in feature_stats
    
    # Check if plots are created
    plot_files = list(tmp_path.glob('*.png'))
    assert len(plot_files) > 0
    
    # Verify correlation structure
    assert 'feature_correlations' in feature_stats['correlations']
    assert 'target_correlations' in feature_stats['correlations']
    
    # Verify collinearity scores
    for col in X.columns:
        if col in feature_stats['collinearity']:
            vif = feature_stats['collinearity'][col]
            assert isinstance(vif, float)
            assert vif >= 1.0 or np.isnan(vif)  # VIF should be at least 1.0 or NaN
    
    # Verify permutation importance
    perm_importance = feature_stats['permutation_importance']
    assert len(perm_importance) == len(X.columns)
    assert all(isinstance(imp, float) for imp in perm_importance.values())
    assert all(imp >= 0 for imp in perm_importance.values())

def test_calculate_vif(sample_data):
    """Test VIF calculation."""
    print("\n=== Debug: test_calculate_vif ===")
    
    # Convert categorical columns to numeric
    categorical_cols = ['event_type', 'parking_type', 'pricing_tier']
    for col in categorical_cols:
        if col in sample_data.columns:
            sample_data[col] = pd.Categorical(sample_data[col]).codes
    
    print(f"Sample data columns: {sample_data.columns.tolist()}")
    print(f"Sample data dtypes:\n{sample_data.dtypes}")
    
    X = sample_data.drop(['occupancy_class', 'timestamp', 'parking_id'], axis=1)
    
    print(f"X shape: {X.shape}")
    print(f"X dtypes:\n{X.dtypes}")
    
    # Test VIF for a few features
    for col in ['temp_c', 'humidity', 'wind_speed']:
        if col in X.columns:
            vif = calculate_vif(X, col)
            print(f"VIF for {col}: {vif}")
            assert isinstance(vif, float)
            assert vif >= 1.0  # VIF should be at least 1.0

def test_create_time_series_split(sample_data):
    """Test time series split creation."""
    print("\n=== Debug: test_create_time_series_split ===")
    
    # Create a larger dataset for testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='5min')
    data = {
        'timestamp': dates,
        'parking_id': ['P1'] * len(dates),
        'available_spaces': np.random.randint(0, 100, len(dates)),
        'total_spaces': [100] * len(dates)
    }
    df = pd.DataFrame(data)
    
    print(f"Test data shape: {df.shape}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Temporarily modify constants for testing
    original_n_splits = N_SPLITS
    original_test_size = TEST_SIZE_DAYS
    original_gap = GAP_DAYS
    
    try:
        # Set smaller values for testing
        import src.modeling.train_baseline as train_baseline
        train_baseline.N_SPLITS = 2
        train_baseline.TEST_SIZE_DAYS = 1
        train_baseline.GAP_DAYS = 1
        
        tscv = create_time_series_split(df)
        
        # Verify split properties
        assert hasattr(tscv, 'split')
        assert hasattr(tscv, 'n_splits')
        assert tscv.n_splits == train_baseline.N_SPLITS
        
        # Test a few splits
        splits = list(tscv.split(df))
        assert len(splits) == train_baseline.N_SPLITS
        
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"\nSplit {i+1}:")
            print(f"Train size: {len(train_idx)}")
            print(f"Test size: {len(test_idx)}")
            
            # Verify no overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Verify chronological order
            train_times = df.iloc[train_idx]['timestamp']
            test_times = df.iloc[test_idx]['timestamp']
            assert train_times.max() < test_times.min()
            
    finally:
        # Restore original constants
        train_baseline.N_SPLITS = original_n_splits
        train_baseline.TEST_SIZE_DAYS = original_test_size
        train_baseline.GAP_DAYS = original_gap

def test_load_and_prepare_data(sample_data, tmp_path, config_file):
    """Test data loading and preparation."""
    print("\n=== Debug: test_load_and_prepare_data ===")
    
    # Create test data directory
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    
    # Save sample data to parquet
    data_path = data_dir / "features_master_table_historical_FULL.parquet"
    sample_data.to_parquet(data_path)
    
    print(f"Saved test data to: {data_path}")
    print(f"Test data shape: {sample_data.shape}")
    
    # Temporarily modify DATA_PATH
    import src.modeling.train_baseline as train_baseline
    original_data_path = train_baseline.DATA_PATH
    train_baseline.DATA_PATH = str(data_path)
    
    try:
        # Mock the config loading
        original_load_config = train_baseline.load_config
        def mock_load_config(config_path=None):
            return {
                'feature_engineering': {
                    'temporal_features': True,
                    'weather_features': True,
                    'event_features': True,
                    'transport_features': True,
                    'poi_features': False
                }
            }
        train_baseline.load_config = mock_load_config
        
        # Mock the feature engineering pipeline
        original_pipeline = train_baseline.FeatureEngineeringPipeline
        class MockPipeline:
            def __init__(self, config):
                self.config = config
            
            def create_features(self, df):
                # Add some basic features to simulate the pipeline
                df['hour_sin'] = np.sin(df['timestamp'].dt.hour * 2 * np.pi / 24)
                df['hour_cos'] = np.cos(df['timestamp'].dt.hour * 2 * np.pi / 24)
                df['is_weekend'] = df['timestamp'].dt.weekday >= 5
                return df
        
        train_baseline.FeatureEngineeringPipeline = MockPipeline
        
        # Mock the target variable preparation
        original_prepare_target = train_baseline.prepare_target_variable
        def mock_prepare_target(df):
            df['occupancy_class'] = (df['available_spaces'] / df['total_spaces']).apply(
                lambda x: 0 if x < 0.2 else (1 if x < 0.4 else (2 if x < 0.6 else (3 if x < 0.8 else 4)))
            )
            return df, {'class_distribution': df['occupancy_class'].value_counts().to_dict()}
        
        train_baseline.prepare_target_variable = mock_prepare_target
        
        df = load_and_prepare_data()
        
        # Verify loaded data
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'timestamp' in df.columns
        assert 'parking_id' in df.columns
        assert 'available_spaces' in df.columns
        assert 'total_spaces' in df.columns
        assert 'occupancy_class' in df.columns
        assert 'hour_sin' in df.columns
        assert 'hour_cos' in df.columns
        assert 'is_weekend' in df.columns
        
        print(f"Loaded data shape: {df.shape}")
        print(f"Loaded data columns: {df.columns.tolist()}")
        
    finally:
        # Restore original functions and constants
        train_baseline.DATA_PATH = original_data_path
        train_baseline.load_config = original_load_config
        train_baseline.FeatureEngineeringPipeline = original_pipeline
        train_baseline.prepare_target_variable = original_prepare_target

def test_edge_cases_extended(config):
    """Test additional edge cases in the training pipeline."""
    print("\n=== Debug: test_edge_cases_extended ===")
    
    # Create test data with single class
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='5min')
    n_samples = len(dates)
    
    print(f"Number of samples: {n_samples}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Create DataFrame with consistent array lengths
    single_class_data = pd.DataFrame({
        'timestamp': dates,
        'parking_id': ['P1'] * n_samples,
        'available_spaces': [50] * n_samples,
        'total_spaces': [100] * n_samples,
        'latitude': [40.0] * n_samples,
        'longitude': [-74.0] * n_samples,
        'opening_hour': [0] * n_samples,
        'closing_hour': [24] * n_samples
    })
    
    print(f"Single class data shape: {single_class_data.shape}")
    print(f"Column lengths:")
    for col in single_class_data.columns:
        print(f"  {col}: {len(single_class_data[col])}")
    
    # Test with single class data
    try:
        X = single_class_data.drop(['timestamp', 'parking_id'], axis=1)
        y = np.zeros(len(X))  # All samples in class 0
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        model, params = train_model(X, y)
        print("Model trained successfully with single class data")
    except Exception as e:
        print(f"Error training model with single class data: {str(e)}")
        raise
    
    # Test with constant features
    constant_data = single_class_data.copy()
    constant_data['constant_feature'] = 1.0
    try:
        X = constant_data.drop(['timestamp', 'parking_id'], axis=1)
        y = np.zeros(len(X))
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        model, params = train_model(X, y)
        print("Model trained successfully with constant features")
    except Exception as e:
        print(f"Error training model with constant features: {str(e)}")
        raise

def test_data_quality_edge_cases(config):
    """Test edge cases related to data quality."""
    print("\n=== Debug: test_data_quality_edge_cases ===")
    
    # Create test data with high missing rate
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='5min')
    n_samples = len(dates)
    
    print(f"Number of samples: {n_samples}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Create DataFrame with consistent array lengths
    high_missing_data = pd.DataFrame({
        'timestamp': dates,
        'parking_id': ['P1'] * n_samples,
        'available_spaces': [np.nan] * n_samples,  # All missing
        'total_spaces': [100] * n_samples,
        'latitude': [40.0] * n_samples,
        'longitude': [-74.0] * n_samples,
        'opening_hour': [0] * n_samples,
        'closing_hour': [24] * n_samples
    })
    
    print(f"High missing data shape: {high_missing_data.shape}")
    print(f"Column lengths:")
    for col in high_missing_data.columns:
        print(f"  {col}: {len(high_missing_data[col])}")
    print(f"Missing values:\n{high_missing_data.isnull().sum()}")
    
    # Test with high missing rate
    try:
        X = high_missing_data.drop(['timestamp', 'parking_id'], axis=1)
        y = np.zeros(len(X))
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        model, params = train_model(X, y)
        print("Model trained successfully with high missing rate")
    except Exception as e:
        print(f"Error training model with high missing rate: {str(e)}")
        raise
    
    # Test with extreme values
    extreme_data = high_missing_data.copy()
    extreme_data['available_spaces'] = [1e6] * n_samples  # Very large values
    try:
        X = extreme_data.drop(['timestamp', 'parking_id'], axis=1)
        y = np.zeros(len(X))
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        model, params = train_model(X, y)
        print("Model trained successfully with extreme values")
    except Exception as e:
        print(f"Error training model with extreme values: {str(e)}")
        raise

def test_time_series_split(config):
    """Test time series cross-validation."""
    print("\n=== Debug: test_time_series_split ===")
    
    # Create test data with more samples
    n_samples = 1000  # Increased number of samples
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='5min')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'parking_id': ['P1'] * n_samples,
        'available_spaces': np.random.randint(0, 100, n_samples),
        'total_spaces': [100] * n_samples,
        'latitude': [40.0] * n_samples,
        'longitude': [-74.0] * n_samples,
        'opening_hour': [0] * n_samples,
        'closing_hour': [24] * n_samples
    })
    
    print(f"Test data shape: {test_data.shape}")
    
    # Test time series split
    try:
        X = test_data.drop(['timestamp', 'parking_id'], axis=1)
        y = np.random.randint(0, 3, len(X))  # 3 classes
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        
        print(f"Number of splits: {len(splits)}")
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"Split {i+1}:")
            print(f"  Train size: {len(train_idx)}")
            print(f"  Test size: {len(test_idx)}")
            print(f"  Train range: {test_data.iloc[train_idx]['timestamp'].min()} to {test_data.iloc[train_idx]['timestamp'].max()}")
            print(f"  Test range: {test_data.iloc[test_idx]['timestamp'].min()} to {test_data.iloc[test_idx]['timestamp'].max()}")
        
        # Verify splits
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap
            assert max(train_idx) < min(test_idx)  # Time order preserved
        
        print("Time series split test passed")
    except Exception as e:
        print(f"Error in time series split test: {str(e)}")
        raise 