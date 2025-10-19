import pytest
import sys
import os
sys.path.append('src')

# Only import what actually exists
from src.modeling import train_baseline
from src.modeling.train_baseline import (
    load_config,
    load_and_prepare_data,
    validate_data,
    create_time_series_split,
    train_and_evaluate_model,
    generate_evaluation_plots,
    main
)

# Module-level fixtures (outside any class)
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
def mock_data():
    """Create mock data for testing."""
    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    parking_ids = ['P1', 'P2', 'P3']
    
    data = []
    for parking_id in parking_ids:
        for date in dates:
            data.append({
                'timestamp': date,
                'parking_id': parking_id,
                'available_spaces': np.random.randint(0, 100),
                'total_spaces': 100,
                'prediction_code': np.random.randint(0, 6),  # Assuming 6 classes
                'latitude': 41.3851 + np.random.uniform(-0.01, 0.01),
                'longitude': 2.1734 + np.random.uniform(-0.01, 0.01)
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    parking_ids = ['P1', 'P2', 'P3']
    
    data = []
    for parking_id in parking_ids:
        for date in dates:
            data.append({
                'timestamp': date,
                'parking_id': parking_id,
                'available_spaces': np.random.randint(0, 100),
                'total_spaces': 100,
                'prediction_code': np.random.randint(0, 6),  # Assuming 6 classes
                'latitude': 41.3851 + np.random.uniform(-0.01, 0.01),
                'longitude': 2.1734 + np.random.uniform(-0.01, 0.01)
            })
    
    return pd.DataFrame(data)

class TestTrainBaseline:
    """Unit tests for baseline training functionality"""

    def test_module_imports(self):
        """Basic test that the module can be imported and main functions exist."""
        import src.modeling.train_baseline as train_baseline
        assert hasattr(train_baseline, 'load_config')
        assert hasattr(train_baseline, 'load_and_prepare_data')
        assert hasattr(train_baseline, 'validate_data')
        assert hasattr(train_baseline, 'create_time_series_split')
        assert hasattr(train_baseline, 'train_and_evaluate_model')
        assert hasattr(train_baseline, 'generate_evaluation_plots')
        assert hasattr(train_baseline, 'main')

    # TODO: Re-implement when handle_duplicate_timestamps is added
    # def test_handle_duplicate_timestamps(self):
    #     pass

    # TODO: Re-implement when validate_data_frequency is added
    # def test_validate_data_frequency(self):
    #     pass

    # TODO: Re-implement when calculate_class_weights is added
    # def test_calculate_class_weights(self):
    #     pass

    # TODO: Re-implement when calculate_metrics is added
    # def test_calculate_metrics(self):
    #     pass

    # TODO: Re-implement when analyze_features is added
    # def test_analyze_features(self):
    #     pass

    # TODO: Re-implement when calculate_vif is added
    # def test_calculate_vif(self):
    #     pass

    # TODO: Re-implement when more baseline functions are available

    # Test data fixtures
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

    def test_handle_duplicate_timestamps(self):
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

    def test_validate_data_frequency(self):
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

    def test_calculate_class_weights(self):
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

    def test_calculate_metrics(self):
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

    def test_train_model(self, sample_data):
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

    def test_analyze_features(self, sample_data, tmp_path):
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

    def test_calculate_vif(self, sample_data):
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

    def test_create_time_series_split(self, mock_data):
        """Test time series split creation."""
        tscv = create_time_series_split(mock_data)
        assert tscv is not None
        # Test that splits are created correctly
        splits = list(tscv.split(mock_data))
        assert len(splits) > 0

    def test_train_and_evaluate_model(self, mock_data):
        """Test model training and evaluation."""
        # Create features
        features = ['available_spaces', 'total_spaces']
        tscv = create_time_series_split(mock_data)
        
        # Train and evaluate
        all_preds, all_true, fold_metrics, feature_importance = train_and_evaluate_model(
            mock_data, features, tscv
        )
        
        assert len(all_preds) > 0
        assert len(all_true) > 0
        assert len(fold_metrics) > 0
        assert len(feature_importance) > 0

    def test_generate_evaluation_plots(self, mock_data, tmp_path):
        """Test evaluation plot generation."""
        # Create temporary output directories
        plots_dir = tmp_path / "plots"
        metrics_dir = tmp_path / "metrics"
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Generate some mock results
        all_preds = np.random.randint(0, 6, size=100)
        all_true = np.random.randint(0, 6, size=100)
        fold_metrics = [{'fold': i, 'accuracy': 0.8} for i in range(3)]
        feature_importance = pd.DataFrame({
            'feature': ['f1', 'f2'],
            'importance': [0.6, 0.4]
        })
        
        # Test plot generation
        generate_evaluation_plots(
            all_preds, all_true, fold_metrics, feature_importance,
            str(plots_dir), str(metrics_dir)
        )
        
        # Check if files were created
        assert len(list(plots_dir.glob('*.png'))) > 0
        assert len(list(metrics_dir.glob('*.json'))) > 0

    def test_load_and_prepare_data(self, mock_data, tmp_path):
        """Test data loading and preparation."""
        # Save mock data to temporary parquet file
        data_path = tmp_path / "test_data.parquet"
        mock_data.to_parquet(data_path)
        
        # Test loading
        df = load_and_prepare_data(str(data_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'timestamp' in df.columns
        assert 'prediction_code' in df.columns

    def test_edge_cases_extended(self, config):
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

    def test_data_quality_edge_cases(self, config):
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

    def test_time_series_split(self, config):
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