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
def raw_data():
    """Create raw data that would be loaded from parquet."""
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
                'latitude': np.random.uniform(40.0, 41.0),
                'longitude': np.random.uniform(-74.0, -73.0),
                'parking_type': np.random.choice(['underground', 'surface', 'multi_level']),
                'pricing_tier': np.random.choice(['low', 'medium', 'high']),
                'is_24h': np.random.choice([0, 1]),
                'has_security': np.random.choice([0, 1]),
                'has_ev_charging': np.random.choice([0, 1]),
                'is_underground': np.random.choice([0, 1]),
                'is_public': np.random.choice([0, 1]),
                'opening_hour': np.random.randint(0, 24),
                'closing_hour': np.random.randint(0, 24)
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_config():
    """Create mock configuration."""
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

class TestTrainBaselineIntegration:
    """Integration tests for baseline training functionality"""

    def test_module_imports(self):
        """Basic test that the module can be imported and main functions exist."""
        import src.modeling.train_baseline
        assert hasattr(train_baseline, 'load_config')
        assert hasattr(train_baseline, 'load_and_prepare_data')
        assert hasattr(train_baseline, 'validate_data')
        assert hasattr(train_baseline, 'create_time_series_split')
        assert hasattr(train_baseline, 'train_and_evaluate_model')
        assert hasattr(train_baseline, 'generate_evaluation_plots')
        assert hasattr(train_baseline, 'main')

    # TODO: Re-implement when end-to-end and feature engineering integration is available
    # def test_full_pipeline(self):
    #     pass

    # TODO: Re-implement when error handling and config handling are available
    # def test_pipeline_with_missing_data(self):
    #     pass
    # def test_pipeline_with_invalid_data(self):
    #     pass
    # def test_config_handling_integration(self):
    #     pass
    # def test_parallel_processing_integration(self):
    #     pass

    def test_full_pipeline(self, mock_data, tmp_path):
        """Test the full training pipeline."""
        # Create temporary directories
        data_dir = tmp_path / "data"
        plots_dir = tmp_path / "plots"
        metrics_dir = tmp_path / "metrics"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save mock data
        data_path = data_dir / "test_data.parquet"
        mock_data.to_parquet(data_path)
        
        # Load and prepare data
        df = load_and_prepare_data(str(data_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Create features
        features = ['available_spaces', 'total_spaces']
        
        # Create time series split
        tscv = create_time_series_split(df)
        assert tscv is not None
        
        # Train and evaluate model
        all_preds, all_true, fold_metrics, feature_importance = train_and_evaluate_model(
            df, features, tscv
        )
        
        # Verify results
        assert len(all_preds) > 0
        assert len(all_true) > 0
        assert len(fold_metrics) > 0
        assert len(feature_importance) > 0
        
        # Generate evaluation plots
        generate_evaluation_plots(
            all_preds, all_true, fold_metrics, feature_importance,
            str(plots_dir), str(metrics_dir)
        )
        
        # Verify output files
        assert len(list(plots_dir.glob('*.png'))) > 0
        assert len(list(metrics_dir.glob('*.json'))) > 0

    def test_end_to_end_training_flow(self, raw_data, tmp_path, mock_config):
        """Test the complete training flow from raw data to model evaluation."""
        # Save raw data to temporary parquet file
        data_path = tmp_path / "test_data.parquet"
        raw_data.to_parquet(data_path)
        
        # Mock environment variables and config
        with patch.dict(os.environ, {'DATA_PATH': str(data_path)}), \
             patch('src.modeling.train_baseline.load_config', return_value=mock_config):
            
            # Run the complete pipeline
            df = load_and_prepare_data()
            
            # Verify data preparation
            assert isinstance(df, pd.DataFrame)
            assert 'occupancy_class' in df.columns
            assert not df.isnull().any().any()
            
            # Create time series split
            tscv = create_time_series_split(df)
            
            # Train model
            X = df.drop(['occupancy_class', 'timestamp', 'parking_id'], axis=1)
            y = df['occupancy_class']
            model, params = train_model(X, y)
            
            # Verify model training
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
            
            # Evaluate model
            metrics = evaluate_model(model, X, y, str(tmp_path))
            
            # Verify evaluation
            assert 'accuracy' in metrics
            assert 'f1_weighted' in metrics
            assert 'f1_macro' in metrics
            
            # Verify output files
            assert os.path.exists(os.path.join(str(tmp_path), 'confusion_matrix.png'))
            assert os.path.exists(os.path.join(str(tmp_path), 'feature_importance.png'))
            assert os.path.exists(os.path.join(str(tmp_path), 'metrics.json'))

    def test_feature_engineering_integration(self, raw_data, tmp_path, mock_config):
        """Test integration between feature engineering and model training."""
        # Initialize feature engineering pipeline
        pipeline = FeatureEngineeringPipeline(mock_config)
        
        # Create features
        df = pipeline.create_features(raw_data)
        
        # Verify feature creation
        assert isinstance(df, pd.DataFrame)
        assert not df.isnull().any().any()
        
        # Train model with engineered features
        X = df.drop(['timestamp', 'parking_id'], axis=1)
        y = df['occupancy_class']
        model, params = train_model(X, y)
        
        # Verify model can use engineered features
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y.unique()))

    def test_cross_validation_integration(self, raw_data, tmp_path, mock_config):
        """Test integration of cross-validation with feature engineering and model training."""
        # Create features
        pipeline = FeatureEngineeringPipeline(mock_config)
        df = pipeline.create_features(raw_data)
        
        # Create time series split
        tscv = create_time_series_split(df)
        
        # Perform cross-validation
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
            # Split data
            X_train = df.iloc[train_idx].drop(['timestamp', 'parking_id', 'occupancy_class'], axis=1)
            y_train = df.iloc[train_idx]['occupancy_class']
            X_val = df.iloc[val_idx].drop(['timestamp', 'parking_id', 'occupancy_class'], axis=1)
            y_val = df.iloc[val_idx]['occupancy_class']
            
            # Train model
            model, params = train_model(X_train, y_train)
            
            # Evaluate
            metrics = evaluate_model(model, X_val, y_val, str(tmp_path), fold)
            cv_scores.append(metrics)
        
        # Verify cross-validation results
        assert len(cv_scores) == tscv.n_splits
        assert all('accuracy' in scores for scores in cv_scores)
        assert all('f1_weighted' in scores for scores in cv_scores)

    def test_model_persistence_integration(self, raw_data, tmp_path, mock_config):
        """Test integration of model training, saving, and loading."""
        # Create features and train model
        pipeline = FeatureEngineeringPipeline(mock_config)
        df = pipeline.create_features(raw_data)
        X = df.drop(['timestamp', 'parking_id', 'occupancy_class'], axis=1)
        y = df['occupancy_class']
        model, params = train_model(X, y)
        
        # Save model
        model_path = os.path.join(str(tmp_path), 'test_model.pkl')
        model.save_model(model_path)
        
        # Load model and verify predictions
        loaded_model = lgb.Booster(model_file=model_path)
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_error_handling_integration(self, raw_data, tmp_path, mock_config):
        """Test error handling across the pipeline."""
        # Test with invalid data
        invalid_data = raw_data.copy()
        invalid_data['available_spaces'] = -1  # Invalid value
        
        with pytest.raises(ValueError):
            pipeline = FeatureEngineeringPipeline(mock_config)
            pipeline.create_features(invalid_data)
        
        # Test with missing required columns
        missing_col_data = raw_data.drop('timestamp', axis=1)
        
        with pytest.raises(ValueError):
            pipeline = FeatureEngineeringPipeline(mock_config)
            pipeline.create_features(missing_col_data)
        
        # Test with empty dataframe
        empty_data = pd.DataFrame(columns=raw_data.columns)
        
        with pytest.raises(ValueError):
            pipeline = FeatureEngineeringPipeline(mock_config)
            pipeline.create_features(empty_data)

    def test_config_handling_integration(self, raw_data, tmp_path):
        """Test configuration handling across the pipeline."""
        # Test with missing config
        with patch('src.modeling.train_baseline.load_config', return_value={}):
            with pytest.raises(ValueError):
                pipeline = FeatureEngineeringPipeline({})
                pipeline.create_features(raw_data)
        
        # Test with invalid config
        invalid_config = {
            'feature_engineering': {
                'temporal_features': {'enabled': 'invalid'}  # Should be boolean
            }
        }
        
        with pytest.raises(ValueError):
            pipeline = FeatureEngineeringPipeline(invalid_config)
            pipeline.create_features(raw_data)

    def test_parallel_processing_integration(self, raw_data, tmp_path, mock_config):
        """Test parallel processing integration."""
        # Create large dataset
        large_data = pd.concat([raw_data] * 10)
        
        # Add parallel processing configuration
        mock_config['parallel_processing'] = {'enabled': True}
        
        # Test with parallel processing enabled
        with patch('src.modeling.feature_engineering_v2.ray.init') as mock_ray_init, \
             patch('src.modeling.feature_engineering_v2.ray.shutdown') as mock_ray_shutdown:
            
            # Run pipeline
            pipeline = FeatureEngineeringPipeline(mock_config)
            df = pipeline.create_features(large_data)
            
            # Verify Ray was initialized and shut down
            mock_ray_init.assert_called_once()
            mock_ray_shutdown.assert_called_once()
            
            # Verify results
            assert isinstance(df, pd.DataFrame)
            assert not df.isnull().any().any()
            
            # Cleanup
            del pipeline 