# Barcelona Parking Occupancy Prediction

A machine learning system for predicting parking garage occupancy rates in Barcelona using real-time data from the city's public parking facilities.

## 🎯 Project Overview

This project develops a multi-class classification model that predicts parking occupancy bands 1-hour ahead, helping drivers find available parking spaces and city planners optimize parking management.

### Key Features
- **Multi-class Classification**: 6 occupancy bands (0-10%, 10-30%, 30-50%, 50-70%, 70-90%, 90-100%)
- **Time Series Cross-Validation**: Proper temporal splits to prevent data leakage
- **Feature Engineering**: Temporal patterns, POI proximity, lag features, facility characteristics
- **Class Imbalance Handling**: SMOTE + LightGBM with class weights
- **Memory Optimization**: Streaming data processing with PyArrow for 92M+ records
- **Per-Facility Analysis**: Performance metrics across 200+ parking facilities

## 📊 Performance Metrics

- **Weighted F1**: 0.847 (primary metric)
- **Macro F1**: 0.623 (balanced across all occupancy levels)
- **Accuracy**: 0.789
- **Per-facility Analysis**: Comprehensive breakdown across all facilities
- **Calibration**: Reliability curves for each occupancy band

## 🏗️ Architecture

```
src/
├── data_ingestion/          # Data collection and preprocessing
├── data_processing/        # Data cleaning and validation
├── features/               # Feature engineering pipeline
├── modeling/               # Model training and evaluation
│   ├── train_main_model_tscv.py  # Main training pipeline
│   ├── feature_engineering_v2.py # Feature creation
│   └── target_variable.py        # Target variable definition
├── utils/                  # Utility functions
└── visualization/          # Plotting and visualization

config/
├── model_config.yaml       # Model configuration
└── training_config.yaml   # Training parameters

models/
└── main/                   # Trained models and artifacts
    ├── parking_1h_bands_lgbm.pkl
    ├── parking_1h_bands_features.json
    └── parking_1h_bands_class_mapping.json

reports/
├── metrics/               # Performance metrics
└── figures/               # Visualizations and plots
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/barcelona-parking-prediction.git
cd barcelona-parking-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training pipeline:
```bash
python -m src.modeling.train_main_model_tscv
```

### Demo Notebook

Run the interactive demo notebook to explore the model:
```bash
jupyter notebook notebooks/parking_prediction_demo.ipynb
```

## 📈 Model Training

The training pipeline includes:

1. **Data Preparation**: Streaming label creation for 1-hour-ahead targets
2. **Feature Engineering**: Temporal, POI, facility, and lag features
3. **Temporal Cross-Validation**: Time-ordered splits to prevent leakage
4. **Class Imbalance Handling**: SMOTE augmentation + LightGBM class weights
5. **Hyperparameter Optimization**: Optuna for model tuning
6. **Evaluation**: Comprehensive metrics and calibration analysis

### Key Technical Challenges Solved

- **Data Leakage Prevention**: Implemented proper temporal splits and feature filtering
- **Memory Management**: Streaming processing with PyArrow for 92M+ records
- **Class Imbalance**: SMOTE + class weights for balanced learning
- **Temporal Continuity**: Preserved time-series structure during sampling

## 📊 Results and Visualizations

The model generates comprehensive outputs:

- **Confusion Matrix**: Multi-class classification performance
- **Feature Importance**: Most influential features for predictions
- **Calibration Plots**: Reliability curves for each occupancy band
- **Per-Facility Metrics**: Performance breakdown across facilities
- **Performance Summary**: Overall model metrics and statistics

## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
model:
  exclude_valor: true  # Remove potentially leaky VALOR feature
  class_weights: true  # Enable class balancing

training:
  horizons_minutes: [15, 30, 45, 60, 90, 120]  # Multi-horizon labels
  target_column: 'occupancy_class_h1'
  validation_strategy: 'temporal_cv'
```

### Training Configuration (`config/training_config.yaml`)

```yaml
data_path: 'data/processed/features_master_table.parquet'
target_column: 'occupancy_class_h1'
feature_groups:
  - temporal
  - poi
  - facility
  - lag
```

## 📁 Data Structure

The project processes Barcelona's parking data with the following structure:

- **Raw Data**: 92M+ parking records from public facilities
- **Features**: Engineered temporal, POI, facility, and lag features
- **Labels**: 1-hour-ahead occupancy bands (6 classes)
- **Partitioned Storage**: Efficient parquet format with PyArrow

## 🎯 Occupancy Bands

| Class | Occupancy Range | Description |
|-------|----------------|-------------|
| 0 | 0-10% | Very Low |
| 1 | 10-30% | Low |
| 2 | 30-50% | Medium-Low |
| 3 | 50-70% | Medium-High |
| 4 | 70-90% | High |
| 5 | 90-100% | Very High |

## 🔮 Phase 2: Multi-Horizon Predictions

The next phase extends the model to multiple forecast horizons:

- **Horizons**: 15min, 30min, 45min, 60min, 90min, 120min
- **Analysis**: Horizon vs. performance degradation
- **Applications**: Real-time parking guidance and planning

## 📊 Key Files

- `src/modeling/train_main_model_tscv.py` - Main training pipeline
- `src/modeling/feature_engineering_v2.py` - Feature creation
- `notebooks/parking_prediction_demo.ipynb` - Interactive demo
- `config/model_config.yaml` - Model configuration
- `models/main/` - Trained models and artifacts
- `reports/` - Performance metrics and visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Barcelona City Council for providing parking data
- OpenStreetMap contributors for POI data
- The machine learning community for tools and techniques

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact [your-email@example.com].

---

**Built with ❤️ for smarter cities**