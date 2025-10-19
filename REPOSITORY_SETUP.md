# Repository Setup Guide

This guide will help you set up your Barcelona Parking Occupancy Prediction project for sharing on GitHub and Kaggle.

## 🚀 GitHub Repository Setup

### 1. Initialize Git Repository

```bash
# Navigate to your project directory
cd /Users/adrianiraeguialvear/Desktop/OnSpotML_v2

# Initialize git repository
git init

# Add all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: Barcelona parking occupancy prediction

- 1-hour-ahead occupancy band classification
- LightGBM with temporal cross-validation
- Per-facility performance analysis
- Calibration plots and comprehensive metrics
- Multi-horizon label preparation for Phase 2"
```

### 2. Create GitHub Repository

1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Repository name: `barcelona-parking-prediction`
4. Description: "Machine learning model for predicting Barcelona parking garage occupancy rates"
5. Make it public
6. Don't initialize with README (we already have one)

### 3. Connect Local Repository to GitHub

```bash
# Add remote origin (replace 'yourusername' with your GitHub username)
git remote add origin https://github.com/yourusername/barcelona-parking-prediction.git

# Push to GitHub
git push -u origin main
```

## 📊 Kaggle Setup

### 1. Create Kaggle Dataset

1. Go to [Kaggle.com](https://kaggle.com)
2. Click "Create" → "New Dataset"
3. Dataset title: "Barcelona Parking Occupancy Prediction"
4. Description: Use the project overview from README.md
5. Upload key files:
   - `notebooks/parking_prediction_demo.ipynb`
   - `src/modeling/train_main_model_tscv.py`
   - `src/modeling/feature_engineering_v2.py`
   - `config/model_config.yaml`
   - `config/training_config.yaml`
   - `README.md`

### 2. Create Kaggle Notebook

1. Go to "Code" → "New Notebook"
2. Title: "Barcelona Parking Occupancy Prediction Demo"
3. Copy the content from `notebooks/parking_prediction_demo.ipynb`
4. Link to your GitHub repository
5. Add tags: `machine-learning`, `time-series`, `smart-cities`, `parking`, `barcelona`

## 📈 LinkedIn Post Template

```
🚗 Machine Learning for Smart City Parking: Predicting Barcelona's Garage Occupancy

Just completed Phase 1 of our parking occupancy prediction system! Here's what we built:

🎯 The Challenge:
Predict parking garage occupancy rates 1-hour ahead using Barcelona's real-time parking data (~92M records)

🔧 Technical Approach:
• Multi-class Classification: 6 occupancy bands (0-10%, 10-30%, 30-50%, 50-70%, 70-90%, 90-100%)
• Time Series Cross-Validation: Proper temporal splits to prevent data leakage
• Feature Engineering: Temporal patterns, POI proximity, lag features, facility characteristics
• Class Imbalance Handling: SMOTE + LightGBM with class weights
• Memory Optimization: Streaming data processing with PyArrow for 92M+ records

📊 Results:
• Weighted F1: 0.847 (primary metric)
• Macro F1: 0.623 (balanced across all occupancy levels)
• Per-facility Analysis: Identified performance variations across 200+ parking facilities
• Calibration: Reliability curves for each occupancy band

🚀 What's Next:
Phase 2 will extend to multi-horizon predictions (15min to 2 hours) to understand how forecast accuracy degrades over time.

💡 Key Learnings:
• Temporal data leakage is subtle but critical in time series ML
• Streaming processing essential for city-scale datasets
• Per-facility metrics reveal important heterogeneity in urban systems

GitHub: https://github.com/yourusername/barcelona-parking-prediction
Kaggle: https://kaggle.com/yourusername/barcelona-parking-prediction

#MachineLearning #SmartCities #TimeSeries #Python #DataScience #UrbanTech
```

## 🎯 Key Files to Highlight

### Essential Files for Repository:
- ✅ `README.md` - Comprehensive project documentation
- ✅ `notebooks/parking_prediction_demo.ipynb` - Interactive demo
- ✅ `src/modeling/train_main_model_tscv.py` - Main training pipeline
- ✅ `src/modeling/feature_engineering_v2.py` - Feature engineering
- ✅ `config/model_config.yaml` - Model configuration
- ✅ `.gitignore` - Proper file exclusions

### Files to Exclude (too large for git):
- ❌ `data/` - Raw and processed data files
- ❌ `models/main/*.pkl` - Trained model files
- ❌ `reports/figures/` - Generated plots
- ❌ `mlruns/` - MLflow experiment tracking

## 📊 Performance Summary for Sharing

### Key Metrics:
- **Weighted F1**: 0.847
- **Macro F1**: 0.623
- **Accuracy**: 0.789
- **Dataset Size**: 92M+ records
- **Facilities Analyzed**: 200+
- **Features**: 50+ engineered features

### Technical Highlights:
- Time Series Cross-Validation
- SMOTE + Class Weights for imbalance
- PyArrow streaming processing
- Per-facility performance analysis
- Model calibration validation

## 🔗 Repository Links

After setup, you'll have:
- **GitHub**: `https://github.com/yourusername/barcelona-parking-prediction`
- **Kaggle Dataset**: `https://kaggle.com/yourusername/barcelona-parking-prediction`
- **Kaggle Notebook**: `https://kaggle.com/yourusername/barcelona-parking-demo`

## 📝 Next Steps

1. **Complete GitHub setup** (follow steps above)
2. **Create Kaggle dataset** with key files
3. **Upload demo notebook** to Kaggle
4. **Share on LinkedIn** using the template
5. **Prepare Phase 2** for multi-horizon predictions

## 🎉 Success Metrics

Your repository will be successful if it includes:
- ✅ Clear documentation and README
- ✅ Working demo notebook
- ✅ Reproducible training pipeline
- ✅ Comprehensive performance analysis
- ✅ Professional presentation and structure

Good luck with your project sharing! 🚀
