# OnSpotML - Barcelona Parking Prediction

A machine learning project for predicting parking availability in Barcelona using various data sources including traffic data, POI information, and historical parking patterns.

## 🚀 Features

- Real-time parking availability prediction
- Integration with Barcelona's traffic data
- Point of Interest (POI) analysis
- Historical pattern analysis
- Temporal and spatial feature engineering
- Baseline and advanced ML models

## 📋 Prerequisites

- Python 3.8+
- Git LFS (for large file handling)
- Virtual environment (recommended)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/prototyp33/OnSpotML.git
cd OnSpotML
```

2. Install Git LFS:
```bash
git lfs install
```

3. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
OnSpotML/
├── data/               # Data directory
│   ├── raw/           # Raw data
│   ├── processed/     # Processed data
│   ├── external/      # External data sources
│   └── interim/       # Intermediate data
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── data_ingestion/    # Data collection
│   ├── data_processing/   # Data processing
│   ├── features/          # Feature engineering
│   ├── modeling/          # ML models
│   └── visualization/     # Visualization tools
├── tests/            # Test files
├── docs/             # Documentation
└── reports/          # Generated reports
```

## 🚀 Usage

1. Data Collection:
```bash
python src/data_ingestion/barcelona_data_collector.py
```

2. Feature Engineering:
```bash
python src/features/build_features.py
```

3. Model Training:
```bash
python src/modeling/train_baseline.py
```

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Project Status

🚧 Under Development

## 🔄 Development Workflow

1. Create a feature branch from `develop`
2. Make your changes
3. Write/update tests
4. Create a pull request
5. Get code review
6. Merge to `develop`
7. Deploy to staging (if applicable)
8. Merge to `main` for production

## 📈 Future Improvements

- [ ] Add more advanced ML models
- [ ] Implement real-time prediction API
- [ ] Add more data sources
- [ ] Improve feature engineering
- [ ] Add model monitoring
- [ ] Implement A/B testing framework

## 📞 Support

For support, please open an issue in the GitHub repository.