# Behavior Rules: Barcelona Parking Prediction Project

These rules guide the AI's behavior specifically for the `@BCN_Parking_Prediction_PRD_v1.0.md` project.

## Core Focus

1.  **Primary Objective:** All actions should align with the goal of developing, evaluating, and potentially deploying an ML model to predict parking occupancy in Barcelona, as detailed in the PRD.
2.  **PRD Adherence:** The `@BCN_Parking_Prediction_PRD_v1.0.md` is the primary source of truth. Refer to it for project goals, scope, technical choices, and constraints. When deviating or making assumptions, state them clearly.
3.  **Data-Centricity:** Recognize that data acquisition (esp. from B:SM), quality, and processing are critical. Prioritize robust data handling, time-series awareness, and documentation of data steps.

## Technical Guidelines

1.  **Language & Libraries:** Primarily use Python 3. Standard libraries include `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`. For specific tasks, use libraries mentioned in the PRD: `statsmodels` (ARIMA), `prophet`, `xgboost`/`lightgbm` (GBM), potentially `tensorflow`/`pytorch` (LSTMs/GRUs if deemed necessary), `flask`/`fastapi` (API), `docker`, `mlflow`/`dvc` (MLOps), `shap`/`lime` (Explainability).
2.  **Code Style:** Follow PEP 8 conventions for Python code. Emphasize readability, type hinting (where appropriate), and clear variable/function names.
3.  **Modularity:** Structure code logically (e.g., separate scripts/modules for data loading, processing, feature engineering, training, evaluation, API).
4.  **Time Series:** Apply time-series-specific techniques for data splitting (`TimeSeriesSplit`), cross-validation, and feature engineering (lags, rolling windows, cyclical features) as outlined in the PRD.
5.  **MLOps Practices:** Integrate MLOps considerations early: use Git, assist with setting up experiment tracking (`mlflow`/`dvc`), plan for containerization (`Docker`), and consider monitoring/retraining needs.
6.  **Documentation:** Generate clear docstrings (e.g., Google or NumPy style) for functions/classes. Add comments for complex logic. Assist in maintaining the README and potentially other documentation files linked in the PRD.

## Interaction & Problem Solving

1.  **Reference the PRD:** When proposing solutions or discussing strategies, explicitly link back to relevant sections of the `@BCN_Parking_Prediction_PRD_v1.0.md`.
2.  **Acknowledge Risks:** Be mindful of the risks outlined in the PRD (esp. data availability). Suggest contingency plans or alternative approaches if roadblocks arise.
3.  **Explainability & Ethics:** Keep model explainability (using SHAP/LIME) and ethical considerations (bias analysis) in mind during development, as per the PRD.
4.  **Simplicity First:** Start with simpler models/approaches (baselines, ARIMA, basic ML) before moving to more complex ones (Deep Learning), justifying the need for increased complexity based on performance evaluation.
5.  **Tool Use:** Use tools (codebase search, file reading, web search) proactively to understand context, find relevant information (e.g., about `Open Data BCN` APIs, specific library usage), or verify details.

## How to Use This File

Refer to this file (`@behavior_rules/ml_parking_project.md`) at the beginning of work sessions or when context switching to this project. You can also paste relevant rules directly into prompts to reinforce specific behaviors. 