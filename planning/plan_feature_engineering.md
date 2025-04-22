# Planning Prompt: Feature Engineering Iteration

**Goal:** Plan a specific iteration of feature engineering for the Barcelona Parking Prediction project, based on insights from data exploration or previous modeling attempts.

**Instructions for User/AI:**

1.  **Objective of this Iteration:** Clearly state the goal. (e.g., "Create cyclical time features," "Engineer features based on weather data," "Develop features capturing zone interactions," "Refine lagged occupancy features based on autocorrelation analysis").

2.  **Input Data:** Specify the data source(s) or pre-processed dataframes that will be used as input for this iteration.

3.  **Proposed Features:** List the specific new features to be created. For each feature:
    *   Provide a clear name (e.g., `hour_sin`, `temp_x_is_weekend`, `avg_occupancy_last_3h`).
    *   Describe the calculation logic or transformation.
    *   Justify its potential relevance based on the PRD (Section 3), domain knowledge, or previous findings.

4.  **Implementation Plan:** Outline the steps to implement these features:
    *   Which script/notebook will be modified or created?
    *   What libraries are needed (`pandas`, `numpy`, `scikit-learn`, etc.)?
    *   Any specific functions to be written?

5.  **Evaluation Strategy:** How will the usefulness of these new features be assessed?
    *   Correlation analysis with the target variable?
    *   Feature importance methods (permutation importance, SHAP) after retraining a model?
    *   Comparing model performance (e.g., MAE, RMSE on validation set) with and without the new features?

6.  **Reference PRD:** Link back to relevant parts of `@BCN_Parking_Prediction_PRD_v1.0.md`, especially Section 3 (Feature Engineering) and potentially Section 4 (Model Development) or Section 5 (Model Evaluation).

**Example Usage:**
"Let's plan a feature engineering iteration focused on weather. We'll use the pre-processed data including weather columns. We propose creating `is_precipitating` (binary flag), interaction terms like `temp_x_hour_of_day`, and maybe bucketing temperature. We'll implement this in `feature_engineering.py` and evaluate using feature importance from a LightGBM model, referencing PRD Section 3." 