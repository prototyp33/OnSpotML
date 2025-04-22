Version: 1.0

Date: April 17, 2025

Status: Draft

## Introduction

This document serves as the central hub and comprehensive plan for the Barcelona Parking Occupancy Prediction project. It outlines the project's objectives, context, data strategy, technical approach, deployment, monitoring, and strategic alignment, following best practices for machine learning project development. Hyperlinks connect this document to detailed sub-documents within the project's Google Drive structure.

## 1. Problem Definition and Context

- **Clearly Defined Objective:**
    - **Primary Goal:** To develop, evaluate, and potentially deploy a machine learning model that accurately predicts parking occupancy rates (e.g., percentage full or estimated free spots) for specific public parking areas (initially focusing on [Specify target areas, e.g., selected AREA Blue/Green zones in Eixample and key B:SM garages]) in Barcelona.
    - **Prediction Horizon:** Forecasts will target [Specify, e.g., 15, 30, and 60 minutes] into the future.
    - **Value Proposition:** Provide drivers with reliable future parking availability information to reduce search time, decrease associated traffic congestion and emissions, and inform better parking choices. For parking operators (e.g., B:SM), insights can inform operational decisions.
- **Urban Context Analysis (Barcelona):**
    - Barcelona faces significant parking pressure due to high vehicle density, limited on-street space (partially reduced by initiatives like Superblocks and terrace expansions), complex regulations (AREA zones with resident priority, time limits), and high demand in central/commercial areas. [cite: 1]
    - Congestion caused by drivers searching for parking is a known issue, impacting travel times and air quality.
    - Existing apps (Smou, Parclick, etc.) offer payment and booking, but predictive availability remains a key opportunity. [cite: 1]
    - **Link:** [`📄 Barcelona Parking Market Analysis`] (Your uploaded doc)
- **Stakeholder Mapping:**
    - **End-Users:** Residents, commuters, visitors, delivery services (DUM zones).
    - **Municipal Authorities:** Ajuntament de Barcelona (City Council), Barcelona de Serveis Municipals (B:SM - manages AREA and public garages).
    - **Potential Data Providers:** B:SM, Open Data BCN, Weather Services, Event Organizers.
    - **Private Operators:** Owners of private garages (potential for future integration/partnership).
    - **Internal Team:** Project developers, data scientists.

## 2. Data Strategy

- **Comprehensive Data Sources:**
    - **Historical Occupancy:** Seek access via B:SM/Ajuntament for AREA zone sensor data (if available) and B:SM garage occupancy records. Plan for ongoing, automated collection if API access is granted.
    - **Static Parking Data:** Utilize Open Data BCN for AREA zone definitions, B:SM garage locations/capacities.
    - **Real-time Traffic:** Investigate APIs (e.g., Google Maps, Waze, potentially Open Data BCN feeds) for flow/incident data.
    - **Weather:** Integrate historical and forecast data from a chosen weather API (e.g., [Specify Weather API Provider]).
    - **Local Events:** Use Open Data BCN public holiday calendar; investigate APIs or scraping for major events (concerts, football matches, festivals).
    - **Public Transport:** Explore Open Data BCN for usage stats (Metro, Bus) if available, as it may correlate negatively with parking demand.
    - **Urban Planning Data:** Geospatial data (district boundaries, points of interest) from Open Data BCN.
    - **Link:** [`📄 Data Source Investigation - BCN Parking`]
- **Variable Selection and Justification:**
    - Prioritize: Timestamp (hour, day, week, month, holiday flag), Location ID/Coordinates, Lagged Occupancy (previous intervals), Weather (temp, precipitation), Event Flags (major local events).
    - Secondary: Zone Type (Blue/Green/Garage), Capacity, Day Type (weekday/weekend), Time-based cyclical features (sine/cosine transforms), Proximity to POIs, Real-time traffic indicators (if available). Weighting to be determined during feature engineering and model evaluation.
- **Data Quality Management:**
    - **Missing Data:** Analyze patterns (MCAR, MAR, MNAR). Employ appropriate imputation techniques (e.g., mean/median/mode for simple cases, time-series interpolation like linear or spline, or model-based imputation like KNNImputer or MICE for more complex scenarios). Document chosen methods.
    - **Outliers:** Identify using statistical methods (IQR, Z-score) or visualization. Investigate causes (sensor errors, real events?) before deciding on removal, capping, or transformation.
    - **Inconsistencies:** Validate timestamps, location IDs, occupancy ranges (0-100%). Implement data validation checks in processing scripts.
    - **Link:** [`📄 Data Processing Steps`]
- **Data Privacy and Compliance:**
    - Adhere strictly to GDPR. Anonymize any user-specific data if ever incorporated (unlikely for this phase).
    - Focus on aggregated occupancy data. Ensure data sharing agreements (if needed, e.g., with B:SM) comply with regulations. Document compliance measures.

## 3. Feature Engineering

- **Domain-Specific Features:**
    - `time_since_last_meal_peak` (lunch/dinner hours influence)
    - `distance_to_nearest_metro/bus_stop`
    - `poi_density_nearby` (shops, restaurants, offices)
    - `zone_interaction` (e.g., occupancy spillover from adjacent zones)
    - Categorical encoding for `zone_type`, `district`.
- **Temporal and Interaction Features:**
    - Cyclical features for time of day, day of week, month using sine/cosine transformations.
    - Lagged occupancy features (e.g., `occupancy_1h_ago`, `occupancy_24h_ago`).
    - Rolling window statistics (e.g., `avg_occupancy_last_3h`).
    - Interaction terms: `is_weekend * hour_of_day`, `is_event_day * district`, `weather_condition * time_of_day`.
- **Iterative Feature Evaluation:**
    - Use techniques like feature importance plots from tree-based models (Gradient Boosting, Random Forest).
    - Employ permutation importance to assess feature relevance.
    - Consider using SHAP values to understand feature contributions during model development and potentially for final model explainability. Refine feature set based on impact and complexity.
    - **Link:** [`📄 Feature Engineering - BCN Parking`]

## 4. Model Development

- **Algorithm Selection:**
    - **Baseline:** Simple models (predict last value, predict historical average for time/day).
    - **Time Series:** ARIMA/SARIMA (for individual locations if sufficient history), Prophet (good handling of seasonality/holidays).
    - **Machine Learning:** Gradient Boosting (XGBoost, LightGBM - handle tabular data well, robust), Random Forests.
    - **Deep Learning:** LSTMs or GRUs if complex sequential dependencies are identified and sufficient data is available.
    - Justification: Start with simpler, interpretable models. Move to more complex ones if performance warrants. Choice depends on data granularity, feature interactions, and scalability needs.
- **Training and Validation:**
    - **Data Splitting:** Time-series aware splitting (e.g., train on older data, validate on more recent, test on latest unseen data). Avoid random shuffling that breaks temporal order.
    - **Cross-Validation:** Use time-series cross-validation techniques (e.g., `TimeSeriesSplit` in scikit-learn) for hyperparameter tuning.
    - **Hyperparameter Tuning:** Employ methods like Grid Search, Randomized Search, or Bayesian Optimization (e.g., using Optuna or Hyperopt).
- **Preventing Overfitting:**
    - Regularization (L1/L2 for linear models/NNs).
    - Early stopping based on validation set performance.
    - Dropout (for NNs).
    - Feature selection/reduction.
    - Using ensemble methods (inherent in RF, GBM).
- **Transfer Learning:**
    - Consider only if facing severe data scarcity in specific zones *and* if relevant pre-trained models exist (e.g., traffic prediction models, though direct parking occupancy transfer learning is less common). Evaluate feasibility carefully.
    - **Link:** [`📄 Model Design & Experiments Log`]

## 5. Model Evaluation

- **Comprehensive Metrics:**
    - **Regression:** MAE (Mean Absolute Error - interpretable), RMSE (Root Mean Squared Error - penalizes large errors), MAPE (Mean Absolute Percentage Error - relative error), R² (Coefficient of Determination - proportion of variance explained).
    - **Classification (if predicting categories like 'Low/Medium/High' occupancy):** Accuracy, Precision, Recall, F1-Score, AUC-ROC/PR curves (especially if classes are imbalanced).
    - Monitor metrics on validation and test sets. Compare against baseline performance.
- **Continuous Monitoring:**
    - Evaluate model performance on new data batches as they become available (post-deployment). Track metric trends over time to detect drift.
    - **Link:** [`📄 Model Evaluation Results`]

## 6. Deployment and Integration

- **API and Backend Integration (Plan):**
    - Develop a RESTful API using Flask or FastAPI.
    - Define endpoints: e.g., `/predict` (input: location, timestamp; output: predicted occupancy), `/health` (status check), potentially `/retrain` trigger.
    - Containerize the application (Docker) for portability and scalability.
    - Choose deployment platform (e.g., GCP Cloud Run, AWS SageMaker/Lambda, Azure ML).
- **Version Control and Security:**
    - Use Git for code version control.
    - Use MLflow or DVC for model/data versioning and experiment tracking.
    - Implement API key authentication or other security measures for API endpoints.
    - Manage secrets (API keys, database credentials) securely (e.g., environment variables, secret manager services).
- **Municipal and Private Data Integration:**
    - If B:SM API access is granted, specify API protocols (REST, SOAP?), authentication methods, data formats (JSON, XML).
    - Define data exchange frequency and error handling mechanisms. Document integration points clearly.

## 7. Monitoring, Maintenance, and Feedback

- **Performance Monitoring Tools:**
    - Log prediction requests, responses, and latencies.
    - Use monitoring dashboards (e.g., Grafana with Prometheus, or cloud provider tools like CloudWatch/Google Cloud Monitoring) to visualize key metrics (prediction accuracy drift, API errors, resource usage).
- **Alerting and Retraining:**
    - Set up alerts (e.g., via PagerDuty, Slack) for significant drops in prediction accuracy (compared to a threshold or baseline), high error rates, or system downtime.
    - Define retraining strategy: Scheduled (e.g., weekly/monthly) and/or triggered by performance degradation alerts or significant data drift detection. Automate retraining pipeline where feasible.
- **User Feedback Loops (Future Phase):**
    - If integrated into a user-facing app, include mechanisms for users to report inaccurate predictions. Use this feedback qualitatively and potentially quantitatively (if structured) to inform model improvements.

## 8. Scalability, Explainability, and Ethics

- **System Scalability:**
    - Design API and backend for horizontal scaling (handling more requests by adding more instances).
    - Optimize database queries and data processing pipelines.
    - Choose cloud services that support auto-scaling.
- **Model Explainability:**
    - Use SHAP or LIME libraries to generate explanations for individual predictions (e.g., "Why is occupancy predicted to be high in this zone?"). This builds trust and aids debugging.
    - Provide global feature importance plots.
- **Ethical Considerations:**
    - Analyze potential biases: Does the model perform equally well across different Barcelona districts (e.g., central vs. peripheral, high-income vs. low-income)? Does data availability skew predictions?
    - Assess impact: Could predictions inadvertently worsen traffic by directing everyone to the same few predicted spots? Consider fairness in accessibility. Document potential biases and mitigation steps.

## 9. Business and Strategic Alignment

- **Market and Competitive Analysis:**
    - Position the predictive model relative to existing Barcelona parking apps (Smou, Parclick, etc.). Highlight the unique value of *prediction* vs. just real-time status or booking.
    - Identify potential integration partners or customers (navigation apps, fleet management, city dashboards).
    - **Link:** [`📄 Barcelona Parking Market Analysis`] (Your uploaded doc)
- **Resource Planning:**
    - Estimate costs: Cloud infrastructure, API usage (weather, maps), development time, ongoing maintenance.
    - Identify required personnel: Data scientists, ML engineers, potentially backend developers.
- **Public-Private Collaboration:**
    - Outline strategy for engaging with B:SM/Ajuntament for data access and potential pilot programs.
    - Define value proposition for the municipality (e.g., reduced congestion, better resource planning).

## 10. Risk Management and Assumptions

- **Assumptions Documentation:**
    - Data Availability: Assumes reliable access to historical and reasonably real-time occupancy data from B:SM/sensors.
    - Data Quality: Assumes data quality is sufficient after cleaning/imputation.
    - Predictability: Assumes parking behavior has predictable patterns influenced by documented variables.
    - User Adoption (if applicable): Assumes users would trust and use the predictions.
- **Risk Identification and Mitigation:**
    - **Risk:** Lack of access to granular/real-time occupancy data. **Mitigation:** Engage early with B:SM, explore alternative proxy data, adjust project scope to available data (e.g., predict only for B:SM garages if AREA data is unavailable).
    - **Risk:** Poor model performance due to unforeseen factors or high randomness. **Mitigation:** Rigorous evaluation, focus on specific predictable scenarios/locations, clearly communicate model limitations.
    - **Risk:** Data drift (parking patterns change). **Mitigation:** Implement robust monitoring and retraining strategy.
    - **Risk:** Regulatory changes affecting parking or data access. **Mitigation:** Stay informed about local policies, design for flexibility.
    - **Risk:** Technical debt / Scalability issues. **Mitigation:** Follow good software engineering practices, plan for scaling during design.

***This document is a living plan and will be updated as the project progresses.*** 