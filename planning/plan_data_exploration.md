# Planning Prompt: Data Source Exploration

**Goal:** Plan the exploration of a specific data source identified in the PRD for the Barcelona Parking Prediction project.

**Instructions for User/AI:**

1.  **Identify Data Source:** Specify the data source to explore (e.g., "B:SM Historical Garage Occupancy", "Open Data BCN AREA Zones", "[Specific Weather API]"). Reference the relevant section in `@BCN_Parking_Prediction_PRD_v1.0.md`.

2.  **Access Method:** Detail how to access the data (e.g., API endpoint, documentation link, file path, database query, contact person for B:SM).

3.  **Key Variables of Interest:** List the variables expected or desired from this source, based on the PRD's Section 2 (Data Strategy) and Section 3 (Feature Engineering).

4.  **Initial Exploration Steps:** Outline the initial analysis plan:
    *   Load a sample of the data.
    *   Check data format, structure, and size.
    *   Identify key columns (timestamps, location identifiers, occupancy measures, relevant features).
    *   Perform initial descriptive statistics (mean, median, min, max, counts).
    *   Visualize distributions and time series patterns for key variables.
    *   Assess initial data quality: missing values (patterns, percentages), potential outliers, inconsistencies.

5.  **Expected Challenges/Questions:** Note any anticipated difficulties (e.g., complex API authentication, unclear variable definitions, data sparsity, time zone issues) or specific questions to answer during exploration.

6.  **Define Next Steps:** What actions will be taken after the initial exploration (e.g., write data loading script, plan specific cleaning steps, formulate questions for data provider)?

**Example Usage:**
"Let's plan the exploration for the Open Data BCN AREA zone definitions, referencing Section 2 of the PRD. We need to find the API endpoint or download link, identify columns for zone ID, geometry/coordinates, and capacity, and check for consistency across districts." 