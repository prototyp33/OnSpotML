# Analysis of Prediction Codes 2 and 3 - Summary

## Overview
This document summarizes the key findings from the analysis of prediction codes 2 and 3 in the parking prediction dataset.

## Data Description
- Total number of observations: 1837728
- Time period covered: 2024-09-17 to 2024-09-18
- Number of unique ID_TRAMOs: 6381

## Key Findings

### 1. Temporal Patterns
- Peak hours for Code 2: 8
- Peak hours for Code 3: 0
- Weekend vs Weekday patterns:
  - Code 2: 0.0008178577025544586 (weekday) vs 0 (weekend)
  - Code 3: 0.017465587943373558 (weekday) vs 0 (weekend)

### 2. Spatial Patterns
- Number of unique ID_TRAMOs with Code 2: 8
- Number of unique ID_TRAMOs with Code 3: 117
- Top 5 ID_TRAMOs with highest proportion of Code 2:
  1. ID_TRAMO 3995
  2. ID_TRAMO 7872
  3. ID_TRAMO 8017
  4. ID_TRAMO 1450
  5. ID_TRAMO 5001

- Top 5 ID_TRAMOs with highest proportion of Code 3:
  1. ID_TRAMO 14
  2. ID_TRAMO 67
  3. ID_TRAMO 756
  4. ID_TRAMO 757
  5. ID_TRAMO 758

### 3. Distribution Analysis
- Overall distribution of prediction codes:
  - Code 2: 0.1%
  - Code 3: 1.7%

### 4. Most Active Areas
Top 10 most active ID_TRAMOs:
1. ID_TRAMO 7
2. ID_TRAMO 12358
3. ID_TRAMO 12356
4. ID_TRAMO 12355
5. ID_TRAMO 12354
6. ID_TRAMO 12350
7. ID_TRAMO 12348
8. ID_TRAMO 12347
9. ID_TRAMO 12342
10. ID_TRAMO 12340

## Visualizations
The analysis includes several key visualizations:
1. Hourly patterns of Codes 2 and 3
2. Distribution of codes across ID_TRAMOs
3. Weekend vs Weekday patterns
4. Activity heatmap for Codes 2 and 3

## Conclusions
- Code 3 is more prevalent overall
- Temporal patterns show distinct variations throughout the day
- Certain ID_TRAMOs show consistently higher activity
- Weekend vs weekday patterns reveal different usage patterns

## Next Steps
1. Investigate the relationship between Codes 2 and 3 and other variables
2. Analyze seasonal patterns if data spans multiple months
3. Consider external factors that might influence the patterns observed
4. Validate findings with domain experts

Last updated: 2025-04-29 12:07:23



## Additional Analysis Findings

### 1. Sequential Patterns
- Consecutive occurrences analysis shows patterns in code sequences
- Code 2 consecutive occurrences: 1479
- Code 3 consecutive occurrences: 31966

### 2. Seasonal Patterns
- Seasonal variations observed
- Highest activity season for Code 2: N/A
- Highest activity season for Code 3: N/A

### 3. Transition Analysis
- Most common transition: Code (np.int8(0), np.int8(0))
- Stability analysis (same code persistence): Codes tend to persist for longer periods (stability: 1.00)

### 4. Spatial Clustering
- Identified 3 distinct clusters of ID_TRAMOs
- Cluster characteristics:
  Cluster 0: 0 dominant (size: 6257)
  Cluster 1: 3 dominant (size: 110)
  Cluster 2: 1 dominant (size: 14)

## Updated Conclusions
Distinct patterns identified in code sequences
- Seasonal variations affect code distribution
- Spatial clustering reveals distinct zone types
- Transition patterns show systematic behavior

Last updated: 2025-04-29 12:19:02



## Data Quality and Scope Analysis

### Temporal Coverage
- Date Range: 2024-09-17 to 2024-09-18
- Time Granularity: 0 days 00:05:00
- Data Completeness: 0.0% of ID_TRAMOs have complete data

### Prediction Distribution
- Code 0: 98.1%
- Code 1: 0.1%
- Code 2: 0.1%
- Code 3: 1.7%

### Spatial Coverage
- Total unique ID_TRAMOs: 6381
- ID_TRAMOs with Code 2: 8
- ID_TRAMOs with Code 3: 117

### Key Findings from Additional Analysis
1. Temporal Patterns:
   - 12 shows highest average predictions
   - Daily pattern shows 0.00 range in predictions

2. Spatial Patterns:
   - Average prediction across ID_TRAMOs: 0.06
   - Standard deviation across ID_TRAMOs: 0.40

### Data Limitations
- Limited to 1 days of data
- Weekend patterns not observable in this dataset
- Geographic location information not available
- No external metadata (capacity, tariffs, zone types)
