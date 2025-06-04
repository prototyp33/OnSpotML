import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from typing import Dict, Tuple, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PLOTS_OUTPUT_DIR = 'reports/figures/baseline'
METRICS_OUTPUT_DIR = 'reports/metrics/baseline'

def define_parking_occupancy_classes() -> Dict:
    """Define the target classes for parking occupancy prediction."""
    OCCUPANCY_CLASSES = {
        0: {
            'label': 'Empty',
            'range': (0, 10),
            'description': '0-10% occupancy',
            'color': 'green'
        },
        1: {
            'label': 'Very Low',
            'range': (10, 30),
            'description': '10-30% occupancy',
            'color': 'lightgreen'
        },
        2: {
            'label': 'Low',
            'range': (30, 50),
            'description': '30-50% occupancy',
            'color': 'yellow'
        },
        3: {
            'label': 'Medium',
            'range': (50, 70),
            'description': '50-70% occupancy',
            'color': 'orange'
        },
        4: {
            'label': 'High',
            'range': (70, 90),
            'description': '70-90% occupancy',
            'color': 'red'
        },
        5: {
            'label': 'Full',
            'range': (90, 100),
            'description': '90-100% occupancy',
            'color': 'darkred'
        }
    }
    return OCCUPANCY_CLASSES

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable from raw occupancy data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing raw parking data with columns:
        - 'available_spaces': Number of available parking spaces
        - 'total_spaces': Total capacity of the parking facility
        - 'timestamp': Timestamp of the observation
        - 'parking_id': Unique identifier for the parking facility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added target variable 'occupancy_class'
    """
    # Calculate occupancy percentage
    df['occupancy_percentage'] = (
        (df['total_spaces'] - df['available_spaces']) / 
        df['total_spaces'] * 100
    )
    
    # Define class boundaries
    class_boundaries = [0, 10, 30, 50, 70, 90, 100]
    
    # Create target variable
    df['occupancy_class'] = pd.cut(
        df['occupancy_percentage'],
        bins=class_boundaries,
        labels=range(6),
        include_lowest=True
    ).astype(int)
    
    # Add validation checks
    validate_target_variable(df)
    
    return df

def validate_target_variable(df: pd.DataFrame) -> None:
    """
    Validate the target variable creation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the target variable
    """
    # Check for missing values
    if df['occupancy_class'].isnull().any():
        raise ValueError("Found null values in occupancy_class")
    
    # Check for invalid class values
    if not df['occupancy_class'].between(0, 5).all():
        raise ValueError("Found invalid class values in occupancy_class")
    
    # Check class distribution
    class_distribution = df['occupancy_class'].value_counts(normalize=True)
    logger.info("\nTarget Variable Distribution:")
    for cls, prop in class_distribution.items():
        logger.info(f"Class {cls}: {prop:.2%}")
    
    # Check for extreme imbalance
    if class_distribution.min() / class_distribution.max() < 0.1:
        logger.warning("Severe class imbalance detected!")

def analyze_target_variable(df: pd.DataFrame) -> Dict:
    """
    Analyze the target variable distribution and patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the target variable
    
    Returns:
    --------
    Dict
        Dictionary containing analysis results
    """
    analysis = {}
    
    # Overall distribution
    analysis['class_distribution'] = df['occupancy_class'].value_counts(normalize=True).to_dict()
    
    # Time-based patterns
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Hourly patterns
    hourly_patterns = df.groupby('hour')['occupancy_class'].agg([
        'mean',
        'std',
        lambda x: (x == 5).mean()  # Proportion of "Full" class
    ]).rename(columns={'<lambda_0>': 'full_proportion'})
    
    analysis['hourly_patterns'] = hourly_patterns.to_dict()
    
    # Day of week patterns
    daily_patterns = df.groupby('dayofweek')['occupancy_class'].agg([
        'mean',
        'std',
        lambda x: (x == 5).mean()  # Proportion of "Full" class
    ]).rename(columns={'<lambda_0>': 'full_proportion'})
    
    analysis['daily_patterns'] = daily_patterns.to_dict()
    
    # Facility-specific patterns
    facility_patterns = df.groupby('parking_id')['occupancy_class'].agg([
        'mean',
        'std',
        lambda x: (x == 5).mean()  # Proportion of "Full" class
    ]).rename(columns={'<lambda_0>': 'full_proportion'})
    
    analysis['facility_patterns'] = facility_patterns.to_dict()
    
    return analysis

def visualize_target_variable(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations for the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the target variable
    output_dir : str
        Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='occupancy_class')
    plt.title('Distribution of Parking Occupancy Classes')
    plt.xlabel('Occupancy Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # Hourly patterns
    plt.figure(figsize=(12, 6))
    hourly_means = df.groupby('hour')['occupancy_class'].mean()
    plt.plot(hourly_means.index, hourly_means.values)
    plt.title('Average Occupancy Class by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Occupancy Class')
    plt.savefig(os.path.join(output_dir, 'hourly_patterns.png'))
    plt.close()
    
    # Heatmap of occupancy by hour and day
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(
        values='occupancy_class',
        index='hour',
        columns='dayofweek',
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, cmap='YlOrRd')
    plt.title('Average Occupancy Class by Hour and Day')
    plt.xlabel('Day of Week')
    plt.ylabel('Hour of Day')
    plt.savefig(os.path.join(output_dir, 'occupancy_heatmap.png'))
    plt.close()

def prepare_target_variable(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare the target variable for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame containing parking data
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        Processed DataFrame and analysis results
    """
    # Create target variable
    df = create_target_variable(df)
    
    # Validate target variable
    validate_target_variable(df)
    
    # Analyze target variable
    analysis = analyze_target_variable(df)
    
    # Visualize target variable
    visualize_target_variable(df, PLOTS_OUTPUT_DIR)
    
    # Save analysis results
    with open(os.path.join(METRICS_OUTPUT_DIR, 'target_variable_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return df, analysis 