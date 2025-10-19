import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def generate_synthetic_parking_data(n_samples=500):
    """
    Generate synthetic parking data with realistic occupancy patterns.
    
    Parameters:
    -----------
    n_samples : int
        Number of data points to generate
        
    Returns:
    --------
    pd.DataFrame
        Synthetic parking data with realistic occupancy patterns
    """
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Time-based features
    hours = np.array([t.hour for t in timestamps])
    days = np.array([t.weekday() for t in timestamps])
    months = np.array([t.month for t in timestamps])
    
    # Cyclical encoding of time features
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    dayofweek_sin = np.sin(2 * np.pi * days / 7)
    dayofweek_cos = np.cos(2 * np.pi * days / 7)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    
    # Binary indicators
    is_weekend = np.array([1 if d >= 5 else 0 for d in days])
    is_holiday = np.random.binomial(1, 0.02, n_samples)  # 2% chance of holiday
    is_rush_hour = np.array([1 if h in [8, 9, 17, 18] else 0 for h in hours])
    
    # Weather features
    temperature = np.random.normal(20, 5, n_samples)  # Mean 20°C, std 5°C
    precipitation = np.random.exponential(2, n_samples)  # Exponential distribution
    wind_speed = np.random.gamma(2, 2, n_samples)  # Gamma distribution
    
    # Contextual features
    nearby_events_count = np.random.poisson(2, n_samples)  # Poisson distribution
    distance_to_center = np.random.uniform(0, 10, n_samples)  # 0-10 km
    parking_capacity = np.random.randint(50, 500, n_samples)  # 50-500 spots
    
    # Generate realistic occupancy rate
    base_occupancy = 0.4  # Base occupancy rate
    
    # Time effects
    time_effect = (
        0.2 * hour_sin +  # Daily pattern
        0.1 * dayofweek_sin +  # Weekly pattern
        0.05 * month_sin  # Monthly pattern
    )
    
    # Rush hour effect
    rush_effect = 0.3 * is_rush_hour
    
    # Weekend effect
    weekend_effect = 0.2 * is_weekend
    
    # Holiday effect
    holiday_effect = 0.4 * is_holiday
    
    # Weather effects
    weather_effect = (
        -0.1 * (temperature - 20) / 10 +  # Temperature effect
        -0.2 * precipitation / 5 +  # Precipitation effect
        -0.1 * wind_speed / 10  # Wind effect
    )
    
    # Location effect
    location_effect = -0.15 * distance_to_center / 10
    
    # Event effect
    event_effect = 0.1 * nearby_events_count
    
    # Capacity effect
    capacity_effect = -0.1 * (parking_capacity - 275) / 225
    
    # Combine all effects
    occupancy_rate = (
        base_occupancy +
        time_effect +
        rush_effect +
        weekend_effect +
        holiday_effect +
        weather_effect +
        location_effect +
        event_effect +
        capacity_effect
    )
    
    # Add some random noise
    noise = np.random.normal(0, 0.05, n_samples)
    occupancy_rate += noise
    
    # Ensure occupancy rate is between 0 and 1
    occupancy_rate = np.clip(occupancy_rate, 0, 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dayofweek_sin': dayofweek_sin,
        'dayofweek_cos': dayofweek_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_weekend': is_weekend,
        'temperature': temperature,
        'precipitation': precipitation,
        'wind_speed': wind_speed,
        'is_holiday': is_holiday,
        'nearby_events_count': nearby_events_count,
        'distance_to_center': distance_to_center,
        'parking_capacity': parking_capacity,
        'is_rush_hour': is_rush_hour,
        'occupancy_rate': occupancy_rate
    })
    
    # Save to parquet file
    output_path = Path("data/processed/synthetic_parking_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    
    print(f"Generated synthetic dataset with shape: {df.shape}")
    print(f"Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_parking_data() 