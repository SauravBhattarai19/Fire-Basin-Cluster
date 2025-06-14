#!/usr/bin/env python3
"""
FIRMS Data Detailed Exploration Script
Analyzes temporal and spatial patterns in FIRMS fire detection data
to inform gridding and clustering approach
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_fire_data(file_path):
    """Load fire data and perform detailed exploration"""
    print(f"Loading fire data from: {file_path}")
    
    with open(file_path, 'r') as f:
        fire_data = json.load(f)
    
    fire_df = pd.DataFrame(fire_data)
    
    # Convert data types
    fire_df['confidence'] = pd.to_numeric(fire_df['confidence'], errors='coerce')
    fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date'])
    fire_df['acq_datetime'] = pd.to_datetime(
        fire_df['acq_date'].astype(str) + ' ' + 
        fire_df['acq_time'].str.zfill(4).str[:2] + ':' + 
        fire_df['acq_time'].str.zfill(4).str[2:]
    )
    
    return fire_df

def analyze_temporal_patterns(fire_df):
    """Analyze temporal patterns in fire detections"""
    print("\n=== TEMPORAL PATTERN ANALYSIS ===")
    
    # Daily detection counts
    daily_counts = fire_df.groupby('acq_date').size()
    print(f"Date range: {fire_df['acq_date'].min()} to {fire_df['acq_date'].max()}")
    print(f"Total days: {len(daily_counts)}")
    print(f"Days with detections: {(daily_counts > 0).sum()}")
    print(f"Mean detections per day: {daily_counts.mean():.1f}")
    print(f"Max detections per day: {daily_counts.max()}")
    
    # Hour of day patterns
    fire_df['hour'] = fire_df['acq_datetime'].dt.hour
    hourly_counts = fire_df['hour'].value_counts().sort_index()
    print(f"\nHourly distribution:")
    for hour, count in hourly_counts.items():
        print(f"  Hour {hour:02d}: {count:,} detections")
    
    # Day vs Night patterns
    daynight_stats = fire_df.groupby(['acq_date', 'daynight']).size().unstack(fill_value=0)
    print(f"\nDay vs Night patterns:")
    if 'D' in daynight_stats.columns:
        print(f"  Days with day detections: {(daynight_stats['D'] > 0).sum()}")
    if 'N' in daynight_stats.columns:
        print(f"  Days with night detections: {(daynight_stats['N'] > 0).sum()}")
    
    # Days with both day and night detections
    if 'D' in daynight_stats.columns and 'N' in daynight_stats.columns:
        both_detections = ((daynight_stats['D'] > 0) & (daynight_stats['N'] > 0)).sum()
        print(f"  Days with both day and night: {both_detections}")
    
    return daily_counts, hourly_counts, daynight_stats

def analyze_spatial_patterns(fire_df):
    """Analyze spatial patterns and potential clustering"""
    print("\n=== SPATIAL PATTERN ANALYSIS ===")
    
    # Coordinate precision analysis
    lat_precision = fire_df['latitude'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
    lon_precision = fire_df['longitude'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
    
    print(f"Latitude precision (decimal places):")
    print(f"  Mean: {lat_precision.mean():.1f}, Mode: {lat_precision.mode().iloc[0]}")
    print(f"Longitude precision (decimal places):")
    print(f"  Mean: {lon_precision.mean():.1f}, Mode: {lon_precision.mode().iloc[0]}")
    
    # Round to potential grid resolution
    fire_df['lat_rounded_001'] = np.round(fire_df['latitude'], 3)  # ~100m precision
    fire_df['lon_rounded_001'] = np.round(fire_df['longitude'], 3)
    fire_df['lat_rounded_01'] = np.round(fire_df['latitude'], 2)   # ~1km precision
    fire_df['lon_rounded_01'] = np.round(fire_df['longitude'], 2)
    
    # Count unique locations at different precisions
    unique_exact = len(fire_df[['latitude', 'longitude']].drop_duplicates())
    unique_100m = len(fire_df[['lat_rounded_001', 'lon_rounded_001']].drop_duplicates())
    unique_1km = len(fire_df[['lat_rounded_01', 'lon_rounded_01']].drop_duplicates())
    
    print(f"\nSpatial clustering potential:")
    print(f"  Unique exact locations: {unique_exact:,}")
    print(f"  Unique ~100m grid cells: {unique_100m:,}")
    print(f"  Unique ~1km grid cells: {unique_1km:,}")
    print(f"  Compression ratio (exact -> 1km): {unique_exact/unique_1km:.2f}")
    
    # Repeated locations
    location_counts = fire_df.groupby(['lat_rounded_01', 'lon_rounded_01']).size()
    repeated_locations = location_counts[location_counts > 1]
    
    print(f"\nRepeated 1km locations:")
    print(f"  Locations with multiple detections: {len(repeated_locations):,}")
    print(f"  Max detections at single location: {repeated_locations.max()}")
    print(f"  Mean detections per repeated location: {repeated_locations.mean():.1f}")
    
    return location_counts, repeated_locations

def analyze_detection_quality(fire_df):
    """Analyze detection quality and satellite patterns"""
    print("\n=== DETECTION QUALITY ANALYSIS ===")
    
    # Satellite and instrument combinations
    sat_inst_counts = fire_df.groupby(['satellite', 'instrument']).size()
    print("Satellite-Instrument combinations:")
    for (sat, inst), count in sat_inst_counts.items():
        print(f"  {sat}-{inst}: {count:,} detections")
    
    # Confidence distribution by satellite
    print(f"\nConfidence statistics by satellite:")
    conf_by_sat = fire_df.groupby('satellite')['confidence'].describe()
    print(conf_by_sat)
    
    # FRP distribution
    print(f"\nFire Radiative Power (FRP) statistics:")
    print(fire_df['frp'].describe())
    
    # Quality filtering impact
    high_conf = fire_df[fire_df['confidence'] >= 70]
    print(f"\nHigh confidence (≥70%) filtering:")
    print(f"  Original detections: {len(fire_df):,}")
    print(f"  High confidence detections: {len(high_conf):,}")
    print(f"  Retention rate: {len(high_conf)/len(fire_df)*100:.1f}%")
    
    return conf_by_sat

def analyze_same_location_patterns(fire_df):
    """Analyze patterns of multiple detections at same locations"""
    print("\n=== SAME LOCATION DETECTION PATTERNS ===")
    
    # Group by 1km grid and date
    grid_daily = fire_df.groupby(['lat_rounded_01', 'lon_rounded_01', 'acq_date']).agg({
        'confidence': ['count', 'mean', 'max'],
        'frp': ['mean', 'max', 'sum'],
        'brightness': ['mean', 'max'],
        'daynight': lambda x: list(x.unique()),
        'satellite': lambda x: list(x.unique()),
        'acq_datetime': ['min', 'max']
    }).round(2)
    
    # Flatten column names
    grid_daily.columns = ['_'.join(col).strip() for col in grid_daily.columns]
    grid_daily = grid_daily.reset_index()
    
    # Analyze daily patterns
    daily_detection_counts = grid_daily['confidence_count']
    print(f"Daily detections per grid cell:")
    print(f"  Mean: {daily_detection_counts.mean():.1f}")
    print(f"  Max: {daily_detection_counts.max()}")
    print(f"  Cells with >1 detection per day: {(daily_detection_counts > 1).sum():,}")
    
    # Multiple satellite detections
    multi_sat = grid_daily[grid_daily['satellite_<lambda>'].apply(len) > 1]
    print(f"\nMultiple satellite detections:")
    print(f"  Grid-days with >1 satellite: {len(multi_sat):,}")
    
    # Day and night detections
    day_night = grid_daily[grid_daily['daynight_<lambda>'].apply(lambda x: len(x) > 1)]
    print(f"  Grid-days with both day and night: {len(day_night):,}")
    
    return grid_daily

def create_exploration_visualizations(fire_df, daily_counts, hourly_counts, location_counts):
    """Create visualizations for data exploration"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Daily detection counts
    ax1 = axes[0, 0]
    daily_counts.plot(kind='line', ax=ax1, color='red', alpha=0.7)
    ax1.set_title('Daily Fire Detection Counts', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Detections')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Hourly patterns
    ax2 = axes[0, 1]
    hourly_counts.plot(kind='bar', ax=ax2, color='orange', alpha=0.7)
    ax2.set_title('Hourly Detection Distribution', fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Number of Detections')
    ax2.tick_params(axis='x', rotation=0)
    
    # 3. Spatial density
    ax3 = axes[0, 2]
    ax3.scatter(fire_df['longitude'], fire_df['latitude'], alpha=0.5, s=0.5, c='red')
    ax3.set_title('Geographic Distribution of Detections', fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    
    # 4. Location repeat frequency
    ax4 = axes[1, 0]
    repeat_freq = location_counts.value_counts().sort_index()
    repeat_freq.plot(kind='bar', ax=ax4, color='blue', alpha=0.7)
    ax4.set_title('Detection Frequency per Location', fontweight='bold')
    ax4.set_xlabel('Number of Detections')
    ax4.set_ylabel('Number of Locations')
    ax4.set_yscale('log')
    
    # 5. FRP distribution
    ax5 = axes[1, 1]
    fire_df['frp'].hist(bins=50, ax=ax5, color='purple', alpha=0.7, edgecolor='black')
    ax5.set_title('Fire Radiative Power Distribution', fontweight='bold')
    ax5.set_xlabel('FRP (MW)')
    ax5.set_ylabel('Frequency')
    ax5.set_yscale('log')
    
    # 6. Confidence distribution
    ax6 = axes[1, 2]
    fire_df['confidence'].hist(bins=30, ax=ax6, color='green', alpha=0.7, edgecolor='black')
    ax6.set_title('Confidence Level Distribution', fontweight='bold')
    ax6.set_xlabel('Confidence (%)')
    ax6.set_ylabel('Frequency')
    plt.tight_layout()
    
    # Create organized output directory
    import os
    os.makedirs('output/03_Fire_Analysis', exist_ok=True)
    
    plt.savefig('output/03_Fire_Analysis/firms_data_exploration.png', dpi=300, bbox_inches='tight')
    print("Exploration visualization saved as 'output/03_Fire_Analysis/firms_data_exploration.png'")
    
    return fig

def main():
    """Main exploration function"""
    print("FIRMS Data Detailed Exploration")
    print("="*50)
    
    # Load data from the main archive file
    fire_archive_path = "Json_files/fire_modis_us.json"
    fire_df = load_and_explore_fire_data(fire_archive_path)
    
    print(f"\nDataset Overview:")
    print(f"Total fire detections: {len(fire_df):,}")
    print(f"Columns: {list(fire_df.columns)}")
    
    # Perform analyses
    daily_counts, hourly_counts, daynight_stats = analyze_temporal_patterns(fire_df)
    location_counts, repeated_locations = analyze_spatial_patterns(fire_df)
    conf_by_sat = analyze_detection_quality(fire_df)
    grid_daily = analyze_same_location_patterns(fire_df)
    
    # Create visualizations
    print("\nCreating exploration visualizations...")
    fig = create_exploration_visualizations(fire_df, daily_counts, hourly_counts, location_counts)
    
    # Summary recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS FOR GRIDDING APPROACH")
    print("="*50)
    
    print("1. Grid Resolution:")
    print("   - 1km grid (0.01° precision) seems appropriate")
    print(f"   - Reduces {len(fire_df):,} points to ~{len(fire_df.groupby(['lat_rounded_01', 'lon_rounded_01']))} grid cells")
    
    print("\n2. Temporal Aggregation:")
    print("   - Daily aggregation recommended")
    print(f"   - {len(daynight_stats)} unique dates in dataset")
    print("   - Handle day/night detections appropriately")
    
    print("\n3. Quality Filtering:")
    print("   - Consider confidence threshold (≥70% recommended)")
    print("   - Multiple satellite detections provide validation")
    
    print("\n4. Clustering Strategy:")
    print("   - Grid-based approach with daily aggregation")
    print("   - Additional spatial clustering for connected grid cells")
    print("   - Aggregate fire properties (max FRP, mean confidence)")
    
    print("\nExploration complete!")

if __name__ == "__main__":
    main()
