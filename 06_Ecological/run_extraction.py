#!/usr/bin/env python3
"""
Ecological Data Extraction for Fire Regime Analysis
Extracts ecological variables using Google Earth Engine

Usage:
    # Basic extraction with default parameters
    python run_extraction.py
    
    # With custom parameters
    python run_extraction.py --batch-size 25 --n-jobs 5

This script will:
1. Load watershed data from available sources
2. Extract ecological variables from Google Earth Engine
3. Merge ecological data with fire metrics
4. Export enhanced dataset
5. Generate summary statistics

Output files are saved to: 06_Ecological_Context/
"""

import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Import our ecological extraction modules
from extract_ecological_data import EcologicalDataExtractor

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(title)
    print("="*60)

def load_watershed_data():
    """
    Load watershed data from available sources
    """
    print_header("LOADING WATERSHED DATA")
    
    # Try different possible input files
    possible_files = [
        Path('outputs_fixed') / 'watersheds_with_fire_metrics.gpkg',
        Path('05_Clustering_Fast') / 'clustered_watersheds.parquet',
        Path('04_Integration_partb') / 'watersheds_with_fire_metrics.gpkg'
    ]
    
    # Find most recent analysis directory if default doesn't exist
    outputs_dir = Path('outputs_fixed')
    if outputs_dir.exists():
        analysis_dirs = sorted([d for d in outputs_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('analysis_run_')])
        if analysis_dirs:
            possible_files.insert(0, analysis_dirs[-1] / 'watersheds_with_fire_metrics.gpkg')
    
    watersheds_gdf = None
    input_file = None
    
    for file_path in possible_files:
        if file_path.exists():
            input_file = file_path
            print(f"Found input file: {input_file}")
            
            try:
                if file_path.suffix == '.parquet':
                    watersheds_gdf = pd.read_parquet(file_path)
                    # Convert to GeoDataFrame if geometry column exists
                    if 'geometry' in watersheds_gdf.columns:
                        watersheds_gdf = gpd.GeoDataFrame(watersheds_gdf)
                else:
                    watersheds_gdf = gpd.read_file(file_path)
                
                print(f"✓ Successfully loaded {len(watersheds_gdf):,} watersheds")
                break
                
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                continue
    
    if watersheds_gdf is None:
        raise FileNotFoundError("Could not find any valid watershed data file")
    
    return watersheds_gdf

def extract_ecological_context(watersheds_gdf, batch_size=50, n_jobs=10):
    """
    Extract ecological context variables using Google Earth Engine
    
    Args:
        watersheds_gdf: GeoDataFrame with watershed boundaries
        batch_size: Batch size for GEE extraction
        n_jobs: Number of parallel jobs for extraction
    """
    print_header("ECOLOGICAL CONTEXT EXTRACTION")
    
    print("Using Google Earth Engine for ecological data extraction")
    print("⚠️  This will take significant time for large datasets")
    
    # Initialize GEE extractor
    extractor = EcologicalDataExtractor(watersheds_gdf)
    
    # Prepare datasets
    extractor.prepare_datasets()
    
    # Extract in batches
    ecological_df = extractor.batch_extract_parallel(
        batch_size=batch_size,
        n_jobs=n_jobs
    )
    
    # Merge with fire data and export
    enhanced_watersheds = extractor.merge_with_fire_data(ecological_df)
    output_file = extractor.export_enhanced_data(enhanced_watersheds)
    
    print(f"\n✓ Enhanced dataset shape: {enhanced_watersheds.shape}")
    print(f"✓ Ecological variables: {len(ecological_df.columns)}")
    
    return enhanced_watersheds, ecological_df, output_file

def create_summary_statistics(enhanced_watersheds, ecological_df, output_dir):
    """
    Create summary statistics for the extracted ecological data
    """
    print_header("CREATING SUMMARY STATISTICS")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic statistics
    eco_stats = ecological_df.describe()
    eco_stats.to_csv(output_dir / 'ecological_variables_summary.csv')
    
    # Correlation matrix
    correlation_matrix = ecological_df.corr()
    correlation_matrix.to_csv(output_dir / 'ecological_correlation_matrix.csv')
    
    # Data completeness report
    completeness = pd.DataFrame({
        'Variable': ecological_df.columns,
        'Non_null_count': ecological_df.count(),
        'Null_count': ecological_df.isnull().sum(),
        'Completeness_%': (ecological_df.count() / len(ecological_df) * 100).round(2)
    })
    completeness.to_csv(output_dir / 'data_completeness_report.csv', index=False)
    
    print(f"\n✓ Summary statistics:")
    print(f"  - Variable summary: {output_dir / 'ecological_variables_summary.csv'}")
    print(f"  - Correlation matrix: {output_dir / 'ecological_correlation_matrix.csv'}")
    print(f"  - Completeness report: {output_dir / 'data_completeness_report.csv'}")
    
    # Print key statistics
    print(f"\n📊 Data Overview:")
    print(f"  Total watersheds: {len(enhanced_watersheds):,}")
    print(f"  Ecological variables: {len(ecological_df.columns)}")
    print(f"  Average completeness: {completeness['Completeness_%'].mean():.1f}%")
    
    if 'episode_count' in enhanced_watersheds.columns:
        fire_affected = (enhanced_watersheds['episode_count'] > 0).sum()
        print(f"  Fire-affected watersheds: {fire_affected:,} ({fire_affected/len(enhanced_watersheds)*100:.1f}%)")
    
    return output_dir

def main(batch_size=50, n_jobs=10):
    """
    Main execution function for ecological data extraction using Google Earth Engine
    
    Args:
        batch_size: Batch size for GEE extraction
        n_jobs: Number of parallel jobs for extraction
    """
    print("="*60)
    print("ECOLOGICAL DATA EXTRACTION FOR FIRE REGIME ANALYSIS")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mode: Google Earth Engine")
    
    try:
        # Step 1: Load watershed data
        watersheds_gdf = load_watershed_data()
        
        # Step 2: Extract ecological context
        enhanced_watersheds, ecological_df, output_file = extract_ecological_context(
            watersheds_gdf,
            batch_size=batch_size,
            n_jobs=n_jobs
        )
        
        # Step 3: Create summary statistics
        stats_dir = create_summary_statistics(
            enhanced_watersheds,
            ecological_df,
            output_file.parent
        )
        
        # Final summary
        print_header("EXTRACTION COMPLETE")
        print(f"✓ Successfully extracted ecological data for {len(enhanced_watersheds):,} watersheds")
        print(f"✓ {len(ecological_df.columns)} ecological variables extracted")
        print(f"✓ Enhanced dataset saved to: {output_file}")
        print(f"✓ Summary statistics saved to: {stats_dir}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return enhanced_watersheds, ecological_df, output_file
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        print("Please check your input data and try again.")
        return None, None, None

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract ecological variables for fire regime analysis using Google Earth Engine"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for GEE extraction (default: 50)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=10,
        help='Number of parallel jobs (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Set process priority for better performance
    try:
        import os
        os.nice(-5)  # Increase priority slightly
    except:
        pass
    
    # Run the ecological data extraction
    results = main(
        batch_size=args.batch_size,
        n_jobs=args.n_jobs
    )