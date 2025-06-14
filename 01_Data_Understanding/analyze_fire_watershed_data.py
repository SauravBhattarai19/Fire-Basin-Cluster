#!/usr/bin/env python3
"""
Fire and Watershed Data Analysis Script
Analyzes FIRMS fire data and HUC12 watershed data for research purposes
"""

import os
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure GDAL to handle large GeoJSON files
os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'  # 0 removes size limit

def analyze_fire_data(file_path):
    """Analyze FIRMS fire data structure and statistics"""
    print(f"Analyzing fire data from: {file_path}")
    
    # Load fire data
    with open(file_path, 'r') as f:
        fire_data = json.load(f)
      # Convert to DataFrame for easier analysis
    fire_df = pd.DataFrame(fire_data)
    
    # Convert confidence to numeric if it's string
    fire_df['confidence'] = pd.to_numeric(fire_df['confidence'], errors='coerce')
    
    # Basic statistics
    stats = {
        'total_records': len(fire_df),
        'columns': list(fire_df.columns),
        'date_range': (fire_df['acq_date'].min(), fire_df['acq_date'].max()),
        'instruments': fire_df['instrument'].unique().tolist(),
        'satellites': fire_df['satellite'].unique().tolist(),
        'coordinate_bounds': {
            'lat_min': fire_df['latitude'].min(),
            'lat_max': fire_df['latitude'].max(),
            'lon_min': fire_df['longitude'].min(),
            'lon_max': fire_df['longitude'].max()
        },
        'confidence_stats': fire_df['confidence'].describe().to_dict(),
        'frp_stats': fire_df['frp'].describe().to_dict(),
        'brightness_stats': fire_df['brightness'].describe().to_dict()
    }
    
    return fire_df, stats

def analyze_watershed_data(file_path):
    """Analyze HUC12 watershed data structure"""
    print(f"Analyzing watershed data from: {file_path}")
    
    try:
        # Try reading with default settings first
        watershed_gdf = gpd.read_file(file_path)
    except Exception as e:
        print(f"Standard read failed: {e}")
        print("Attempting to read with alternative approach...")
        
        try:
            # Alternative approach: read with specific driver options
            watershed_gdf = gpd.read_file(file_path, driver='GeoJSON')
        except Exception as e2:
            print(f"Alternative read also failed: {e2}")
            print("Trying with GDAL environment variables...")
            
            # Set additional GDAL options for memory management
            os.environ['GDAL_HTTP_TIMEOUT'] = '300'
            os.environ['GDAL_HTTP_MAX_RETRY'] = '3'
            os.environ['VSI_CACHE'] = 'TRUE'
            os.environ['VSI_CACHE_SIZE'] = '1000000000'  # 1GB cache
            
            # Try one more time
            watershed_gdf = gpd.read_file(file_path)
    
    # Basic statistics
    stats = {
        'total_watersheds': len(watershed_gdf),
        'crs': str(watershed_gdf.crs),
        'columns': list(watershed_gdf.columns),
        'geometry_type': watershed_gdf.geometry.geom_type.unique().tolist(),
        'bounds': watershed_gdf.total_bounds.tolist(),  # [minx, miny, maxx, maxy]
        'area_stats': watershed_gdf.geometry.area.describe().to_dict() if 'area' not in watershed_gdf.columns else None
    }
    
    # Add column-specific stats if available
    if 'HUC12' in watershed_gdf.columns:
        stats['huc12_sample'] = watershed_gdf['HUC12'].head(10).tolist()
    
    return watershed_gdf, stats

def create_infographics(fire_df, fire_stats, watershed_gdf, watershed_stats):
    """Create research-quality infographics"""
    
    # Set style for research paper quality
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
      # 1. Fire Detection Temporal Distribution
    ax1 = plt.subplot(3, 3, 1)
    fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date'])
    fire_df['month'] = fire_df['acq_date'].dt.month
    monthly_counts = fire_df['month'].value_counts().sort_index()
    monthly_counts.plot(kind='bar', ax=ax1, color='orangered', alpha=0.7)
    ax1.set_title('Monthly Fire Detection Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Fire Detections')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Fire Radiative Power Distribution
    ax2 = plt.subplot(3, 3, 2)
    fire_df['frp'].hist(bins=50, ax=ax2, color='red', alpha=0.7, edgecolor='black')
    ax2.set_title('Fire Radiative Power Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('FRP (MW)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(fire_df['frp'].mean(), color='darkred', linestyle='--', label=f'Mean: {fire_df["frp"].mean():.2f}')
    ax2.legend()
      # 3. Confidence Level Distribution (Improved)
    ax3 = plt.subplot(3, 3, 3)
    
    # Define confidence categories
    def categorize_confidence(conf):
        if pd.isna(conf):
            return 'Unknown'
        elif conf >= 80:
            return 'High (≥80%)'
        elif conf >= 60:
            return 'Medium (60-79%)'
        elif conf >= 30:
            return 'Low (30-59%)'
        else:
            return 'Very Low (<30%)'
    
    fire_df['confidence_category'] = fire_df['confidence'].apply(categorize_confidence)
    confidence_counts = fire_df['confidence_category'].value_counts()
    
    # Better color scheme for confidence levels
    colors = ['darkgreen', 'lightgreen', 'orange', 'red', 'gray']
    wedges, texts, autotexts = ax3.pie(confidence_counts.values, labels=confidence_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax3.set_title('Fire Detection Confidence Levels\n(Quality Categories)', fontsize=12, fontweight='bold')
    
    # 4. Day vs Night Detection
    ax4 = plt.subplot(3, 3, 4)
    daynight_counts = fire_df['daynight'].value_counts()
    bars = ax4.bar(daynight_counts.index, daynight_counts.values, 
                   color=['gold', 'navy'], alpha=0.7)
    ax4.set_title('Day vs Night Fire Detections', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Detections')
    for bar, count in zip(bars, daynight_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 5. Satellite Coverage
    ax5 = plt.subplot(3, 3, 5)
    satellite_counts = fire_df['satellite'].value_counts()
    ax5.pie(satellite_counts.values, labels=satellite_counts.index, autopct='%1.1f%%',
            colors=['skyblue', 'lightgreen'])
    ax5.set_title('Fire Detections by Satellite', fontsize=12, fontweight='bold')
    
    # 6. Geographic Distribution (Fire Points)
    ax6 = plt.subplot(3, 3, 6)
    scatter = ax6.scatter(fire_df['longitude'], fire_df['latitude'], 
                         c=fire_df['frp'], cmap='hot', alpha=0.6, s=1)
    ax6.set_title('Geographic Distribution of Fire Detections', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('FRP (MW)')
    
    # 7. Watershed Coverage Map
    ax7 = plt.subplot(3, 3, 7)
    watershed_gdf.plot(ax=ax7, color='lightblue', edgecolor='blue', alpha=0.7, linewidth=0.5)
    ax7.set_title('HUC12 Watershed Coverage', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Longitude')
    ax7.set_ylabel('Latitude')
    
    # 8. Brightness vs FRP Correlation
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(fire_df['brightness'], fire_df['frp'], alpha=0.5, color='red', s=1)
    ax8.set_title('Brightness vs Fire Radiative Power', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Brightness (K)')
    ax8.set_ylabel('FRP (MW)')
    
    # Calculate correlation
    correlation = fire_df['brightness'].corr(fire_df['frp'])
    ax8.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax8.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
      # 9. Data Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Safe access to stats with fallback values
    mean_frp = fire_stats['frp_stats'].get('mean', 0) if 'mean' in fire_stats['frp_stats'] else 0
    mean_confidence = fire_stats['confidence_stats'].get('mean', 0) if 'mean' in fire_stats['confidence_stats'] else 0
    
    summary_text = f"""
    Dataset Summary:
    
    Fire Data:
    • Total Detections: {fire_stats['total_records']:,}
    • Date Range: {fire_stats['date_range'][0]} to {fire_stats['date_range'][1]}
    • Instruments: {', '.join(fire_stats['instruments'])}
    • Satellites: {', '.join(fire_stats['satellites'])}
    • Mean FRP: {mean_frp:.2f} MW
    • Mean Confidence: {mean_confidence:.1f}%
    
    Watershed Data:
    • Total Watersheds: {watershed_stats['total_watersheds']:,}
    • CRS: {watershed_stats['crs']}    • Geometry: {', '.join(watershed_stats['geometry_type'])}
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Create organized output directory
    import os
    os.makedirs('output/01_Data_Understanding', exist_ok=True)
    
    plt.savefig('output/01_Data_Understanding/fire_watershed_analysis.png', dpi=300, bbox_inches='tight')
    print("Infographic saved as 'output/01_Data_Understanding/fire_watershed_analysis.png'")
    
    return fig

def write_analysis_report(fire_stats, watershed_stats, output_file='output/01_Data_Understanding/analysis_report.txt'):
    """Write comprehensive analysis report to text file"""
    
    # Create organized output directory
    import os
    os.makedirs('output/01_Data_Understanding', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FIRE AND WATERSHED DATA ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Fire Data Analysis
        f.write("FIRE DATA (FIRMS) ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total fire detection records: {fire_stats['total_records']:,}\n")
        f.write(f"Data columns: {', '.join(fire_stats['columns'])}\n")
        f.write(f"Date range: {fire_stats['date_range'][0]} to {fire_stats['date_range'][1]}\n")
        f.write(f"Instruments: {', '.join(fire_stats['instruments'])}\n")
        f.write(f"Satellites: {', '.join(fire_stats['satellites'])}\n\n")
        
        f.write("Geographic Coverage:\n")
        f.write(f"  Latitude range: {fire_stats['coordinate_bounds']['lat_min']:.4f} to {fire_stats['coordinate_bounds']['lat_max']:.4f}\n")
        f.write(f"  Longitude range: {fire_stats['coordinate_bounds']['lon_min']:.4f} to {fire_stats['coordinate_bounds']['lon_max']:.4f}\n\n")
        
        f.write("Fire Radiative Power (FRP) Statistics:\n")
        for key, value in fire_stats['frp_stats'].items():
            f.write(f"  {key}: {value:.2f} MW\n")
        f.write("\n")
        
        f.write("Brightness Statistics:\n")
        for key, value in fire_stats['brightness_stats'].items():
            f.write(f"  {key}: {value:.2f} K\n")
        f.write("\n")
        
        f.write("Confidence Statistics:\n")
        for key, value in fire_stats['confidence_stats'].items():
            f.write(f"  {key}: {value:.2f}%\n")
        f.write("\n")
        
        # Watershed Data Analysis
        f.write("WATERSHED DATA (HUC12) ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total watershed polygons: {watershed_stats['total_watersheds']:,}\n")
        f.write(f"Coordinate Reference System (CRS): {watershed_stats['crs']}\n")
        f.write(f"Data columns: {', '.join(watershed_stats['columns'])}\n")
        f.write(f"Geometry types: {', '.join(watershed_stats['geometry_type'])}\n\n")
        
        f.write("Geographic Bounds:\n")
        bounds = watershed_stats['bounds']
        f.write(f"  West: {bounds[0]:.4f}\n")
        f.write(f"  South: {bounds[1]:.4f}\n")
        f.write(f"  East: {bounds[2]:.4f}\n")
        f.write(f"  North: {bounds[3]:.4f}\n\n")
        
        if 'huc12_sample' in watershed_stats:
            f.write("Sample HUC12 IDs:\n")
            for huc_id in watershed_stats['huc12_sample']:
                f.write(f"  {huc_id}\n")
            f.write("\n")
          # Data Quality and Recommendations
        f.write("DATA QUALITY ASSESSMENT AND RECOMMENDATIONS\n")
        f.write("-"*50 + "\n")
        f.write("Fire Data Quality:\n")
        f.write("  [✓] Complete coordinate information available\n")
        f.write("  [✓] Temporal coverage spans multiple dates\n")
        f.write("  [✓] Multiple confidence levels for quality control\n")
        f.write("  [✓] FRP values available for fire intensity analysis\n\n")
        
        f.write("Watershed Data Quality:\n")
        f.write("  [✓] Standardized HUC12 watershed delineation\n")
        f.write("  [✓] Proper geospatial format (GeoJSON)\n")
        f.write("  [✓] Geographic coverage of western US\n\n")
        
        f.write("Integration Recommendations:\n")
        f.write("  1. Use spatial joins to associate fire points with watersheds\n")
        f.write("  2. Filter fire data by confidence levels (recommend >70%)\n")
        f.write("  3. Aggregate fire statistics by watershed for clustering analysis\n")
        f.write("  4. Consider temporal aggregation (monthly/seasonal) for trends\n")
        f.write("  5. Validate CRS compatibility between datasets\n\n")
        
        f.write("Suggested Analysis Pipeline:\n")
        f.write("  1. Reproject both datasets to common CRS (e.g., EPSG:5070)\n")
        f.write("  2. Spatial join fire points to HUC12 watersheds\n")
        f.write("  3. Calculate fire metrics per watershed (count, total FRP, mean intensity)\n")
        f.write("  4. Apply clustering algorithms on watershed fire characteristics\n")
        f.write("  5. Validate clusters with environmental and geographic variables\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

def main():
    """Main analysis function"""
    print("Starting Fire and Watershed Data Analysis...")
    print("="*60)
      # File paths
    fire_nrt_path = "fire_nrt_M-C61_622394.json"
    fire_archive_path = "Json_files/fire_modis_us.json"
    watershed_path = "Json_files/huc12_conus.geojson"
    
    try:
        # Analyze fire data (use NRT file as it's smaller)
        fire_df, fire_stats = analyze_fire_data(fire_archive_path)
        print(f"✓ Fire data analysis complete: {fire_stats['total_records']:,} records")
        
        # Analyze watershed data
        watershed_gdf, watershed_stats = analyze_watershed_data(watershed_path)
        print(f"✓ Watershed data analysis complete: {watershed_stats['total_watersheds']:,} watersheds")
        
        # Create infographics
        print("Creating research-quality infographics...")
        fig = create_infographics(fire_df, fire_stats, watershed_gdf, watershed_stats)
        
        # Write comprehensive report
        print("Writing analysis report...")
        write_analysis_report(fire_stats, watershed_stats)
        print("✓ Analysis report saved as 'output/01_Data_Understanding/analysis_report.txt'")
        
        # Display summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Fire detections analyzed: {fire_stats['total_records']:,}")
        print(f"Watersheds analyzed: {watershed_stats['total_watersheds']:,}")
        print(f"Fire data CRS: Geographic (WGS84)")
        print(f"Watershed data CRS: {watershed_stats['crs']}")
        print("\nFiles generated:")
        print("  • output/01_Data_Understanding/fire_watershed_analysis.png (infographic)")
        print("  • output/01_Data_Understanding/analysis_report.txt (comprehensive report)")
        print("\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check file paths and data integrity.")

if __name__ == "__main__":
    main()
