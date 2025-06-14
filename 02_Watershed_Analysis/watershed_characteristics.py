#!/usr/bin/env python3
"""
Fixed Watershed Characteristics Analysis Script
Using EPSG:6933 (World Cylindrical Equal Area) projection that actually works
"""
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'  # 0 removes size limit

def load_watershed_data(file_path):
    """Load and prepare watershed data with working projection"""
    print(f"Loading watershed data from: {file_path}")
    
    watershed_gdf = gpd.read_file(file_path)
    print(f"Original CRS: {watershed_gdf.crs}")
    print(f"Sample geometry types: {watershed_gdf.geometry.geom_type.value_counts().head()}")
    
    # Check for valid geometries
    valid_geom = watershed_gdf.geometry.is_valid
    print(f"Valid geometries: {valid_geom.sum()} out of {len(watershed_gdf)}")
    
    # Use EPSG:6933 (World Cylindrical Equal Area) - the working projection
    print("Reprojecting to EPSG:6933 (World Cylindrical Equal Area) for area calculations...")
    watershed_gdf_proj = watershed_gdf.to_crs('EPSG:6933')
    print(f"Projected CRS: {watershed_gdf_proj.crs}")
    
    # Calculate area and perimeter on projected data
    areas_m2 = watershed_gdf_proj.geometry.area
    perimeters_m = watershed_gdf_proj.geometry.length
    
    print(f"Raw areas (m²) - min: {areas_m2.min():.0f}, max: {areas_m2.max():.0f}, mean: {areas_m2.mean():.0f}")
    print(f"Raw perimeters (m) - min: {perimeters_m.min():.0f}, max: {perimeters_m.max():.0f}, mean: {perimeters_m.mean():.0f}")
    
    watershed_gdf['area_km2'] = areas_m2 / 1e6  # Convert to km²
    watershed_gdf['perimeter_km'] = perimeters_m / 1000  # Convert to km
    
    print(f"Calculated areas (km²) - min: {watershed_gdf['area_km2'].min():.2f}, max: {watershed_gdf['area_km2'].max():.2f}")
    print(f"Calculated perimeters (km) - min: {watershed_gdf['perimeter_km'].min():.2f}, max: {watershed_gdf['perimeter_km'].max():.2f}")
    
    # Check for zero/negative values before filtering
    zero_area = (watershed_gdf['area_km2'] <= 0).sum()
    zero_perimeter = (watershed_gdf['perimeter_km'] <= 0).sum()
    print(f"Watersheds with zero/negative area: {zero_area}")
    print(f"Watersheds with zero/negative perimeter: {zero_perimeter}")
    
    # Filter out invalid geometries (zero or negative areas/perimeters)
    print(f"Before filtering: {len(watershed_gdf)} watersheds")
    valid_mask = (watershed_gdf['area_km2'] > 0) & (watershed_gdf['perimeter_km'] > 0)
    print(f"Valid mask sum: {valid_mask.sum()}")
    
    watershed_gdf_filtered = watershed_gdf[valid_mask].copy()
    print(f"After filtering invalid geometries: {len(watershed_gdf_filtered)} watersheds")
    
    # Calculate shape complexity metrics
    watershed_gdf_filtered['compactness'] = 4 * np.pi * watershed_gdf_filtered['area_km2'] / (watershed_gdf_filtered['perimeter_km'] ** 2)
    watershed_gdf_filtered['shape_index'] = watershed_gdf_filtered['perimeter_km'] / (2 * np.sqrt(np.pi * watershed_gdf_filtered['area_km2']))
    
    # Filter out any remaining invalid values
    finite_mask = (np.isfinite(watershed_gdf_filtered['compactness']) & 
                   np.isfinite(watershed_gdf_filtered['shape_index']) &
                   (watershed_gdf_filtered['compactness'] > 0) &
                   (watershed_gdf_filtered['shape_index'] > 0))
    
    print(f"Finite values mask sum: {finite_mask.sum()}")
    watershed_gdf_filtered = watershed_gdf_filtered[finite_mask].copy()
    print(f"After filtering infinite/zero values: {len(watershed_gdf_filtered)} watersheds")
    
    # Get centroid coordinates
    centroids = watershed_gdf_filtered.geometry.centroid
    watershed_gdf_filtered['centroid_lon'] = centroids.x
    watershed_gdf_filtered['centroid_lat'] = centroids.y
    
    return watershed_gdf_filtered

def analyze_watershed_properties(watershed_gdf):
    """Analyze watershed geometric and spatial properties"""
    
    stats = {
        'total_watersheds': len(watershed_gdf),
        'area_stats': watershed_gdf['area_km2'].describe().to_dict(),
        'perimeter_stats': watershed_gdf['perimeter_km'].describe().to_dict(),
        'compactness_stats': watershed_gdf['compactness'].describe().to_dict(),
        'shape_index_stats': watershed_gdf['shape_index'].describe().to_dict(),
        'geographic_extent': {
            'min_lon': watershed_gdf['centroid_lon'].min(),
            'max_lon': watershed_gdf['centroid_lon'].max(),
            'min_lat': watershed_gdf['centroid_lat'].min(),
            'max_lat': watershed_gdf['centroid_lat'].max()
        }
    }
    
    return stats

def create_watershed_visualizations(watershed_gdf, stats):
    """Create comprehensive watershed analysis visualizations"""
    
    # Set style for research paper quality
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Watershed Area Distribution
    ax1 = plt.subplot(3, 4, 1)
    watershed_gdf['area_km2'].hist(bins=50, ax=ax1, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Watershed Area Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Area (km²)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(watershed_gdf['area_km2'].median(), color='red', linestyle='--', 
               label=f'Median: {watershed_gdf["area_km2"].median():.1f} km²')
    ax1.legend()
    
    # 2. Log-scale Area Distribution
    ax2 = plt.subplot(3, 4, 2)
    positive_areas = watershed_gdf['area_km2'][watershed_gdf['area_km2'] > 0]
    log_areas = np.log10(positive_areas)
    log_areas.hist(bins=50, ax=ax2, color='darkgreen', alpha=0.7, edgecolor='black')
    ax2.set_title('Log-Scale Area Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Log₁₀(Area) (km²)')
    ax2.set_ylabel('Frequency')
    
    # 3. Perimeter vs Area Relationship
    ax3 = plt.subplot(3, 4, 3)
    scatter = ax3.scatter(watershed_gdf['area_km2'], watershed_gdf['perimeter_km'], 
                         alpha=0.6, c=watershed_gdf['compactness'], cmap='plasma', s=2)
    ax3.set_title('Perimeter vs Area Relationship', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Area (km²)')
    ax3.set_ylabel('Perimeter (km)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Compactness')
    
    # 4. Shape Complexity (Compactness) Distribution
    ax4 = plt.subplot(3, 4, 4)
    watershed_gdf['compactness'].hist(bins=50, ax=ax4, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_title('Watershed Compactness Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Compactness Index')
    ax4.set_ylabel('Frequency')
    ax4.axvline(1.0, color='red', linestyle='--', label='Perfect Circle (1.0)')
    ax4.legend()
    
    # 5. Geographic Distribution of Watersheds
    ax5 = plt.subplot(3, 4, 5)
    try:
        watershed_gdf.plot(ax=ax5, column='area_km2', cmap='viridis', legend=True, 
                           legend_kwds={'shrink': 0.8, 'label': 'Area (km²)'})
        ax5.set_title('Watershed Areas - Geographic Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Error plotting map: {str(e)[:50]}...', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Geographic Distribution (Error)', fontsize=12, fontweight='bold')
    
    # 6. Shape Index Distribution
    ax6 = plt.subplot(3, 4, 6)
    watershed_gdf['shape_index'].hist(bins=50, ax=ax6, color='purple', alpha=0.7, edgecolor='black')
    ax6.set_title('Shape Index Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Shape Index')
    ax6.set_ylabel('Frequency')
    ax6.axvline(1.0, color='red', linestyle='--', label='Circle (1.0)')
    ax6.legend()
    
    # 7. Size Categories
    ax7 = plt.subplot(3, 4, 7)
    
    # Define size categories
    def categorize_size(area):
        if area < 10:
            return 'Very Small (<10 km²)'
        elif area < 50:
            return 'Small (10-50 km²)'
        elif area < 200:
            return 'Medium (50-200 km²)'
        elif area < 500:
            return 'Large (200-500 km²)'
        else:
            return 'Very Large (>500 km²)'
    
    watershed_gdf['size_category'] = watershed_gdf['area_km2'].apply(categorize_size)
    size_counts = watershed_gdf['size_category'].value_counts()
    
    colors = ['lightblue', 'skyblue', 'steelblue', 'navy', 'darkblue']
    wedges, texts, autotexts = ax7.pie(size_counts.values, labels=size_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax7.set_title('Watershed Size Categories', fontsize=12, fontweight='bold')
    
    # 8. Compactness vs Size
    ax8 = plt.subplot(3, 4, 8)
    ax8.scatter(watershed_gdf['area_km2'], watershed_gdf['compactness'], 
               alpha=0.6, color='green', s=1)
    ax8.set_title('Compactness vs Watershed Size', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Area (km²)')
    ax8.set_ylabel('Compactness')
    ax8.set_xscale('log')
    
    # Calculate correlation
    correlation = watershed_gdf['area_km2'].corr(watershed_gdf['compactness'])
    ax8.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax8.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 9. Latitudinal Distribution
    ax9 = plt.subplot(3, 4, 9)
    lat_normalized = (watershed_gdf['centroid_lat'] - watershed_gdf['centroid_lat'].min()) / \
                    (watershed_gdf['centroid_lat'].max() - watershed_gdf['centroid_lat'].min())
    ax9.scatter(watershed_gdf['centroid_lon'], watershed_gdf['centroid_lat'], 
               c=lat_normalized, cmap='terrain', s=1, alpha=0.7)
    ax9.set_title('Latitudinal Distribution', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Longitude')
    ax9.set_ylabel('Latitude')
    
    # 10. Box Plot of Area by Region
    ax10 = plt.subplot(3, 4, 10)
    
    # Create regional categories based on longitude
    def categorize_region(lon):
        if lon < -120:
            return 'West Coast'
        elif lon < -110:
            return 'Mountain West'
        elif lon < -100:
            return 'Central West'
        else:
            return 'Great Plains'
    
    watershed_gdf['region'] = watershed_gdf['centroid_lon'].apply(categorize_region)
    
    # Create box plot
    regions = watershed_gdf['region'].unique()
    area_data = []
    region_labels = []
    
    for region in regions:
        region_areas = watershed_gdf[watershed_gdf['region'] == region]['area_km2']
        area_data.append(region_areas.values)
        region_labels.append(region)
    
    box_plot = ax10.boxplot(area_data, labels=region_labels, patch_artist=True)
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax10.set_title('Area Distribution by Region', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Area (km²)')
    ax10.set_yscale('log')
    plt.setp(ax10.get_xticklabels(), rotation=45)
    
    # 11. Summary Statistics Table
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    summary_text = f"""
    Watershed Summary Statistics:
    
    Total Watersheds: {stats['total_watersheds']:,}
    
    Area Statistics (km²):
    • Mean: {stats['area_stats']['mean']:.1f}
    • Median: {stats['area_stats']['50%']:.1f}
    • Min: {stats['area_stats']['min']:.1f}
    • Max: {stats['area_stats']['max']:.1f}
    • Std: {stats['area_stats']['std']:.1f}
    
    Compactness Statistics:
    • Mean: {stats['compactness_stats']['mean']:.3f}
    • Median: {stats['compactness_stats']['50%']:.3f}
    • Min: {stats['compactness_stats']['min']:.3f}
    • Max: {stats['compactness_stats']['max']:.3f}
    
    Geographic Extent:
    • Longitude: {stats['geographic_extent']['min_lon']:.2f}° to {stats['geographic_extent']['max_lon']:.2f}°
    • Latitude: {stats['geographic_extent']['min_lat']:.2f}° to {stats['geographic_extent']['max_lat']:.2f}°
    """
    
    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 12. Perimeter Distribution
    ax12 = plt.subplot(3, 4, 12)
    watershed_gdf['perimeter_km'].hist(bins=50, ax=ax12, color='coral', alpha=0.7, edgecolor='black')
    ax12.set_title('Perimeter Distribution', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Perimeter (km)')
    ax12.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Create organized output directory
    os.makedirs('output/02_Watershed_Analysis', exist_ok=True)
    
    plt.savefig('output/02_Watershed_Analysis/watershed_characteristics_analysis_fixed.png', dpi=300, bbox_inches='tight')
    print("Watershed analysis saved as 'output/02_Watershed_Analysis/watershed_characteristics_analysis_fixed.png'")
    
    return fig

def write_watershed_report(watershed_gdf, stats, output_file='output/02_Watershed_Analysis/watershed_characteristics_report_fixed.txt'):
    """Write comprehensive watershed analysis report"""
    
    # Create organized output directory
    os.makedirs('output/02_Watershed_Analysis', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WATERSHED CHARACTERISTICS ANALYSIS REPORT (FIXED)\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Projection used: EPSG:6933 (World Cylindrical Equal Area)\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-"*40 + "\n")
        f.write(f"Total number of HUC12 watersheds: {stats['total_watersheds']:,}\n")
        f.write(f"Data source: Western United States HUC12 watersheds\n")
        f.write(f"Original CRS: EPSG:4326 (WGS84)\n")
        f.write(f"Analysis CRS: EPSG:6933 (World Cylindrical Equal Area)\n\n")
        
        f.write("GEOMETRIC CHARACTERISTICS\n")
        f.write("-"*40 + "\n")
        f.write("Area Statistics (km²):\n")
        for key, value in stats['area_stats'].items():
            f.write(f"  {key}: {value:.2f}\n")
        f.write("\n")
        
        f.write("Perimeter Statistics (km):\n")
        for key, value in stats['perimeter_stats'].items():
            f.write(f"  {key}: {value:.2f}\n")
        f.write("\n")
        
        f.write("SHAPE COMPLEXITY ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write("Compactness Index Statistics:\n")
        f.write("(Values closer to 1.0 indicate more circular/compact shapes)\n")
        for key, value in stats['compactness_stats'].items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Shape Index Statistics:\n")
        f.write("(Values closer to 1.0 indicate more circular shapes)\n")
        for key, value in stats['shape_index_stats'].items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")
        
        # Size categories analysis
        size_categories = watershed_gdf['area_km2'].apply(lambda x: 
            'Very Small (<10 km²)' if x < 10 else
            'Small (10-50 km²)' if x < 50 else
            'Medium (50-200 km²)' if x < 200 else
            'Large (200-500 km²)' if x < 500 else
            'Very Large (>500 km²)'
        ).value_counts()
        
        f.write("SIZE DISTRIBUTION ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write("Watershed Count by Size Category:\n")
        for category, count in size_categories.items():
            percentage = (count / len(watershed_gdf)) * 100
            f.write(f"  {category}: {count:,} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Regional analysis
        if 'region' in watershed_gdf.columns:
            regional_stats = watershed_gdf.groupby('region')['area_km2'].agg(['count', 'mean', 'median', 'std'])
            
            f.write("REGIONAL ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write("Watershed characteristics by region:\n")
            for region in regional_stats.index:
                count = regional_stats.loc[region, 'count']
                mean_area = regional_stats.loc[region, 'mean']
                median_area = regional_stats.loc[region, 'median']
                std_area = regional_stats.loc[region, 'std']
                f.write(f"\n{region}:\n")
                f.write(f"  Count: {count:,} watersheds\n")
                f.write(f"  Mean area: {mean_area:.1f} km²\n")
                f.write(f"  Median area: {median_area:.1f} km²\n")
                f.write(f"  Area std dev: {std_area:.1f} km²\n")
        
        f.write("\n\nKEY FINDINGS\n")
        f.write("-"*40 + "\n")
        f.write("1. Watershed Size Diversity:\n")
        f.write(f"   - Range spans from {stats['area_stats']['min']:.1f} to {stats['area_stats']['max']:.1f} km²\n")
        f.write(f"   - Coefficient of variation: {(stats['area_stats']['std']/stats['area_stats']['mean']):.2f}\n")
        f.write("   - High variability indicates diverse hydrological scales\n\n")
        
        f.write("2. Shape Complexity:\n")
        mean_compactness = stats['compactness_stats']['mean']
        if mean_compactness > 0.5:
            shape_desc = "relatively compact"
        elif mean_compactness > 0.3:
            shape_desc = "moderately irregular"
        else:
            shape_desc = "highly irregular"
        f.write(f"   - Average compactness: {mean_compactness:.3f} ({shape_desc})\n")
        f.write("   - Lower values suggest more dendritic/elongated watersheds\n\n")
        
        f.write("TECHNICAL NOTES\n")
        f.write("-"*40 + "\n")
        f.write("This analysis used EPSG:6933 (World Cylindrical Equal Area) projection\n")
        f.write("instead of the originally intended EPSG:5070 due to coordinate\n")
        f.write("transformation issues. EPSG:6933 is an equal-area projection\n")
        f.write("suitable for accurate area calculations across the study region.\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF WATERSHED CHARACTERISTICS REPORT\n")
        f.write("="*80 + "\n")

def main():
    """Main watershed analysis function"""
    print("Starting Fixed Watershed Characteristics Analysis...")
    print("="*60)
    
    # File path
    watershed_path = "Json_files/huc12_conus.geojson"
    
    try:
        # Load and prepare data
        watershed_gdf = load_watershed_data(watershed_path)
        print(f"✓ Loaded {len(watershed_gdf):,} watersheds")
        
        # Analyze properties
        stats = analyze_watershed_properties(watershed_gdf)
        print("✓ Calculated watershed geometric properties")
        
        # Create visualizations
        print("Creating watershed characteristics visualizations...")
        fig = create_watershed_visualizations(watershed_gdf, stats)
        
        # Write report
        print("Writing watershed characteristics report...")
        write_watershed_report(watershed_gdf, stats)
        print("✓ Report saved")
        
        # Display summary
        print("\n" + "="*60)
        print("WATERSHED ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total watersheds analyzed: {stats['total_watersheds']:,}")
        print(f"Area range: {stats['area_stats']['min']:.1f} - {stats['area_stats']['max']:.1f} km²")
        print(f"Mean area: {stats['area_stats']['mean']:.1f} km²")
        print(f"Mean compactness: {stats['compactness_stats']['mean']:.3f}")
        print("\nFiles generated:")
        print("  • output/02_Watershed_Analysis/watershed_characteristics_analysis.png")
        print("  • output/02_Watershed_Analysis/watershed_characteristics_report.txt")
        print("\n✓ Watershed analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check file paths and data integrity.")

if __name__ == "__main__":
    main()