#!/usr/bin/env python3
"""
Complete Fire-Watershed Analysis Pipeline
Runs the entire analysis from fire detection to watershed clustering
"""

import os
import sys
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
import platform
try:
    import resource
except Exception:
    resource = None

# Import our modules
from fixed_episode_detection import ImprovedFireEpisodeDetector, FireRegimeMetrics
from watershed_fire_clustering import WatershedFireRegimeClustering

# Set environment variable for large GeoJSON files
os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'

class CompleteFireWatershedAnalysis:
    """Complete pipeline for fire-watershed analysis"""
    
    def __init__(self, fire_data_path, watershed_data_path, output_dir='outputs'):
        """
        Initialize analysis pipeline
        
        Args:
            fire_data_path: Path to MODIS fire data JSON
            watershed_data_path: Path to HUC12 watershed GeoJSON
            output_dir: Directory for outputs
        """
        self.fire_path = Path(fire_data_path)
        self.watershed_path = Path(watershed_data_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f'analysis_run_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("COMPLETE FIRE-WATERSHED ANALYSIS PIPELINE")
        print("="*60)
        print(f"Fire data: {self.fire_path}")
        print(f"Watershed data: {self.watershed_path}")
        print(f"Output directory: {self.run_dir}")
        
    def load_and_prepare_fire_data(self, quality_threshold=70, test_mode=False):
        """Load and prepare fire data"""
        print("\n" + "="*60)
        print("STEP 1: LOADING AND PREPARING FIRE DATA")
        print("="*60)
        
        # Load fire data
        print(f"Loading fire data from {self.fire_path}...")
        with open(self.fire_path, 'r') as f:
            fire_data = json.load(f)
        
        fire_df = pd.DataFrame(fire_data)
        del fire_data
        gc.collect()
        print(f"Loaded {len(fire_df):,} fire detections")
        
        # Convert data types (optionally downcast for memory)
        strict_precision = os.environ.get('STRICT_PRECISION', '0') == '1'
        conf_series = pd.to_numeric(fire_df['confidence'], errors='coerce')
        frp_series = pd.to_numeric(fire_df['frp'], errors='coerce')
        lat_series = pd.to_numeric(fire_df['latitude'], errors='coerce')
        lon_series = pd.to_numeric(fire_df['longitude'], errors='coerce')
        if strict_precision:
            fire_df['confidence'] = conf_series
            fire_df['frp'] = frp_series
            fire_df['latitude'] = lat_series
            fire_df['longitude'] = lon_series
        else:
            fire_df['confidence'] = conf_series.astype('float32')
            fire_df['frp'] = frp_series.astype('float32')
            fire_df['latitude'] = lat_series.astype('float32')
            fire_df['longitude'] = lon_series.astype('float32')
        
        # Parse dates
        fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date'])
        fire_df['acq_time_str'] = pd.to_numeric(fire_df['acq_time'], errors='coerce').astype('Int32').astype(str).str.zfill(4)
        fire_df['acq_datetime'] = pd.to_datetime(
            fire_df['acq_date'].astype(str) + ' ' + 
            fire_df['acq_time_str'].str[:2] + ':' + 
            fire_df['acq_time_str'].str[2:]
        )
        # Drop helper column to free memory
        fire_df.drop(columns=['acq_time_str'], inplace=True)
        
        # Quality filtering
        print(f"Applying quality filters (confidence >= {quality_threshold}%)...")
        fire_df = fire_df[fire_df['confidence'] >= quality_threshold]
        fire_df = fire_df[fire_df['frp'] > 0]
        fire_df = fire_df.dropna(subset=['latitude', 'longitude'])
        print(f"After filtering: {len(fire_df):,} detections")
        
        # Test mode - use subset
        if test_mode:
            print("TEST MODE: Using California subset...")
            fire_df = fire_df[(fire_df['longitude'] >= -125) & 
                             (fire_df['longitude'] <= -114) &
                             (fire_df['latitude'] >= 32) & 
                             (fire_df['latitude'] <= 42)]
            print(f"Test subset: {len(fire_df):,} detections")
        
        # Add projected coordinates
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True)
        x_proj, y_proj = transformer.transform(
            fire_df['longitude'].values, fire_df['latitude'].values
        )
        if strict_precision:
            fire_df['x_proj'] = x_proj
            fire_df['y_proj'] = y_proj
        else:
            fire_df['x_proj'] = x_proj.astype('float32')
            fire_df['y_proj'] = y_proj.astype('float32')
        
        self.fire_df = fire_df
        
        # Save prepared data
        fire_df.to_parquet(self.run_dir / 'fire_detections_prepared.parquet')
        gc.collect()
        
        return fire_df
    
    def load_watersheds(self, test_mode=False):
        """Load watershed boundaries"""
        print("\n" + "="*60)
        print("STEP 2: LOADING WATERSHED DATA")
        print("="*60)
        
        print(f"Loading watersheds from {self.watershed_path}...")
        watersheds = gpd.read_file(self.watershed_path)
        print(f"Loaded {len(watersheds):,} watersheds")
        
        # Test mode - use subset
        if test_mode:
            print("TEST MODE: Using California watersheds...")
            from shapely.geometry import box
            ca_bbox = box(-125, 32, -114, 42)
            watersheds = watersheds[watersheds.intersects(ca_bbox)]
            print(f"Test subset: {len(watersheds):,} watersheds")
        
        # Ensure HUC12 column exists
        if 'HUC12' in watersheds.columns and 'huc12' not in watersheds.columns:
            watersheds['huc12'] = watersheds['HUC12']
        elif 'huc_12' in watersheds.columns and 'huc12' not in watersheds.columns:
            watersheds['huc12'] = watersheds['huc_12']
        
        # Calculate area
        watersheds_proj = watersheds.to_crs('EPSG:6933')
        watersheds['area_km2'] = watersheds_proj.geometry.area / 1e6
        
        self.watersheds = watersheds
        
        return watersheds
    
    def detect_fire_episodes(self):
        """Detect fire episodes using improved algorithm"""
        print("\n" + "="*60)
        print("STEP 3: DETECTING FIRE EPISODES")
        print("="*60)
        
        # Reuse precomputed episodes if provided
        reuse_path = os.environ.get('EPISODES_GPKG', '').strip()
        if reuse_path and Path(reuse_path).exists():
            print(f"Reusing precomputed episodes from: {reuse_path}")
            self.episodes_gdf = gpd.read_file(reuse_path)
        else:
            # Initialize detector
            detector = ImprovedFireEpisodeDetector()
            # Detect episodes
            self.fire_df, self.episodes_gdf = detector.detect_episodes(self.fire_df)
        
        # Save episodes
        self.episodes_gdf.to_file(
            self.run_dir / 'fire_episodes.gpkg', 
            driver='GPKG'
        )
        
        print(f"Saved {len(self.episodes_gdf)} fire episodes")
        
        return self.episodes_gdf
    
    def calculate_watershed_metrics(self):
        """Calculate fire regime metrics for watersheds"""
        print("\n" + "="*60)
        print("STEP 4: CALCULATING WATERSHED FIRE METRICS")
        print("="*60)
        
        # Reuse precomputed watershed metrics if provided
        reuse_path = os.environ.get('WATERSHEDS_WITH_FIRE_GPKG', '').strip()
        if reuse_path and Path(reuse_path).exists():
            print(f"Reusing precomputed watershed metrics from: {reuse_path}")
            self.watersheds_with_fire = gpd.read_file(reuse_path)
        else:
            # Initialize calculator
            calculator = FireRegimeMetrics()
            # Calculate metrics
            self.watersheds_with_fire = calculator.calculate_watershed_fire_metrics(
                self.episodes_gdf, self.watersheds
            )
        
        # Validate HSBF
        max_hsbf = self.watersheds_with_fire['hsbf'].max()
        print(f"\nValidation: Maximum HSBF = {max_hsbf:.3f}")
        if max_hsbf > 1.0:
            print("WARNING: HSBF > 1.0 detected! Check calculations.")
        else:
            print("✓ HSBF values are valid (all ≤ 1.0)")
        
        # Save watershed metrics
        self.watersheds_with_fire.to_file(
            self.run_dir / 'watersheds_with_fire_metrics.gpkg',
            driver='GPKG'
        )
        
        return self.watersheds_with_fire
    
    def cluster_watersheds(self, min_fire_count=1):
        """Cluster watersheds by fire regime"""
        print("\n" + "="*60)
        print("STEP 5: CLUSTERING WATERSHEDS BY FIRE REGIME")
        print("="*60)
        
        # Initialize clusterer
        clusterer = WatershedFireRegimeClustering(self.watersheds_with_fire)
        
        # Prepare features
        clusterer.prepare_features(min_fire_count=min_fire_count)
        
        # Perform clustering
        clusterer.perform_clustering(n_clusters=None, method='kmeans')
        
        # Statistical validation
        clusterer.statistical_validation()
        
        # Export results
        self.watersheds_clustered = clusterer.export_results(
            self.run_dir / 'watersheds_fire_regimes.gpkg'
        )
        
        return self.watersheds_clustered
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        report_path = self.run_dir / 'analysis_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FIRE-WATERSHED ANALYSIS SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            f.write("DATA SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Fire detections: {len(self.fire_df):,}\n")
            f.write(f"Fire episodes: {len(self.episodes_gdf):,}\n")
            f.write(f"Watersheds analyzed: {len(self.watersheds):,}\n")
            f.write(f"Watersheds with fires: {(self.watersheds_with_fire['episode_count'] > 0).sum():,}\n\n")
            
            # Episode statistics
            f.write("FIRE EPISODE STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean duration: {self.episodes_gdf['duration_days'].mean():.1f} days\n")
            f.write(f"Mean area: {self.episodes_gdf['area_km2'].mean():.1f} km²\n")
            f.write(f"Total area burned: {self.episodes_gdf['area_km2'].sum():.0f} km²\n\n")
            
            # Watershed statistics
            f.write("WATERSHED FIRE METRICS\n")
            f.write("-"*40 + "\n")
            
            watersheds_with_fires = self.watersheds_with_fire[
                self.watersheds_with_fire['episode_count'] > 0
            ]
            
            f.write(f"Max HSBF: {self.watersheds_with_fire['hsbf'].max():.3f}\n")
            f.write(f"Mean HSBF (where >0): {watersheds_with_fires['hsbf'].mean():.3f}\n")
            f.write(f"Watersheds with HSBF >0.3: {(self.watersheds_with_fire['hsbf'] > 0.3).sum():,}\n")
            f.write(f"Watersheds with HSBF >0.5: {(self.watersheds_with_fire['hsbf'] > 0.5).sum():,}\n\n")
            
            # Clustering results
            if hasattr(self, 'watersheds_clustered'):
                f.write("FIRE REGIME CLUSTERING\n")
                f.write("-"*40 + "\n")
                
                cluster_counts = self.watersheds_clustered['fire_regime_cluster'].value_counts()
                f.write(f"Number of clusters: {len(cluster_counts)}\n")
                for cluster_id, count in cluster_counts.items():
                    f.write(f"  Cluster {cluster_id}: {count} watersheds\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Report saved to: {report_path}")
    
    def create_final_visualizations(self):
        """Create final research visualizations"""
        print("\n" + "="*60)
        print("CREATING FINAL VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Fire episode locations
        ax1 = axes[0, 0]
        self.watersheds.boundary.plot(ax=ax1, linewidth=0.1, color='gray', alpha=0.3)
        self.episodes_gdf.plot(ax=ax1, color='red', alpha=0.5, markersize=10)
        ax1.set_title('Fire Episode Locations', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Episode count per watershed
        ax2 = axes[0, 1]
        self.watersheds.boundary.plot(ax=ax2, linewidth=0.1, color='gray', alpha=0.3)
        fire_watersheds = self.watersheds_with_fire[self.watersheds_with_fire['episode_count'] > 0]
        if len(fire_watersheds) > 0:
            fire_watersheds.plot(ax=ax2, column='episode_count', cmap='YlOrRd', 
                                legend=True, legend_kwds={'shrink': 0.7})
        ax2.set_title('Fire Episodes per Watershed', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. HSBF distribution
        ax3 = axes[0, 2]
        self.watersheds.boundary.plot(ax=ax3, linewidth=0.1, color='gray', alpha=0.3)
        hsbf_watersheds = self.watersheds_with_fire[self.watersheds_with_fire['hsbf'] > 0]
        if len(hsbf_watersheds) > 0:
            hsbf_watersheds.plot(ax=ax3, column='hsbf', cmap='RdPu', 
                                legend=True, legend_kwds={'shrink': 0.7})
        ax3.set_title('Hydrologically Significant Burn Fraction', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. Fire regime clusters
        ax4 = axes[1, 0]
        if 'fire_regime_cluster' in self.watersheds_clustered.columns:
            self.watersheds.boundary.plot(ax=ax4, linewidth=0.1, color='gray', alpha=0.3)
            clustered = self.watersheds_clustered[self.watersheds_clustered['fire_regime_cluster'] >= 0]
            if len(clustered) > 0:
                # For categorical legends, matplotlib Legend does not accept 'shrink'
                clustered.plot(ax=ax4, column='fire_regime_cluster', cmap='viridis',
                             categorical=True, legend=True)
        ax4.set_title('Fire Regime Clusters', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. Episode duration histogram
        ax5 = axes[1, 1]
        self.episodes_gdf['duration_days'].hist(bins=30, ax=ax5, color='darkred', alpha=0.7)
        ax5.set_xlabel('Duration (days)')
        ax5.set_ylabel('Number of Episodes')
        ax5.set_title('Fire Episode Duration Distribution', fontsize=12, fontweight='bold')
        
        # 6. HSBF histogram
        ax6 = axes[1, 2]
        hsbf_positive = self.watersheds_with_fire[self.watersheds_with_fire['hsbf'] > 0]['hsbf']
        if len(hsbf_positive) > 0:
            hsbf_positive.hist(bins=30, ax=ax6, color='purple', alpha=0.7)
        ax6.set_xlabel('HSBF')
        ax6.set_ylabel('Number of Watersheds')
        ax6.set_title('HSBF Distribution', fontsize=12, fontweight='bold')
        ax6.axvline(1.0, color='red', linestyle='--', label='Maximum possible (1.0)')
        ax6.legend()
        
        plt.suptitle('Fire-Watershed Analysis Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        viz_path = self.run_dir / 'final_results_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {viz_path}")
    
    def run_complete_analysis(self, test_mode=False):
        """Run the complete analysis pipeline"""
        print("\n" + "="*60)
        print("RUNNING COMPLETE ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Load fire data
        self.load_and_prepare_fire_data(quality_threshold=70, test_mode=test_mode)
        
        # Step 2: Load watersheds
        self.load_watersheds(test_mode=test_mode)
        
        # Step 3: Detect fire episodes
        self.detect_fire_episodes()
        
        # Step 4: Calculate watershed metrics
        self.calculate_watershed_metrics()
        
        # Step 5: Cluster watersheds
        self.cluster_watersheds(min_fire_count=1)
        
        # Step 6: Generate report
        self.generate_summary_report()
        
        # Step 7: Create visualizations
        self.create_final_visualizations()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"All results saved to: {self.run_dir}")
        
        return self.watersheds_clustered


def main():
    """Main execution function"""
    
    # Limit BLAS threads to reduce memory overhead
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('BLIS_NUM_THREADS', '1')
    os.environ.setdefault('MALLOC_ARENA_MAX', '2')
    
    # Enforce process memory cap at ~75% of system RAM on Linux
    try:
        if platform.system().lower() == 'linux' and resource is not None:
            total = psutil.virtual_memory().total
            try:
                frac = float(os.environ.get('MEMORY_FRACTION', '0.75'))
                if not (0.1 <= frac <= 0.95):
                    frac = 0.75
            except Exception:
                frac = 0.75
            limit_bytes = int(total * frac)
            # Apply to virtual address space; some libs may allocate more VM than RSS
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            # Also set data segment limit if available
            if hasattr(resource, 'RLIMIT_DATA'):
                resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))
            print(f"Memory limit set to {limit_bytes / (1024**3):.1f} GB (~{frac*100:.0f}% of total)")
    except Exception as e:
        print(f"Warning: Unable to set memory limit: {e}")
    
    # Configure paths
    fire_data_path = "Json_files/fire_modis_us.json"
    watershed_data_path = "Json_files/huc12_conus.geojson"
    
    # Check if files exist
    if not Path(fire_data_path).exists():
        print(f"Error: Fire data not found at {fire_data_path}")
        return
    
    if not Path(watershed_data_path).exists():
        print(f"Error: Watershed data not found at {watershed_data_path}")
        return
    
    # Initialize analyzer
    analyzer = CompleteFireWatershedAnalysis(
        fire_data_path,
        watershed_data_path,
        output_dir='outputs_fixed'
    )
    
    # Run analysis (test mode first)
    print("\n" + "="*60)
    print("STARTING IN TEST MODE (California only)")
    print("="*60)
    
    results = analyzer.run_complete_analysis(test_mode=False)
    
    print("\n" + "="*60)
    print("SUCCESS! Test run complete.")
    print("="*60)
    print("\nTo run full CONUS analysis, change test_mode=False")
    
    return results


if __name__ == "__main__":
    results = main()