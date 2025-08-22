#!/usr/bin/env python3
"""
Execute Advanced Fire Regime Clustering Analysis
Leverages high-performance computing for unbiased clustering
"""

import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the advanced clustering module
from advanced_fire_regime_clustering import AdvancedFireRegimeClustering

# Import visualization utilities
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_comprehensive_visualizations(clusterer, output_dir):
    """
    Create publication-quality visualizations of clustering results
    """
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style for publication quality
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    
    # 1. Create stratification visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Fire activity distribution
    ax1 = axes[0, 0]
    fire_strata = clusterer.watersheds['fire_strata'].value_counts()
    colors = ['#f0f0f0', '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
    fire_strata.plot(kind='bar', ax=ax1, color=colors, alpha=0.8)
    ax1.set_title('Fire Activity Stratification', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fire Activity Level')
    ax1.set_ylabel('Number of Watersheds')
    ax1.tick_params(axis='x', rotation=45)
    
    # HSBF distribution with stratification
    ax2 = axes[0, 1]
    fire_watersheds = clusterer.fire_watersheds
    
    # Create violin plot for each intensity strata
    if 'fire_intensity_strata' in fire_watersheds.columns:
        strata_order = ['Very Low', 'Low', 'Moderate', 'High', 'Extreme']
        existing_strata = [s for s in strata_order if s in fire_watersheds['fire_intensity_strata'].unique()]
        
        data_for_violin = []
        labels_for_violin = []
        for strata in existing_strata:
            mask = fire_watersheds['fire_intensity_strata'] == strata
            data_for_violin.append(fire_watersheds[mask]['hsbf'].values)
            labels_for_violin.append(strata)
        
        parts = ax2.violinplot(data_for_violin, positions=range(len(labels_for_violin)),
                               showmeans=True, showmedians=True)
        ax2.set_xticks(range(len(labels_for_violin)))
        ax2.set_xticklabels(labels_for_violin, rotation=45)
    else:
        fire_watersheds['hsbf'].hist(bins=50, ax=ax2, color='darkred', alpha=0.7)
    
    ax2.set_title('HSBF Distribution by Fire Intensity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fire Intensity Strata')
    ax2.set_ylabel('HSBF Value')
    
    # Clustering results summary
    ax3 = axes[1, 0]
    if hasattr(clusterer, 'cluster_profiles'):
        cluster_sizes = []
        cluster_names = []
        for cid, profile in clusterer.cluster_profiles.items():
            cluster_sizes.append(profile['size'])
            # Shorten name for display
            name_parts = profile['name'].split('_')
            short_name = f"C{cid}: {name_parts[0]}\n{name_parts[1]}"
            cluster_names.append(short_name)
        
        ax3.bar(range(len(cluster_sizes)), cluster_sizes, color='steelblue', alpha=0.7)
        ax3.set_xticks(range(len(cluster_names)))
        ax3.set_xticklabels(cluster_names, rotation=0, ha='center')
        ax3.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Watersheds')
    
    # Feature importance (using PCA loadings)
    ax4 = axes[1, 1]
    if hasattr(clusterer, 'pca'):
        # Get feature names
        feature_names = clusterer.feature_matrix.columns[:10]  # Top 10 features
        
        # Get loadings for first 2 components
        loadings = clusterer.pca.components_[:2, :len(feature_names)]
        
        # Create heatmap
        im = ax4.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, rotation=45, ha='right')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['PC1', 'PC2'])
        ax4.set_title('Feature Importance (PCA Loadings)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    plt.suptitle('Fire Regime Clustering Analysis Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'clustering_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create cluster comparison matrix
    if hasattr(clusterer, 'cluster_profiles'):
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        
        profiles = clusterer.cluster_profiles
        cluster_ids = list(profiles.keys())
        
        # Metrics to compare
        metrics = [
            ('episode_count', 'mean', 'Fire Episodes'),
            ('hsbf', 'mean', 'HSBF'),
            ('fire_size', 'mean_area', 'Mean Fire Area (km²)'),
            ('intensity', 'mean_frp', 'Mean FRP (MW)'),
            ('temporal', 'duration', 'Mean Duration (days)'),
            ('temporal', 'seasonality', 'Seasonality Index'),
            ('temporal', 'return_interval', 'Return Interval (years)'),
            ('hsbf', 'max', 'Max HSBF'),
            ('intensity', 'max_frp', 'Max FRP (MW)')
        ]
        
        for idx, (metric_group, metric_name, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            values = []
            for cid in cluster_ids:
                if isinstance(profiles[cid][metric_group], dict):
                    values.append(profiles[cid][metric_group][metric_name])
                else:
                    values.append(profiles[cid][metric_group])
            
            bars = ax.bar(range(len(cluster_ids)), values, color='teal', alpha=0.7)
            
            # Color bars by value
            norm = plt.Normalize(min(values), max(values))
            colors = plt.cm.YlOrRd(norm(values))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(range(len(cluster_ids)))
            ax.set_xticklabels([f"C{cid}" for cid in cluster_ids])
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Cluster Characteristics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'cluster_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Create spatial visualization if geometry available
    if 'geometry' in clusterer.fire_watersheds.columns:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Original fire activity
        ax1 = axes[0]
        clusterer.fire_watersheds.plot(column='episode_count', cmap='YlOrRd', 
                                      legend=True, ax=ax1,
                                      legend_kwds={'label': 'Fire Episodes'})
        ax1.set_title('Fire Episode Count', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # HSBF distribution
        ax2 = axes[1]
        clusterer.fire_watersheds.plot(column='hsbf', cmap='RdPu',
                                      legend=True, ax=ax2,
                                      legend_kwds={'label': 'HSBF'})
        ax2.set_title('Hydrologically Significant Burn Fraction', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Cluster assignments
        ax3 = axes[2]
        if 'final_cluster' in clusterer.fire_watersheds.columns:
            clusterer.fire_watersheds.plot(column='final_cluster', cmap='viridis',
                                          categorical=True, legend=True, ax=ax3)
            ax3.set_title('Fire Regime Clusters', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        plt.suptitle('Spatial Distribution of Fire Regimes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'spatial_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Create evaluation metrics comparison
    if clusterer.clustering_results:
        # Prepare data for visualization
        results_data = []
        for key, result in clusterer.clustering_results.items():
            if result['metrics'] and 'silhouette' in result['metrics']:
                results_data.append({
                    'method': f"{result['algorithm']}",
                    'transform': result['transform'],
                    'n_clusters': result['n_clusters'],
                    'silhouette': result['metrics']['silhouette'],
                    'davies_bouldin': result['metrics']['davies_bouldin']
                })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Silhouette scores by algorithm
            ax1 = axes[0, 0]
            algorithms = results_df['method'].unique()
            for algo in algorithms:
                algo_data = results_df[results_df['method'] == algo]
                ax1.scatter(algo_data['n_clusters'], algo_data['silhouette'], 
                          label=algo, alpha=0.6, s=50)
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_title('Silhouette Score by Algorithm', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Davies-Bouldin scores
            ax2 = axes[0, 1]
            for algo in algorithms:
                algo_data = results_df[results_df['method'] == algo]
                ax2.scatter(algo_data['n_clusters'], algo_data['davies_bouldin'],
                          label=algo, alpha=0.6, s=50)
            ax2.set_xlabel('Number of Clusters')
            ax2.set_ylabel('Davies-Bouldin Score (lower is better)')
            ax2.set_title('Davies-Bouldin Score by Algorithm', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Transform comparison
            ax3 = axes[1, 0]
            transform_avg = results_df.groupby('transform')['silhouette'].mean()
            transform_avg.plot(kind='bar', ax=ax3, color='skyblue', alpha=0.7)
            ax3.set_title('Average Silhouette by Transform', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Transform Method')
            ax3.set_ylabel('Average Silhouette Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # Best configurations
            ax4 = axes[1, 1]
            top10 = results_df.nlargest(10, 'silhouette')
            y_pos = np.arange(len(top10))
            ax4.barh(y_pos, top10['silhouette'].values, color='green', alpha=0.7)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([f"{row['method'][:8]}-{row['n_clusters']}" 
                                for _, row in top10.iterrows()])
            ax4.set_xlabel('Silhouette Score')
            ax4.set_title('Top 10 Clustering Configurations', fontsize=12, fontweight='bold')
            
            plt.suptitle('Clustering Algorithm Performance Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(viz_dir / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to {viz_dir}")
    
    return viz_dir

def generate_research_report(clusterer, output_dir):
    """
    Generate comprehensive research report
    """
    print("\n" + "="*60)
    print("GENERATING RESEARCH REPORT")
    print("="*60)
    
    report_path = Path(output_dir) / 'fire_regime_clustering_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Fire Regime Clustering Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Summary statistics
        n_total = len(clusterer.watersheds)
        n_fire = len(clusterer.fire_watersheds) if hasattr(clusterer, 'fire_watersheds') else 0
        n_clusters = len(clusterer.cluster_profiles) if hasattr(clusterer, 'cluster_profiles') else 0
        
        f.write(f"- Total watersheds analyzed: {n_total:,}\n")
        f.write(f"- Watersheds with fire activity: {n_fire:,} ({n_fire/n_total*100:.1f}%)\n")
        f.write(f"- Fire regime clusters identified: {n_clusters}\n\n")
        
        f.write("## Methodology\n\n")
        f.write("### Addressing Bias in Fire Regime Clustering\n\n")
        f.write("The analysis employs several strategies to address bias from watersheds with minimal fire activity:\n\n")
        f.write("1. **Stratified Analysis**: Watersheds are first stratified by fire activity level\n")
        f.write("2. **Focused Clustering**: Only fire-affected watersheds are clustered\n")
        f.write("3. **Multi-Algorithm Ensemble**: Multiple clustering algorithms are tested\n")
        f.write("4. **Robust Transformations**: Multiple feature transformations are applied\n")
        f.write("5. **Parallel Processing**: Leverages high-performance computing for comprehensive analysis\n\n")
        
        f.write("### Fire Activity Stratification\n\n")
        if 'fire_strata' in clusterer.watersheds.columns:
            strata_counts = clusterer.watersheds['fire_strata'].value_counts()
            f.write("| Strata | Count | Percentage |\n")
            f.write("|--------|-------|------------|\n")
            for strata, count in strata_counts.items():
                pct = count / len(clusterer.watersheds) * 100
                f.write(f"| {strata} | {count:,} | {pct:.1f}% |\n")
            f.write("\n")
        
        f.write("## Cluster Characteristics\n\n")
        if hasattr(clusterer, 'cluster_profiles'):
            for cid, profile in clusterer.cluster_profiles.items():
                f.write(f"### Cluster {cid}: {profile['name']}\n\n")
                f.write(f"- **Size**: {profile['size']} watersheds\n")
                f.write(f"- **Fire Episodes**: {profile['episode_count']['mean']:.1f} ± {profile['episode_count']['std']:.1f}\n")
                f.write(f"- **HSBF**: {profile['hsbf']['mean']:.3f} (max: {profile['hsbf']['max']:.3f})\n")
                f.write(f"- **Mean Fire Area**: {profile['fire_size']['mean_area']:.1f} km²\n")
                f.write(f"- **Mean FRP**: {profile['intensity']['mean_frp']:.1f} MW\n")
                f.write(f"- **Duration**: {profile['temporal']['duration']:.1f} days\n")
                f.write(f"- **Seasonality**: {profile['temporal']['seasonality']:.2f}\n")
                f.write(f"- **Return Interval**: {profile['temporal']['return_interval']:.1f} years\n\n")
        
        f.write("## Model Performance\n\n")
        if hasattr(clusterer, 'best_clustering'):
            best = clusterer.best_clustering
            f.write(f"### Best Clustering Configuration\n\n")
            f.write(f"- **Algorithm**: {best['algorithm']}\n")
            f.write(f"- **Transform**: {best['transform']}\n")
            f.write(f"- **Number of Clusters**: {best['n_clusters']}\n")
            if best['metrics']:
                f.write(f"- **Silhouette Score**: {best['metrics'].get('silhouette', 'N/A'):.3f}\n")
                f.write(f"- **Davies-Bouldin Score**: {best['metrics'].get('davies_bouldin', 'N/A'):.3f}\n")
                f.write(f"- **Calinski-Harabasz Score**: {best['metrics'].get('calinski_harabasz', 'N/A'):.1f}\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Validation**: Validate clusters against known fire-prone regions\n")
        f.write("2. **Ecological Context**: Incorporate vegetation and climate data\n")
        f.write("3. **Temporal Analysis**: Examine cluster stability over time\n")
        f.write("4. **Management Applications**: Develop cluster-specific fire management strategies\n")
        f.write("5. **Prediction Models**: Use clusters as basis for fire risk prediction\n\n")
        
        f.write("## Technical Notes\n\n")
        f.write(f"- Parallel workers used: {clusterer.n_jobs}\n")
        f.write(f"- Clustering experiments conducted: {len(clusterer.clustering_results)}\n")
        f.write("- Feature transformations: standard, robust, quantile, log-robust\n")
        f.write("- Algorithms tested: K-Means, GMM, Bayesian GMM, HDBSCAN, OPTICS, Spectral, Hierarchical, BIRCH\n\n")
    
    print(f"Report saved to {report_path}")
    return report_path

def main():
    """
    Main execution function for advanced clustering
    """
    print("="*60)
    print("ADVANCED FIRE REGIME CLUSTERING ANALYSIS")
    print("="*60)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configure paths (prefer ecological context dataset if available)
    input_file = Path('06_Ecological_Context') / 'watersheds_with_ecological_context.parquet'
    if not input_file.exists():
        input_file = Path('outputs_fixed') / 'watersheds_with_fire_metrics.gpkg'
    
    # Find most recent analysis output
    if not input_file.exists():
        # Try to find the most recent output
        outputs_dir = Path('outputs_fixed')
        if outputs_dir.exists():
            analysis_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith('analysis_run_')])
            if analysis_dirs:
                latest_dir = analysis_dirs[-1]
                input_file = latest_dir / 'watersheds_with_fire_metrics.gpkg'
    
    if not input_file.exists():
        print(f"Error: Could not find input file: {input_file}")
        print("Please run the fire-watershed integration analysis first.")
        return
    
    print(f"Loading data from: {input_file}")
    
    # Load watershed data (supports Parquet and GeoPackage)
    if input_file.suffix == '.parquet':
        # Use GeoPandas to preserve geometry/CRS
        watersheds_gdf = gpd.read_parquet(input_file)
    else:
        watersheds_gdf = gpd.read_file(input_file)
    print(f"Loaded {len(watersheds_gdf):,} watersheds")
    
    # Initialize advanced clustering (cap workers to 32)
    clusterer = AdvancedFireRegimeClustering(watersheds_gdf, n_jobs=32)
    
    # Step 1: Stratified analysis
    fire_watersheds = clusterer.stratified_fire_analysis()
    
    if len(fire_watersheds) < 30:
        print("Warning: Too few fire-affected watersheds for robust clustering")
        print("Consider adjusting thresholds or using a larger study area")
        return
    
    # Step 2: Prepare multiscale features (include ecological variables)
    transformed_features = clusterer.prepare_multiscale_features(include_ecological=True)
    
    # Step 3: Run parallel clustering suite
    clustering_results = clusterer.parallel_clustering_suite(
        transformed_features,
        n_clusters_range=range(3, min(11, len(fire_watersheds) // 20 + 1))
    )
    
    # Step 4: Select best clustering
    best_clustering = clusterer.select_best_clustering(criteria='balanced')
    
    # Step 5: Create ensemble clustering
    if len(clustering_results) >= 5:
        ensemble_labels = clusterer.ensemble_clustering(top_n=min(5, len(clustering_results)))
    
    # Step 6: Analyze and characterize clusters
    cluster_profiles = clusterer.analyze_and_characterize_clusters()
    
    # Step 7: Validate stability
    if best_clustering:
        stability_results = clusterer.validate_clustering_stability(n_iterations=10)
    
    # Step 8: Export results
    output_dir = clusterer.export_results('05_Clustering')
    
    # Step 9: Create visualizations
    viz_dir = create_comprehensive_visualizations(clusterer, output_dir)
    
    # Step 10: Generate research report
    report_path = generate_research_report(clusterer, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations in: {viz_dir}")
    print(f"Report at: {report_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return clusterer

if __name__ == "__main__":
    clusterer = main()