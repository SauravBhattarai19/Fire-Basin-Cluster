#!/usr/bin/env python3
"""
Validation and quality assessment module for fire episode clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ValidationFramework:
    """
    Validate fire episode clustering results and generate quality reports
    """
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger('ValidationFramework')
        
        # Validation settings
        self.validation_config = config['validation']
        self.sample_size = config['output']['validation_sample_size']
        self.generate_plots = config['output']['generate_visualizations']
        
        # Create validation output directory
        self.validation_dir = self.output_dir / 'validation'
        self.validation_dir.mkdir(exist_ok=True)
        
    def validate_clustering(self, fire_df, labels, coords, times):
        """
        Perform comprehensive validation of clustering results
        """
        self.logger.info("Starting clustering validation")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'clustering_metrics': self._calculate_clustering_metrics(labels, coords, times),
            'cluster_statistics': self._analyze_cluster_statistics(fire_df, labels),
            'quality_assessment': self._assess_clustering_quality(fire_df, labels),
            'parameter_sensitivity': None  # Set during parameter optimization
        }
        
        # Generate validation plots if configured
        if self.generate_plots:
            self._generate_validation_plots(fire_df, labels, coords)
        
        return validation_results
    
    def _calculate_clustering_metrics(self, labels, coords, times):
        """Calculate standard clustering evaluation metrics"""
        
        metrics = {
            'total_points': len(labels),
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_noise': (labels == -1).sum(),
            'noise_ratio': (labels == -1).sum() / len(labels)
        }
        
        # Only calculate metrics if we have valid clusters
        if metrics['n_clusters'] > 1 and metrics['n_noise'] < len(labels) - 10:
            non_noise_mask = labels >= 0
            
            # Prepare features for metric calculation
            times_numeric = (times - times.min()) / np.timedelta64(1, 'D')
            features = np.column_stack([
                coords[non_noise_mask] / 1000,  # Convert to km
                times_numeric[non_noise_mask].reshape(-1, 1)
            ])
            
            try:
                # Silhouette coefficient
                metrics['silhouette_score'] = silhouette_score(
                    features, labels[non_noise_mask]
                )
                
                # Calinski-Harabasz index
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                    features, labels[non_noise_mask]
                )
                
                # Davies-Bouldin index
                metrics['davies_bouldin_score'] = davies_bouldin_score(
                    features, labels[non_noise_mask]
                )
            except Exception as e:
                self.logger.warning(f"Error calculating clustering metrics: {e}")
                metrics['silhouette_score'] = -1
                metrics['calinski_harabasz_score'] = -1
                metrics['davies_bouldin_score'] = -1
        else:
            metrics['silhouette_score'] = -1
            metrics['calinski_harabasz_score'] = -1
            metrics['davies_bouldin_score'] = -1
        
        return metrics
    
    def _analyze_cluster_statistics(self, fire_df, labels):
        """Analyze statistical properties of clusters"""
        
        # Add labels to dataframe
        fire_df_labeled = fire_df.copy()
        fire_df_labeled['cluster_id'] = labels
        
        # Filter out noise
        clustered_df = fire_df_labeled[fire_df_labeled['cluster_id'] >= 0]
        
        if len(clustered_df) == 0:
            return {'no_clusters': True}
        
        # Cluster size distribution
        cluster_sizes = clustered_df['cluster_id'].value_counts()
        
        # Temporal span of clusters
        temporal_spans = []
        spatial_extents = []
        
        for cluster_id in cluster_sizes.index[:100]:  # Limit to first 100 for performance
            cluster_data = clustered_df[clustered_df['cluster_id'] == cluster_id]
            
            # Temporal span
            time_span = (cluster_data['acq_datetime'].max() - 
                        cluster_data['acq_datetime'].min()).total_seconds() / 3600
            temporal_spans.append(time_span)
            
            # Spatial extent
            spatial_extent = np.sqrt(
                (cluster_data['x_proj'].max() - cluster_data['x_proj'].min())**2 +
                (cluster_data['y_proj'].max() - cluster_data['y_proj'].min())**2
            ) / 1000  # Convert to km
            spatial_extents.append(spatial_extent)
        
        statistics = {
            'cluster_size_distribution': {
                'mean': float(cluster_sizes.mean()),
                'std': float(cluster_sizes.std()),
                'min': int(cluster_sizes.min()),
                'max': int(cluster_sizes.max()),
                'percentiles': {
                    '25': int(cluster_sizes.quantile(0.25)),
                    '50': int(cluster_sizes.quantile(0.50)),
                    '75': int(cluster_sizes.quantile(0.75)),
                    '90': int(cluster_sizes.quantile(0.90))
                }
            },
            'temporal_span_hours': {
                'mean': float(np.mean(temporal_spans)),
                'std': float(np.std(temporal_spans)),
                'min': float(np.min(temporal_spans)),
                'max': float(np.max(temporal_spans))
            },
            'spatial_extent_km': {
                'mean': float(np.mean(spatial_extents)),
                'std': float(np.std(spatial_extents)),
                'min': float(np.min(spatial_extents)),
                'max': float(np.max(spatial_extents))
            }
        }
        
        return statistics
    
    def _assess_clustering_quality(self, fire_df, labels):
        """Assess quality of clustering based on domain knowledge"""
        
        fire_df_labeled = fire_df.copy()
        fire_df_labeled['cluster_id'] = labels
        clustered_df = fire_df_labeled[fire_df_labeled['cluster_id'] >= 0]
        
        if len(clustered_df) == 0:
            return {'no_clusters': True}
        
        quality_issues = []
        quality_scores = {}
        
        # Check for unrealistic cluster properties
        for cluster_id in clustered_df['cluster_id'].unique()[:100]:
            cluster_data = clustered_df[clustered_df['cluster_id'] == cluster_id]
            
            # Temporal coherence
            time_span_days = (cluster_data['acq_datetime'].max() - 
                            cluster_data['acq_datetime'].min()).days
            
            if time_span_days > self.validation_config['max_episode_duration_days']:
                quality_issues.append(f"Cluster {cluster_id}: duration > {self.validation_config['max_episode_duration_days']} days")
            
            # Spatial coherence
            spatial_std = np.sqrt(
                cluster_data['x_proj'].std()**2 + 
                cluster_data['y_proj'].std()**2
            ) / 1000  # km
            
            if spatial_std > 100:  # More than 100km standard deviation
                quality_issues.append(f"Cluster {cluster_id}: very dispersed (std={spatial_std:.1f}km)")
            
            # Check for unrealistic spread rate
            if len(cluster_data) > 5 and time_span_days > 0:
                # Simple spread rate estimation
                max_distance = np.sqrt(
                    (cluster_data['x_proj'].max() - cluster_data['x_proj'].min())**2 +
                    (cluster_data['y_proj'].max() - cluster_data['y_proj'].min())**2
                ) / 1000  # km
                
                spread_rate = max_distance / (time_span_days * 24)  # km/h
                
                if spread_rate > self.validation_config['max_episode_spread_rate_kmh']:
                    quality_issues.append(f"Cluster {cluster_id}: unrealistic spread rate ({spread_rate:.1f} km/h)")
        
        # Calculate quality scores
        total_clusters = len(clustered_df['cluster_id'].unique())
        problematic_clusters = len(set([int(issue.split(':')[0].split()[-1]) 
                                       for issue in quality_issues 
                                       if 'Cluster' in issue]))
        
        quality_scores['cluster_quality_ratio'] = 1 - (problematic_clusters / total_clusters) if total_clusters > 0 else 0
        quality_scores['n_quality_issues'] = len(quality_issues)
        
        # Sample quality issues for report
        quality_assessment = {
            'quality_scores': quality_scores,
            'sample_issues': quality_issues[:10] if quality_issues else [],
            'total_issues': len(quality_issues)
        }
        
        return quality_assessment
    
    def validate_episodes(self, episodes_df):
        """Validate generated fire episodes"""
        
        self.logger.info(f"Validating {len(episodes_df)} fire episodes")
        
        validation_results = {
            'total_episodes': len(episodes_df),
            'valid_episodes': episodes_df['is_valid'].sum() if 'is_valid' in episodes_df else len(episodes_df),
            'episode_statistics': self._calculate_episode_statistics(episodes_df),
            'quality_distribution': self._analyze_episode_quality(episodes_df),
            'validation_summary': self._generate_validation_summary(episodes_df)
        }
        
        return validation_results
    
    def _calculate_episode_statistics(self, episodes_df):
        """Calculate statistics of episode characteristics"""
        
        stats_columns = [
            'duration_days', 'area_km2', 'total_energy_mwh', 
            'detection_count', 'mean_frp', 'spatial_coherence_score'
        ]
        
        statistics = {}
        
        for col in stats_columns:
            if col in episodes_df.columns:
                statistics[col] = {
                    'mean': float(episodes_df[col].mean()),
                    'std': float(episodes_df[col].std()),
                    'min': float(episodes_df[col].min()),
                    'max': float(episodes_df[col].max()),
                    'percentiles': {
                        '25': float(episodes_df[col].quantile(0.25)),
                        '50': float(episodes_df[col].quantile(0.50)),
                        '75': float(episodes_df[col].quantile(0.75)),
                        '90': float(episodes_df[col].quantile(0.90))
                    }
                }
        
        # Episode type distribution
        if 'episode_type' in episodes_df.columns:
            type_counts = episodes_df['episode_type'].value_counts()
            statistics['episode_types'] = type_counts.to_dict()
        
        return statistics
    
    def _analyze_episode_quality(self, episodes_df):
        """Analyze quality distribution of episodes"""
        
        quality_metrics = [
            'spatial_coherence_score', 'data_completeness_score',
            'mean_confidence', 'high_confidence_ratio'
        ]
        
        quality_dist = {}
        
        for metric in quality_metrics:
            if metric in episodes_df.columns:
                quality_dist[metric] = {
                    'high_quality': (episodes_df[metric] > 0.8).sum(),
                    'medium_quality': ((episodes_df[metric] > 0.5) & 
                                     (episodes_df[metric] <= 0.8)).sum(),
                    'low_quality': (episodes_df[metric] <= 0.5).sum()
                }
        
        return quality_dist
    
    def _generate_validation_summary(self, episodes_df):
        """Generate summary of validation results"""
        
        summary = {
            'data_quality': 'GOOD' if episodes_df['mean_confidence'].mean() > 75 else 'MODERATE',
            'spatial_coherence': 'GOOD' if episodes_df['spatial_coherence_score'].mean() > 0.7 else 'MODERATE',
            'temporal_consistency': 'GOOD' if episodes_df['detection_consistency'].mean() > 0.5 else 'MODERATE',
            'recommendations': []
        }
        
        # Generate recommendations
        if episodes_df['spatial_coherence_score'].mean() < 0.7:
            summary['recommendations'].append(
                "Consider reducing spatial_eps parameter for tighter clusters"
            )
        
        if episodes_df['detection_consistency'].mean() < 0.5:
            summary['recommendations'].append(
                "Many episodes have gaps in detection - consider adjusting temporal_eps"
            )
        
        noise_ratio = 1 - (episodes_df['is_valid'].sum() / len(episodes_df)) if 'is_valid' in episodes_df else 0
        if noise_ratio > 0.3:
            summary['recommendations'].append(
                f"High invalid episode ratio ({noise_ratio:.1%}) - review validation criteria"
            )
        
        return summary
    
    def _generate_validation_plots(self, fire_df, labels, coords):
        """Generate validation visualization plots"""
        
        self.logger.info("Generating validation plots")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster size distribution
        ax1 = axes[0, 0]
        cluster_sizes = pd.Series(labels[labels >= 0]).value_counts()
        cluster_sizes.hist(bins=50, ax=ax1, color='steelblue', edgecolor='black')
        ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cluster Size (number of detections)')
        ax1.set_ylabel('Frequency')
        ax1.set_yscale('log')
        
        # 2. Spatial distribution of clusters
        ax2 = axes[0, 1]
        # Sample clusters for visualization
        unique_clusters = np.unique(labels[labels >= 0])
        n_sample = min(20, len(unique_clusters))
        sample_clusters = np.random.choice(unique_clusters, n_sample, replace=False)
        
        colors = plt.cm.tab20(np.linspace(0, 1, n_sample))
        
        for i, cluster_id in enumerate(sample_clusters):
            mask = labels == cluster_id
            ax2.scatter(coords[mask, 0]/1000, coords[mask, 1]/1000, 
                       c=[colors[i]], s=10, alpha=0.6, label=f'Cluster {cluster_id}')
        
        # Plot noise points
        noise_mask = labels == -1
        if noise_mask.sum() > 0:
            ax2.scatter(coords[noise_mask, 0]/1000, coords[noise_mask, 1]/1000, 
                       c='gray', s=1, alpha=0.3, label='Noise')
        
        ax2.set_title('Sample Clusters - Spatial Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X (km)')
        ax2.set_ylabel('Y (km)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Temporal distribution
        ax3 = axes[1, 0]
        fire_df_labeled = fire_df.copy()
        fire_df_labeled['cluster_id'] = labels
        
        # Daily cluster counts
        daily_clusters = fire_df_labeled[fire_df_labeled['cluster_id'] >= 0].groupby(
            ['acq_date', 'cluster_id']
        ).size().reset_index()
        
        daily_unique_clusters = daily_clusters.groupby('acq_date')['cluster_id'].nunique()
        daily_unique_clusters.plot(ax=ax3, color='darkred', linewidth=2)
        ax3.set_title('Active Clusters Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Active Clusters')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Cluster quality metrics
        ax4 = axes[1, 1]
        
        # Calculate simple quality metrics for clusters
        quality_data = []
        for cluster_id in unique_clusters[:100]:  # Limit for performance
            cluster_data = fire_df_labeled[fire_df_labeled['cluster_id'] == cluster_id]
            
            if len(cluster_data) > 3:
                # Temporal compactness
                time_span = (cluster_data['acq_datetime'].max() - 
                           cluster_data['acq_datetime'].min()).total_seconds() / 3600
                
                # Spatial compactness
                spatial_std = np.sqrt(
                    cluster_data['x_proj'].std()**2 + 
                    cluster_data['y_proj'].std()**2
                ) / 1000
                
                quality_data.append({
                    'temporal_span_hours': time_span,
                    'spatial_std_km': spatial_std,
                    'size': len(cluster_data)
                })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            scatter = ax4.scatter(quality_df['temporal_span_hours'], 
                                quality_df['spatial_std_km'],
                                s=quality_df['size']*2,
                                c=quality_df['size'],
                                cmap='viridis',
                                alpha=0.6)
            ax4.set_title('Cluster Compactness Analysis', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Temporal Span (hours)')
            ax4.set_ylabel('Spatial Standard Deviation (km)')
            ax4.set_xscale('log')
            ax4.set_yscale('log')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Cluster Size')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.validation_dir / 'clustering_validation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Validation plots saved to {plot_path}")
    
    def generate_validation_report(self, all_results):
        """Generate comprehensive validation report"""
        
        self.logger.info("Generating validation report")
        
        # Save JSON report
        json_path = self.validation_dir / 'validation_report.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate HTML report
        html_path = self.validation_dir / 'validation_report.html'
        self._generate_html_report(all_results, html_path)
        
        # Generate text summary
        text_path = self.validation_dir / 'validation_summary.txt'
        self._generate_text_summary(all_results, text_path)
        
        self.logger.info(f"Validation reports saved to {self.validation_dir}")
    
    def _generate_html_report(self, results, output_path):
        """Generate HTML validation report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fire Episode Clustering Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; }}
                .warning {{ color: #ff6600; }}
                .good {{ color: #00cc00; }}
            </style>
        </head>
        <body>
            <h1>Fire Episode Clustering Validation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Clustering Performance</h2>
            <div class="metric">
                <p><strong>Total Points:</strong> {results.get('clustering_validation', {}).get('clustering_metrics', {}).get('total_points', 'N/A'):,}</p>
                <p><strong>Clusters Found:</strong> {results.get('clustering_validation', {}).get('clustering_metrics', {}).get('n_clusters', 'N/A')}</p>
                <p><strong>Noise Points:</strong> {results.get('clustering_validation', {}).get('clustering_metrics', {}).get('n_noise', 'N/A'):,} 
                   ({results.get('clustering_validation', {}).get('clustering_metrics', {}).get('noise_ratio', 0)*100:.1f}%)</p>
                <p><strong>Silhouette Score:</strong> {results.get('clustering_validation', {}).get('clustering_metrics', {}).get('silhouette_score', 'N/A'):.3f}</p>
            </div>
            
            <h2>Episode Quality Assessment</h2>
            <div class="metric">
                <p><strong>Total Episodes:</strong> {results.get('episode_validation', {}).get('total_episodes', 'N/A')}</p>
                <p><strong>Valid Episodes:</strong> {results.get('episode_validation', {}).get('valid_episodes', 'N/A')}</p>
                <p><strong>Data Quality:</strong> <span class="{('good' if results.get('episode_validation', {}).get('validation_summary', {}).get('data_quality') == 'GOOD' else 'warning')}">
                    {results.get('episode_validation', {}).get('validation_summary', {}).get('data_quality', 'N/A')}</span></p>
            </div>
            
            <h2>Recommendations</h2>
            <ul>
        """
        
        # Add recommendations
        recommendations = results.get('episode_validation', {}).get('validation_summary', {}).get('recommendations', [])
        for rec in recommendations:
            html_content += f"<li>{rec}</li>\n"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_text_summary(self, results, output_path):
        """Generate text summary of validation results"""
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FIRE EPISODE CLUSTERING VALIDATION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Clustering results
            clustering_metrics = results.get('clustering_validation', {}).get('clustering_metrics', {})
            f.write("CLUSTERING RESULTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total fire detections: {clustering_metrics.get('total_points', 'N/A'):,}\n")
            f.write(f"Clusters identified: {clustering_metrics.get('n_clusters', 'N/A')}\n")
            f.write(f"Noise points: {clustering_metrics.get('n_noise', 'N/A'):,} ")
            f.write(f"({clustering_metrics.get('noise_ratio', 0)*100:.1f}%)\n")
            f.write(f"Silhouette score: {clustering_metrics.get('silhouette_score', -1):.3f}\n\n")
            
            # Episode validation
            episode_val = results.get('episode_validation', {})
            f.write("EPISODE VALIDATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Total episodes generated: {episode_val.get('total_episodes', 'N/A')}\n")
            f.write(f"Valid episodes: {episode_val.get('valid_episodes', 'N/A')}\n\n")
            
            # Quality summary
            val_summary = episode_val.get('validation_summary', {})
            f.write("QUALITY ASSESSMENT\n")
            f.write("-"*40 + "\n")
            f.write(f"Data quality: {val_summary.get('data_quality', 'N/A')}\n")
            f.write(f"Spatial coherence: {val_summary.get('spatial_coherence', 'N/A')}\n")
            f.write(f"Temporal consistency: {val_summary.get('temporal_consistency', 'N/A')}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            for i, rec in enumerate(val_summary.get('recommendations', []), 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n" + "="*80 + "\n") 