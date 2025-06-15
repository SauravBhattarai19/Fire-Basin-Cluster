#!/usr/bin/env python3
"""
Visualization script to validate fire episode clustering results
Place this in 04_Integration/ directory
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils import load_config

class FireEpisodeValidator:
    """Validate and visualize fire episode clustering results"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.episodes_dir = self.output_dir / 'episodes'
        self.validation_dir = self.output_dir / 'validation'
        
        # Create visualization directory
        self.viz_dir = self.output_dir / 'validation_visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_results()
        
    def load_results(self):
        """Load fire episodes and watershed statistics"""
        print("Loading results...")
        
        # Load fire episodes
        episode_path = self.episodes_dir / 'fire_episodes.parquet'
        if episode_path.exists():
            self.episodes_df = pd.read_parquet(episode_path)
            print(f"Loaded {len(self.episodes_df)} fire episodes")
        else:
            # Try GeoJSON
            episode_path = self.episodes_dir / 'fire_episodes.geojson'
            if episode_path.exists():
                self.episodes_gdf = gpd.read_file(episode_path)
                self.episodes_df = pd.DataFrame(self.episodes_gdf.drop(columns='geometry'))
                print(f"Loaded {len(self.episodes_df)} fire episodes from GeoJSON")
            else:
                raise FileNotFoundError("No fire episodes file found")
        
        # Load watershed statistics
        watershed_path = self.episodes_dir / 'watershed_fire_statistics.geojson'
        if watershed_path.exists():
            self.watersheds_gdf = gpd.read_file(watershed_path)
            print(f"Loaded {len(self.watersheds_gdf)} watersheds")
            
            # Count watersheds with fires
            fire_cols = ['episode_count', 'total_energy_mwh']
            for col in fire_cols:
                if col in self.watersheds_gdf.columns:
                    watersheds_with_fires = self.watersheds_gdf[
                        self.watersheds_gdf[col].notna() & (self.watersheds_gdf[col] > 0)
                    ]
                    print(f"Watersheds with fires: {len(watersheds_with_fires)}")
                    break
        else:
            print("Warning: No watershed statistics file found")
            self.watersheds_gdf = None
    
    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard"""
        print("Creating overview dashboard...")
        
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Episode Duration Distribution
        ax1 = plt.subplot(3, 4, 1)
        if 'duration_days' in self.episodes_df.columns:
            self.episodes_df['duration_days'].hist(bins=50, ax=ax1, color='darkred', alpha=0.7)
            ax1.set_title('Fire Episode Duration Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Duration (days)')
            ax1.set_ylabel('Number of Episodes')
            ax1.axvline(self.episodes_df['duration_days'].median(), color='red', 
                       linestyle='--', label=f'Median: {self.episodes_df["duration_days"].median():.1f} days')
            ax1.legend()
        
        # 2. Fire Size Distribution (Area)
        ax2 = plt.subplot(3, 4, 2)
        if 'areasqm' in self.episodes_df.columns:
            area_data = self.episodes_df['areasqm'][self.episodes_df['areasqm'] > 0]
            if len(area_data) > 0:
                area_data.hist(bins=50, ax=ax2, color='orange', alpha=0.7)
                ax2.set_title('Fire Episode Area Distribution', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Area (km²)')
                ax2.set_ylabel('Number of Episodes')
                ax2.set_xscale('log')
        
        # 3. Temporal Pattern - Episodes over Time
        ax3 = plt.subplot(3, 4, 3)
        if 'start_datetime' in self.episodes_df.columns:
            # Convert to datetime if string
            start_times = pd.to_datetime(self.episodes_df['start_datetime'])
            
            # Monthly episode counts
            monthly_counts = start_times.dt.to_period('M').value_counts().sort_index()
            monthly_counts.plot(kind='bar', ax=ax3, color='darkgreen', alpha=0.7)
            ax3.set_title('Monthly Fire Episode Frequency', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Number of Episodes')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Fire Intensity Distribution (FRP)
        ax4 = plt.subplot(3, 4, 4)
        if 'peak_frp' in self.episodes_df.columns:
            self.episodes_df['peak_frp'].hist(bins=50, ax=ax4, color='red', alpha=0.7)
            ax4.set_title('Peak Fire Radiative Power Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Peak FRP (MW)')
            ax4.set_ylabel('Number of Episodes')
            ax4.set_yscale('log')
        
        # 5. Geographic Distribution of Episodes
        ax5 = plt.subplot(3, 4, 5)
        if 'centroid_lon' in self.episodes_df.columns and 'centroid_lat' in self.episodes_df.columns:
            scatter = ax5.scatter(self.episodes_df['centroid_lon'], 
                                self.episodes_df['centroid_lat'],
                                c=self.episodes_df['total_energy_mwh'] if 'total_energy_mwh' in self.episodes_df.columns else 'red',
                                cmap='YlOrRd', s=20, alpha=0.6)
            ax5.set_title('Fire Episode Locations', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Longitude')
            ax5.set_ylabel('Latitude')
            if 'total_energy_mwh' in self.episodes_df.columns:
                cbar = plt.colorbar(scatter, ax=ax5)
                cbar.set_label('Total Energy (MWh)')
        
        # 6. Day vs Night Activity
        ax6 = plt.subplot(3, 4, 6)
        if 'day_night_ratio' in self.episodes_df.columns:
            # Create categories
            day_dominant = (self.episodes_df['day_night_ratio'] > 0.6).sum()
            night_dominant = (self.episodes_df['day_night_ratio'] < 0.4).sum()
            mixed = len(self.episodes_df) - day_dominant - night_dominant
            
            labels = ['Day Dominant', 'Night Dominant', 'Mixed']
            sizes = [day_dominant, night_dominant, mixed]
            colors = ['gold', 'darkblue', 'gray']
            
            ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax6.set_title('Fire Activity Patterns', fontsize=12, fontweight='bold')
        
        # 7. Watershed Fire Frequency
        if self.watersheds_gdf is not None and 'episode_count' in self.watersheds_gdf.columns:
            ax7 = plt.subplot(3, 4, 7)
            
            # Filter watersheds with fires
            watersheds_with_fires = self.watersheds_gdf[
                self.watersheds_gdf['episode_count'].notna() & 
                (self.watersheds_gdf['episode_count'] > 0)
            ]['episode_count']
            
            if len(watersheds_with_fires) > 0:
                watersheds_with_fires.hist(bins=30, ax=ax7, color='steelblue', alpha=0.7)
                ax7.set_title('Fire Episodes per Watershed', fontsize=12, fontweight='bold')
                ax7.set_xlabel('Number of Episodes')
                ax7.set_ylabel('Number of Watersheds')
                ax7.set_yscale('log')
        
        # 8. Fire Spread Rate Distribution
        ax8 = plt.subplot(3, 4, 8)
        if 'max_spread_rate_kmh' in self.episodes_df.columns:
            spread_rates = self.episodes_df['max_spread_rate_kmh'][
                (self.episodes_df['max_spread_rate_kmh'] > 0) & 
                (self.episodes_df['max_spread_rate_kmh'] < 50)  # Filter outliers
            ]
            if len(spread_rates) > 0:
                spread_rates.hist(bins=40, ax=ax8, color='purple', alpha=0.7)
                ax8.set_title('Fire Spread Rate Distribution', fontsize=12, fontweight='bold')
                ax8.set_xlabel('Max Spread Rate (km/h)')
                ax8.set_ylabel('Number of Episodes')
        
        # 9. Quality Scores
        ax9 = plt.subplot(3, 4, 9)
        if 'spatial_coherence_score' in self.episodes_df.columns:
            quality_metrics = ['spatial_coherence_score', 'data_completeness_score', 'mean_confidence']
            quality_data = []
            
            for metric in quality_metrics:
                if metric in self.episodes_df.columns:
                    quality_data.append(self.episodes_df[metric].mean())
            
            if quality_data:
                bars = ax9.bar(range(len(quality_data)), quality_data, 
                              color=['green', 'blue', 'orange'], alpha=0.7)
                ax9.set_xticks(range(len(quality_data)))
                ax9.set_xticklabels(['Spatial\nCoherence', 'Data\nCompleteness', 'Mean\nConfidence'])
                ax9.set_title('Average Quality Metrics', fontsize=12, fontweight='bold')
                ax9.set_ylabel('Score')
                ax9.set_ylim(0, 100)
                
                # Add value labels
                for bar, val in zip(bars, quality_data):
                    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{val:.1f}', ha='center', va='bottom')
        
        # 10. Episode Type Distribution
        ax10 = plt.subplot(3, 4, 10)
        if 'episode_type' in self.episodes_df.columns:
            type_counts = self.episodes_df['episode_type'].value_counts()
            type_counts.plot(kind='bar', ax=ax10, color='darkgreen', alpha=0.7)
            ax10.set_title('Fire Episode Types', fontsize=12, fontweight='bold')
            ax10.set_xlabel('Episode Type')
            ax10.set_ylabel('Count')
            ax10.tick_params(axis='x', rotation=45)
        
        # 11. Summary Statistics
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        summary_text = f"""
        FIRE EPISODE SUMMARY STATISTICS
        
        Total Episodes: {len(self.episodes_df):,}
        Valid Episodes: {self.episodes_df['is_valid'].sum() if 'is_valid' in self.episodes_df.columns else 'N/A'}
        
        Duration:
        • Mean: {self.episodes_df['duration_days'].mean():.1f} days
        • Max: {self.episodes_df['duration_days'].max():.1f} days
        
        Area:
        • Mean: {self.episodes_df['areasqm'].mean():.1f} km²
        • Total: {self.episodes_df['areasqm'].sum():.0f} km²
        
        Energy:
        • Total: {self.episodes_df['total_energy_mwh'].sum():.0f} MWh
        • Mean per episode: {self.episodes_df['total_energy_mwh'].mean():.1f} MWh
        """
        
        if self.watersheds_gdf is not None and 'episode_count' in self.watersheds_gdf.columns:
            watersheds_affected = (self.watersheds_gdf['episode_count'] > 0).sum()
            summary_text += f"\nWatersheds Affected: {watersheds_affected:,} of {len(self.watersheds_gdf):,}"
        
        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 12. Watershed Fire Intensity Map
        if self.watersheds_gdf is not None and 'total_energy_mwh' in self.watersheds_gdf.columns:
            ax12 = plt.subplot(3, 4, 12)
            
            # Plot watersheds colored by total fire energy
            watersheds_with_data = self.watersheds_gdf[
                self.watersheds_gdf['total_energy_mwh'].notna() & 
                (self.watersheds_gdf['total_energy_mwh'] > 0)
            ]
            
            if len(watersheds_with_data) > 0:
                watersheds_with_data.plot(ax=ax12, column='total_energy_mwh', 
                                        cmap='YlOrRd', legend=True,
                                        legend_kwds={'label': 'Total Energy (MWh)', 'shrink': 0.8})
                ax12.set_title('Watershed Fire Intensity', fontsize=12, fontweight='bold')
                ax12.set_xlabel('Longitude')
                ax12.set_ylabel('Latitude')
        
        plt.suptitle(f'Fire Episode Clustering Validation Dashboard\n{self.output_dir.name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'overview_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved overview dashboard to {output_path}")
        plt.close()
    
    def create_temporal_analysis(self):
        """Create detailed temporal analysis plots"""
        print("Creating temporal analysis...")
        
        if 'start_datetime' not in self.episodes_df.columns:
            print("No temporal data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Convert to datetime
        self.episodes_df['start_datetime'] = pd.to_datetime(self.episodes_df['start_datetime'])
        self.episodes_df['end_datetime'] = pd.to_datetime(self.episodes_df['end_datetime'])
        
        # 1. Timeline of episodes
        ax1 = axes[0, 0]
        
        # Sample episodes for visibility
        n_episodes = min(100, len(self.episodes_df))
        sample_episodes = self.episodes_df.nlargest(n_episodes, 'total_energy_mwh')
        
        for idx, episode in sample_episodes.iterrows():
            ax1.plot([episode['start_datetime'], episode['end_datetime']], 
                    [idx, idx], linewidth=2, alpha=0.7)
        
        ax1.set_title('Timeline of Major Fire Episodes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Episode (sorted by energy)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Seasonal patterns
        ax2 = axes[0, 1]
        
        self.episodes_df['month'] = self.episodes_df['start_datetime'].dt.month
        seasonal_energy = self.episodes_df.groupby('month')['total_energy_mwh'].sum()
        
        seasonal_energy.plot(kind='bar', ax=ax2, color='orange', alpha=0.7)
        ax2.set_title('Total Fire Energy by Month', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Total Energy (MWh)')
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # 3. Active episodes over time
        ax3 = axes[1, 0]
        
        # Create daily time series
        date_range = pd.date_range(start=self.episodes_df['start_datetime'].min(),
                                  end=self.episodes_df['end_datetime'].max(),
                                  freq='D')
        
        active_episodes = []
        for date in date_range:
            active = ((self.episodes_df['start_datetime'] <= date) & 
                     (self.episodes_df['end_datetime'] >= date)).sum()
            active_episodes.append(active)
        
        ts_df = pd.DataFrame({'date': date_range, 'active_episodes': active_episodes})
        ts_df.set_index('date')['active_episodes'].plot(ax=ax3, color='darkred', linewidth=1)
        
        ax3.set_title('Active Fire Episodes Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Active Episodes')
        ax3.grid(True, alpha=0.3)
        
        # 4. Duration vs Start Date
        ax4 = axes[1, 1]
        
        scatter = ax4.scatter(self.episodes_df['start_datetime'], 
                            self.episodes_df['duration_days'],
                            c=self.episodes_df['total_energy_mwh'],
                            cmap='YlOrRd', s=30, alpha=0.6)
        
        ax4.set_title('Episode Duration vs Start Date', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Start Date')
        ax4.set_ylabel('Duration (days)')
        ax4.set_yscale('log')
        
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Total Energy (MWh)')
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'temporal_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal analysis to {output_path}")
        plt.close()
    
    def create_spatial_analysis(self):
        """Create spatial analysis maps"""
        print("Creating spatial analysis...")
        
        if self.watersheds_gdf is None:
            print("No watershed data available for spatial analysis")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        
        # 1. Episode count per watershed
        ax1 = axes[0, 0]
        if 'episode_count' in self.watersheds_gdf.columns:
            self.watersheds_gdf.plot(ax=ax1, column='episode_count', 
                                   cmap='YlOrRd', legend=True,
                                   legend_kwds={'label': 'Number of Episodes', 'shrink': 0.8},
                                   missing_kwds={'color': 'lightgray'})
            ax1.set_title('Fire Episodes per Watershed', fontsize=14, fontweight='bold')
        
        # 2. Total burned area per watershed
        ax2 = axes[0, 1]
        if 'total_burned_area_km2' in self.watersheds_gdf.columns:
            self.watersheds_gdf.plot(ax=ax2, column='total_burned_area_km2', 
                                   cmap='YlOrRd', legend=True,
                                   legend_kwds={'label': 'Total Area (km²)', 'shrink': 0.8},
                                   missing_kwds={'color': 'lightgray'})
            ax2.set_title('Total Burned Area per Watershed', fontsize=14, fontweight='bold')
        
        # 3. Mean fire intensity
        ax3 = axes[1, 0]
        if 'watershed_mean_frp' in self.watersheds_gdf.columns:
            self.watersheds_gdf.plot(ax=ax3, column='watershed_mean_frp', 
                                   cmap='hot', legend=True,
                                   legend_kwds={'label': 'Mean FRP (MW)', 'shrink': 0.8},
                                   missing_kwds={'color': 'lightgray'})
            ax3.set_title('Mean Fire Intensity per Watershed', fontsize=14, fontweight='bold')
        
        # 4. HSBF (Hydrologically Significant Burn Fraction)
        ax4 = axes[1, 1]
        if 'hsbf' in self.watersheds_gdf.columns:
            # Create categories for HSBF
            watersheds_with_hsbf = self.watersheds_gdf[self.watersheds_gdf['hsbf'] > 0].copy()
            watersheds_with_hsbf['hsbf_category'] = pd.cut(
                watersheds_with_hsbf['hsbf'],
                bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
                labels=['<10%', '10-30%', '30-50%', '50-70%', '>70%']
            )
            
            self.watersheds_gdf.plot(ax=ax4, color='lightgray')
            watersheds_with_hsbf.plot(ax=ax4, column='hsbf_category', 
                                     cmap='RdYlBu_r', legend=True,
                                     legend_kwds={'title': 'HSBF (%)', 'loc': 'lower right'})
            ax4.set_title('Hydrologically Significant Burn Fraction', fontsize=14, fontweight='bold')
        
        # 5. High severity episodes
        ax5 = axes[2, 0]
        if 'n_high_severity' in self.watersheds_gdf.columns:
            self.watersheds_gdf.plot(ax=ax5, column='n_high_severity', 
                                   cmap='Reds', legend=True,
                                   legend_kwds={'label': 'High Severity Episodes', 'shrink': 0.8},
                                   missing_kwds={'color': 'lightgray'})
            ax5.set_title('High Severity Fire Episodes per Watershed', fontsize=14, fontweight='bold')
        
        # 6. Significant burns (>30% of watershed area)
        ax6 = axes[2, 1]
        if 'n_30pct_burns' in self.watersheds_gdf.columns:
            self.watersheds_gdf.plot(ax=ax6, column='n_30pct_burns', 
                                   cmap='OrRd', legend=True,
                                   legend_kwds={'label': 'Episodes >30% of Area', 'shrink': 0.8},
                                   missing_kwds={'color': 'lightgray'})
            ax6.set_title('Significant Burns (>30% of Watershed Area)', fontsize=14, fontweight='bold')
        
        # Add episode points overlay on first map
        if 'centroid_lon' in self.episodes_df.columns:
            ax1.scatter(self.episodes_df['centroid_lon'], 
                       self.episodes_df['centroid_lat'],
                       c='red', s=5, alpha=0.5, label='Episode Centers')
            ax1.legend()
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'spatial_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved spatial analysis to {output_path}")
        plt.close()
    
    def create_quality_report(self):
        """Create quality assessment report"""
        print("Creating quality assessment report...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Data quality metrics distribution
        ax1 = axes[0, 0]
        quality_cols = ['spatial_coherence_score', 'data_completeness_score', 'mean_confidence']
        quality_data = []
        
        for col in quality_cols:
            if col in self.episodes_df.columns:
                quality_data.append(self.episodes_df[col])
        
        if quality_data:
            ax1.boxplot(quality_data, labels=['Spatial\nCoherence', 'Data\nCompleteness', 'Mean\nConfidence'])
            ax1.set_title('Episode Quality Metrics Distribution', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.grid(True, axis='y', alpha=0.3)
        
        # 2. Detection count vs duration
        ax2 = axes[0, 1]
        if 'detection_count' in self.episodes_df.columns and 'duration_days' in self.episodes_df.columns:
            ax2.scatter(self.episodes_df['duration_days'], 
                       self.episodes_df['detection_count'],
                       alpha=0.5, s=20)
            ax2.set_title('Detection Count vs Episode Duration', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Duration (days)')
            ax2.set_ylabel('Number of Detections')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            
            # Add expected line (assuming ~4 detections per day)
            x_range = np.logspace(np.log10(0.1), np.log10(self.episodes_df['duration_days'].max()), 100)
            expected_y = 4 * x_range
            ax2.plot(x_range, expected_y, 'r--', alpha=0.5, label='Expected (4/day)')
            ax2.legend()
        
        # 3. Valid vs invalid episodes
        ax3 = axes[0, 2]
        if 'is_valid' in self.episodes_df.columns:
            valid_counts = self.episodes_df['is_valid'].value_counts()
            valid_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%', 
                            labels=['Invalid', 'Valid'], colors=['red', 'green'])
            ax3.set_title('Episode Validation Status', fontsize=12, fontweight='bold')
        
        # 4. Confidence distribution
        ax4 = axes[1, 0]
        if 'mean_confidence' in self.episodes_df.columns:
            self.episodes_df['mean_confidence'].hist(bins=30, ax=ax4, color='blue', alpha=0.7)
            ax4.set_title('Mean Confidence Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Mean Confidence (%)')
            ax4.set_ylabel('Number of Episodes')
            ax4.axvline(70, color='red', linestyle='--', label='Quality Threshold')
            ax4.legend()
        
        # 5. Satellite coverage
        ax5 = axes[1, 1]
        if 'satellites_list' in self.episodes_df.columns:
            # Count satellite combinations
            sat_counts = self.episodes_df['satellites_list'].value_counts()
            sat_counts.head(10).plot(kind='bar', ax=ax5, color='purple', alpha=0.7)
            ax5.set_title('Satellite Coverage', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Satellite Combination')
            ax5.set_ylabel('Number of Episodes')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Quality summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        quality_summary = f"""
        QUALITY ASSESSMENT SUMMARY
        
        Total Episodes: {len(self.episodes_df):,}
        """
        
        if 'is_valid' in self.episodes_df.columns:
            quality_summary += f"\nValid Episodes: {self.episodes_df['is_valid'].sum():,} ({self.episodes_df['is_valid'].mean()*100:.1f}%)"
        
        if 'spatial_coherence_score' in self.episodes_df.columns:
            quality_summary += f"\n\nSpatial Coherence:"
            quality_summary += f"\n• Mean: {self.episodes_df['spatial_coherence_score'].mean():.3f}"
            quality_summary += f"\n• High (>0.7): {(self.episodes_df['spatial_coherence_score'] > 0.7).sum():,}"
        
        if 'data_completeness_score' in self.episodes_df.columns:
            quality_summary += f"\n\nData Completeness:"
            quality_summary += f"\n• Mean: {self.episodes_df['data_completeness_score'].mean():.3f}"
            quality_summary += f"\n• High (>0.5): {(self.episodes_df['data_completeness_score'] > 0.5).sum():,}"
        
        if 'mean_confidence' in self.episodes_df.columns:
            quality_summary += f"\n\nConfidence Level:"
            quality_summary += f"\n• Mean: {self.episodes_df['mean_confidence'].mean():.1f}%"
            quality_summary += f"\n• High (>80%): {(self.episodes_df['mean_confidence'] > 80).sum():,}"
        
        ax6.text(0.05, 0.95, quality_summary, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'quality_report.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved quality report to {output_path}")
        plt.close()
    
    def create_sample_episodes_detail(self, n_samples=9):
        """Create detailed view of sample episodes"""
        print(f"Creating detailed view of {n_samples} sample episodes...")
        
        # Select diverse sample of episodes
        sample_episodes = self._select_diverse_episodes(n_samples)
        
        if len(sample_episodes) == 0:
            print("No episodes to visualize")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        
        for idx, (_, episode) in enumerate(sample_episodes.iterrows()):
            if idx >= n_samples:
                break
                
            ax = axes[idx]
            
            # Create episode summary text
            area_value = episode.get('areasqm', episode.get('area_km2', 0))
            duration_days = episode.get('duration_days', 0)
            total_energy = episode.get('total_energy_mwh', 0)
            peak_frp = episode.get('peak_frp', 0)
            detection_count = episode.get('detection_count', 0)
            episode_type = episode.get('episode_type', 'N/A')
            is_valid = episode.get('is_valid', True)
            
            summary = f"""Episode {episode['episode_id']}
Duration: {duration_days:.1f} days
Area: {area_value:.1f} km²
Energy: {total_energy:.0f} MWh
Peak FRP: {peak_frp:.0f} MW
Detections: {detection_count:.0f}
Type: {episode_type}
Valid: {'Yes' if is_valid else 'No'}"""
            
            # Add quality metrics if available
            if 'spatial_coherence_score' in episode:
                summary += f"\nSpatial Coherence: {episode['spatial_coherence_score']:.2f}"
            if 'data_completeness_score' in episode:
                summary += f"\nData Completeness: {episode['data_completeness_score']:.2f}"
            
            ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Add simple visualization (could be enhanced with actual fire point data)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            
            # Draw a circle representing the fire
            # Use 'areasqm' instead of 'area_km2' and handle missing columns gracefully
            circle_size = min(0.4, max(0.05, area_value / 1000))  # Scale by area with min/max bounds
            
            # Calculate alpha value safely
            alpha_value = min(0.8, max(0.3, 0.3 + 0.5 * (peak_frp / 1000)))
            
            circle = plt.Circle((0.5, 0.4), circle_size, 
                              color='red', alpha=alpha_value)
            ax.add_patch(circle)
            
            ax.set_title(f"Episode {episode['episode_id']}", fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(sample_episodes), n_samples):
            axes[idx].axis('off')
        
        plt.suptitle('Sample Fire Episodes - Detailed View', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'sample_episodes_detail.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved sample episodes detail to {output_path}")
        plt.close()
    
    def _select_diverse_episodes(self, n_samples):
        """Select diverse episodes for visualization"""
        if len(self.episodes_df) <= n_samples:
            return self.episodes_df
        
        # Try to get diverse samples based on different criteria
        samples = []
        
        # Largest by area
        if 'areasqm' in self.episodes_df.columns:
            samples.extend(self.episodes_df.nlargest(n_samples//3, 'areasqm').index)
        
        # Longest duration
        if 'duration_days' in self.episodes_df.columns:
            samples.extend(self.episodes_df.nlargest(n_samples//3, 'duration_days').index)
        
        # Highest energy
        if 'total_energy_mwh' in self.episodes_df.columns:
            samples.extend(self.episodes_df.nlargest(n_samples//3, 'total_energy_mwh').index)
        
        # Remove duplicates and limit to n_samples
        unique_samples = list(dict.fromkeys(samples))[:n_samples]
        
        # If not enough, add random samples
        if len(unique_samples) < n_samples:
            remaining = n_samples - len(unique_samples)
            additional = self.episodes_df.index.difference(unique_samples).tolist()
            if additional:
                unique_samples.extend(np.random.choice(additional, 
                                                      min(remaining, len(additional)), 
                                                      replace=False))
        
        return self.episodes_df.loc[unique_samples]
    
    def create_threshold_analysis(self):
        """Create threshold exceedance analysis visualizations"""
        print("Creating threshold exceedance analysis...")
        
        if self.watersheds_gdf is None:
            print("No watershed data available for threshold analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Bar chart of watersheds exceeding each threshold
        ax1 = axes[0, 0]
        threshold_columns = [col for col in self.watersheds_gdf.columns if col.startswith('n_') and col.endswith('pct_burns')]
        
        if threshold_columns:
            threshold_data = []
            threshold_labels = []
            
            for col in sorted(threshold_columns):
                threshold_pct = col.replace('n_', '').replace('pct_burns', '')
                watersheds_exceeding = (self.watersheds_gdf[col] > 0).sum()
                threshold_data.append(watersheds_exceeding)
                threshold_labels.append(f"{threshold_pct}%")
            
            bars = ax1.bar(threshold_labels, threshold_data, color='darkred', alpha=0.7)
            ax1.set_title('Watersheds with Burns Exceeding Area Thresholds', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Area Threshold (% of watershed)')
            ax1.set_ylabel('Number of Watersheds')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, threshold_data):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:,}', ha='center', va='bottom')
        
        # 2. HSBF distribution histogram
        ax2 = axes[0, 1]
        if 'hsbf' in self.watersheds_gdf.columns:
            hsbf_values = self.watersheds_gdf[self.watersheds_gdf['hsbf'] > 0]['hsbf'] * 100  # Convert to percentage
            
            if len(hsbf_values) > 0:
                hsbf_values.hist(bins=30, ax=ax2, color='orange', alpha=0.7, edgecolor='black')
                ax2.set_title('Distribution of HSBF Values', fontsize=14, fontweight='bold')
                ax2.set_xlabel('HSBF (%)')
                ax2.set_ylabel('Number of Watersheds')
                
                # Add statistics
                ax2.axvline(hsbf_values.mean(), color='red', linestyle='--', 
                           label=f'Mean: {hsbf_values.mean():.1f}%')
                ax2.axvline(hsbf_values.median(), color='blue', linestyle='--', 
                           label=f'Median: {hsbf_values.median():.1f}%')
                ax2.legend()
        
        # 3. Stacked bar chart showing episode counts by threshold
        ax3 = axes[1, 0]
        if threshold_columns:
            threshold_episode_data = []
            
            for col in sorted(threshold_columns):
                total_episodes = self.watersheds_gdf[col].sum()
                threshold_episode_data.append(total_episodes)
            
            ax3.bar(threshold_labels, threshold_episode_data, color='darkgreen', alpha=0.7)
            ax3.set_title('Total Episodes by Area Threshold', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Area Threshold (% of watershed)')
            ax3.set_ylabel('Total Number of Episodes')
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Scatter plot: HSBF vs total episodes
        ax4 = axes[1, 1]
        if 'hsbf' in self.watersheds_gdf.columns and 'episode_count' in self.watersheds_gdf.columns:
            watersheds_with_data = self.watersheds_gdf[
                (self.watersheds_gdf['hsbf'] > 0) & 
                (self.watersheds_gdf['episode_count'] > 0)
            ]
            
            if len(watersheds_with_data) > 0:
                scatter = ax4.scatter(watersheds_with_data['episode_count'], 
                                    watersheds_with_data['hsbf'] * 100,
                                    c=watersheds_with_data.get('n_high_severity', 0),
                                    cmap='YlOrRd', s=50, alpha=0.6)
                
                ax4.set_title('HSBF vs Episode Count', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Total Episodes')
                ax4.set_ylabel('HSBF (%)')
                ax4.set_xscale('log')
                
                if 'n_high_severity' in watersheds_with_data.columns:
                    cbar = plt.colorbar(scatter, ax=ax4)
                    cbar.set_label('High Severity Episodes')
        
        plt.suptitle('Threshold Exceedance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'threshold_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved threshold analysis to {output_path}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate text summary report"""
        print("Generating summary report...")
        
        report_path = self.viz_dir / 'validation_summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FIRE EPISODE CLUSTERING VALIDATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # Episode statistics
            f.write("FIRE EPISODE STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total episodes: {len(self.episodes_df):,}\n")
            
            if 'is_valid' in self.episodes_df.columns:
                f.write(f"Valid episodes: {self.episodes_df['is_valid'].sum():,} "
                       f"({self.episodes_df['is_valid'].mean()*100:.1f}%)\n")
            
            # Temporal statistics
            if 'duration_days' in self.episodes_df.columns:
                f.write(f"\nDuration:\n")
                f.write(f"  Mean: {self.episodes_df['duration_days'].mean():.1f} days\n")
                f.write(f"  Median: {self.episodes_df['duration_days'].median():.1f} days\n")
                f.write(f"  Max: {self.episodes_df['duration_days'].max():.1f} days\n")
            
            # Spatial statistics
            if 'areasqm' in self.episodes_df.columns:
                f.write(f"\nArea:\n")
                f.write(f"  Mean: {self.episodes_df['areasqm'].mean():.1f} km²\n")
                f.write(f"  Total: {self.episodes_df['areasqm'].sum():.0f} km²\n")
            
            # Intensity statistics
            if 'total_energy_mwh' in self.episodes_df.columns:
                f.write(f"\nEnergy:\n")
                f.write(f"  Total: {self.episodes_df['total_energy_mwh'].sum():.0f} MWh\n")
                f.write(f"  Mean per episode: {self.episodes_df['total_energy_mwh'].mean():.1f} MWh\n")
            
            # Watershed statistics
            if self.watersheds_gdf is not None:
                f.write(f"\n\nWATERSHED STATISTICS\n")
                f.write("-"*40 + "\n")
                f.write(f"Total watersheds: {len(self.watersheds_gdf):,}\n")
                
                if 'episode_count' in self.watersheds_gdf.columns:
                    watersheds_affected = (self.watersheds_gdf['episode_count'] > 0).sum()
                    f.write(f"Watersheds with fires: {watersheds_affected:,} "
                           f"({watersheds_affected/len(self.watersheds_gdf)*100:.1f}%)\n")
                    
                    # Top watersheds by fire activity
                    top_watersheds = self.watersheds_gdf.nlargest(10, 'episode_count')
                    f.write(f"\nTop 10 watersheds by episode count:\n")
                    for _, ws in top_watersheds.iterrows():
                        f.write(f"  HUC12 {ws.get('huc12', 'N/A')}: "
                               f"{ws['episode_count']:.0f} episodes\n")
                
                # HSBF statistics
                if 'hsbf' in self.watersheds_gdf.columns:
                    f.write(f"\n\nHYDROLOGICALLY SIGNIFICANT BURN FRACTION (HSBF)\n")
                    f.write("-"*40 + "\n")
                    watersheds_with_burns = self.watersheds_gdf[self.watersheds_gdf['hsbf'] > 0]
                    f.write(f"Watersheds with burns: {len(watersheds_with_burns):,}\n")
                    f.write(f"Max HSBF: {self.watersheds_gdf['hsbf'].max():.3f} ({self.watersheds_gdf['hsbf'].max()*100:.1f}%)\n")
                    f.write(f"Mean HSBF (where > 0): {watersheds_with_burns['hsbf'].mean():.3f} ({watersheds_with_burns['hsbf'].mean()*100:.1f}%)\n")
                    
                    # HSBF categories
                    hsbf_categories = [
                        (0.5, ">50% burned"),
                        (0.3, ">30% burned"),
                        (0.2, ">20% burned"),
                        (0.1, ">10% burned")
                    ]
                    
                    f.write(f"\nWatersheds by burn severity:\n")
                    for threshold, label in hsbf_categories:
                        count = (self.watersheds_gdf['hsbf'] > threshold).sum()
                        pct = count / len(self.watersheds_gdf) * 100
                        f.write(f"  {label}: {count:,} watersheds ({pct:.2f}%)\n")
                
                # Threshold exceedance metrics
                threshold_columns = [col for col in self.watersheds_gdf.columns if col.startswith('n_') and col.endswith('pct_burns')]
                if threshold_columns:
                    f.write(f"\n\nTHRESHOLD EXCEEDANCE METRICS\n")
                    f.write("-"*40 + "\n")
                    
                    for col in sorted(threshold_columns):
                        threshold_pct = col.replace('n_', '').replace('pct_burns', '')
                        watersheds_with_exceedance = (self.watersheds_gdf[col] > 0).sum()
                        total_episodes = self.watersheds_gdf[col].sum()
                        f.write(f"{threshold_pct}% threshold: {watersheds_with_exceedance:,} watersheds with {total_episodes:.0f} total episodes\n")
                
                # High severity episodes
                if 'n_high_severity' in self.watersheds_gdf.columns:
                    f.write(f"\n\nHIGH SEVERITY EPISODES\n")
                    f.write("-"*40 + "\n")
                    watersheds_with_high_severity = (self.watersheds_gdf['n_high_severity'] > 0).sum()
                    total_high_severity = self.watersheds_gdf['n_high_severity'].sum()
                    f.write(f"Watersheds with high severity episodes: {watersheds_with_high_severity:,}\n")
                    f.write(f"Total high severity episodes: {total_high_severity:.0f}\n")
                    
                    # Top watersheds by high severity
                    if watersheds_with_high_severity > 0:
                        top_high_severity = self.watersheds_gdf.nlargest(5, 'n_high_severity')
                        f.write(f"\nTop 5 watersheds by high severity episodes:\n")
                        for _, ws in top_high_severity.iterrows():
                            if ws['n_high_severity'] > 0:
                                f.write(f"  HUC12 {ws.get('huc12', 'N/A')}: {ws['n_high_severity']:.0f} episodes\n")
            
            # Quality assessment
            f.write(f"\n\nQUALITY ASSESSMENT\n")
            f.write("-"*40 + "\n")
            
            quality_metrics = ['spatial_coherence_score', 'data_completeness_score', 'mean_confidence']
            for metric in quality_metrics:
                if metric in self.episodes_df.columns:
                    f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Mean: {self.episodes_df[metric].mean():.3f}\n")
                    f.write(f"  Std: {self.episodes_df[metric].std():.3f}\n")
                    
                    if metric == 'mean_confidence':
                        f.write(f"  Episodes >80%: {(self.episodes_df[metric] > 80).sum():,}\n")
                    else:
                        f.write(f"  Episodes >0.7: {(self.episodes_df[metric] > 0.7).sum():,}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Saved summary report to {report_path}")
    
    def run_all_validations(self):
        """Run all validation visualizations"""
        print("\n" + "="*60)
        print("RUNNING FIRE EPISODE VALIDATION")
        print("="*60)
        
        # Create all visualizations
        self.create_overview_dashboard()
        self.create_temporal_analysis()
        self.create_spatial_analysis()
        self.create_quality_report()
        self.create_sample_episodes_detail()
        self.create_threshold_analysis()
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print(f"All visualizations saved to: {self.viz_dir}")
        print("\nGenerated files:")
        for file in self.viz_dir.glob('*.png'):
            print(f"  • {file.name}")
        print(f"  • validation_summary_report.txt")


def main():
    """Main function to run validation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate fire episode clustering results"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to the output directory from fire_episode_clustering.py'
    )
    
    args = parser.parse_args()
    
    # Check if output directory exists
    output_path = Path(args.output_dir)
    if not output_path.exists():
        print(f"Error: Output directory not found: {output_path}")
        return
    
    # Run validation
    validator = FireEpisodeValidator(output_path)
    validator.run_all_validations()


if __name__ == "__main__":
    # If running standalone
    if len(sys.argv) > 1:
        main()
    else:
        # Example usage
        print("Usage: python validate_results_visualization.py <output_directory>")
        print("\nExample:")
        print("  python validate_results_visualization.py outputs/test_run_20240115_143022")
        
        # Or run on most recent output
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            recent_runs = sorted(outputs_dir.glob("*_run_*"), key=lambda x: x.stat().st_mtime)
            if recent_runs:
                print(f"\nMost recent run found: {recent_runs[-1]}")
                print("Running validation on this directory...")
                validator = FireEpisodeValidator(recent_runs[-1])
                validator.run_all_validations()