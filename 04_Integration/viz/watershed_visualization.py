#!/usr/bin/env python3
"""
Watershed Spatial Visualizations for Research Paper
Creates publication-quality spatial figures showing watershed fire metrics
Place this in 04_Integration/viz/watershed_visualizations.py
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
# Use DejaVu Sans which is available on all systems, or fall back to sans-serif
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

class WatershedSpatialVisualizer:
    """Create research-quality spatial visualizations of watershed fire metrics"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.episodes_dir = self.output_dir / 'episodes'
        
        # Create viz directory
        self.viz_dir = self.output_dir / 'research_figures'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Load watershed data
        self.load_data()
        
        # Define color schemes for different metrics
        self.setup_colormaps()
        
    def load_data(self):
        """Load watershed statistics"""
        print("Loading watershed fire statistics...")
        
        watershed_path = self.episodes_dir / 'watershed_fire_statistics.geojson'
        if not watershed_path.exists():
            raise FileNotFoundError(f"Watershed statistics not found: {watershed_path}")
            
        self.watersheds = gpd.read_file(watershed_path)
        print(f"Loaded {len(self.watersheds)} watersheds")
        
        # Ensure we're in a projected CRS for better visualization
        if self.watersheds.crs and self.watersheds.crs.to_epsg() == 4326:
            # Project to Albers Equal Area for continental US
            self.watersheds = self.watersheds.to_crs('EPSG:5070')
        
    def setup_colormaps(self):
        """Define publication-quality colormaps for different metrics"""
        
        # Fire frequency colormap (white to dark red)
        self.fire_cmap = LinearSegmentedColormap.from_list(
            'fire_frequency',
            ['#ffffff', '#fff5f0', '#fee5d9', '#fcbba1', '#fc9272', 
             '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d']
        )
        
        # Area/intensity colormap (white to orange to red)
        self.intensity_cmap = LinearSegmentedColormap.from_list(
            'fire_intensity',
            ['#ffffff', '#ffffd4', '#fee391', '#fec44f', '#fe9929',
             '#ec7014', '#cc4c02', '#993404', '#662506']
        )
        
        # HSBF colormap (white to purple)
        self.hsbf_cmap = LinearSegmentedColormap.from_list(
            'hsbf',
            ['#ffffff', '#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda',
             '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b']
        )
        
        # Threshold exceedance (white to dark green)
        self.threshold_cmap = LinearSegmentedColormap.from_list(
            'threshold',
            ['#ffffff', '#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b',
             '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        )
        
    def create_figure_1_fire_frequency(self):
        """Figure 1: Fire Episode Frequency and Total Burned Area"""
        print("Creating Figure 1: Fire Episode Frequency...")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Subplot A: Episode Count
            self.watersheds.boundary.plot(ax=ax1, linewidth=0.1, color='gray', alpha=0.3)
            
            # Only plot watersheds with fires
            if 'episode_count' in self.watersheds.columns:
                fire_watersheds = self.watersheds[self.watersheds['episode_count'] > 0]
                
                if len(fire_watersheds) > 0:
                    # Define discrete bins for episode count
                    max_count = fire_watersheds['episode_count'].max()
                    if max_count > 100:
                        bins = [1, 2, 5, 10, 20, 50, 100, int(max_count) + 1]
                    elif max_count > 50:
                        bins = [1, 2, 5, 10, 20, 50, int(max_count) + 1]
                    elif max_count > 20:
                        bins = [1, 2, 5, 10, 20, int(max_count) + 1]
                    elif max_count > 10:
                        bins = [1, 2, 5, 10, int(max_count) + 1]
                    elif max_count > 5:
                        bins = [1, 2, 5, int(max_count) + 1]
                    else:
                        bins = list(range(1, int(max_count) + 2))
                    
                    # Ensure bins are unique and sorted
                    bins = sorted(list(set(bins)))
                    norm = BoundaryNorm(bins, self.fire_cmap.N)
                    
                    fire_watersheds.plot(
                        ax=ax1,
                        column='episode_count',
                        cmap=self.fire_cmap,
                        norm=norm,
                        legend=True,
                        legend_kwds={
                            'label': 'Number of Fire Episodes',
                            'shrink': 0.7,
                            'pad': 0.01,
                            'ticks': bins[:-1],
                            'format': '%.0f'
                        }
                    )
            
            ax1.set_title('(a) Fire Episode Frequency', fontsize=14, fontweight='bold', pad=10)
            ax1.axis('off')
            
            # Add scale bar
            self._add_scale_bar(ax1)
            
            # Subplot B: Total Burned Area
            self.watersheds.boundary.plot(ax=ax2, linewidth=0.1, color='gray', alpha=0.3)
            
            if 'total_burned_area_km2' in self.watersheds.columns:
                area_watersheds = self.watersheds[self.watersheds['total_burned_area_km2'] > 0]
                
                if len(area_watersheds) > 0:
                    # Log scale for area
                    area_watersheds.plot(
                        ax=ax2,
                        column='total_burned_area_km2',
                        cmap=self.intensity_cmap,
                        legend=True,
                        legend_kwds={
                            'label': 'Total Burned Area (km²)',
                            'shrink': 0.7,
                            'pad': 0.01,
                            'format': '%.0f'
                        },
                        norm=plt.matplotlib.colors.LogNorm(
                            vmin=max(0.1, area_watersheds['total_burned_area_km2'].min()),
                            vmax=area_watersheds['total_burned_area_km2'].max()
                        )
                    )
            
            ax2.set_title('(b) Cumulative Burned Area', fontsize=14, fontweight='bold', pad=10)
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Save
            output_path = self.viz_dir / 'Figure_1_Fire_Frequency.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error creating Figure 1: {e}")
            plt.close()
        
    def create_figure_2_fire_intensity(self):
        """Figure 2: Fire Intensity Metrics"""
        print("Creating Figure 2: Fire Intensity Metrics...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Subplot A: Mean Fire Radiative Power
        self.watersheds.boundary.plot(ax=ax1, linewidth=0.1, color='gray', alpha=0.3)
        
        frp_watersheds = self.watersheds[self.watersheds['watershed_mean_frp'] > 0]
        
        if len(frp_watersheds) > 0:
            frp_watersheds.plot(
                ax=ax1,
                column='watershed_mean_frp',
                cmap=self.intensity_cmap,
                legend=True,
                legend_kwds={
                    'label': 'Mean Fire Radiative Power (MW)',
                    'shrink': 0.7,
                    'pad': 0.01,
                    'format': '%.0f'
                }
            )
        
        ax1.set_title('(a) Mean Fire Intensity', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Subplot B: Total Fire Energy
        self.watersheds.boundary.plot(ax=ax2, linewidth=0.1, color='gray', alpha=0.3)
        
        energy_watersheds = self.watersheds[self.watersheds['total_energy_mwh'] > 0]
        
        if len(energy_watersheds) > 0:
            energy_watersheds.plot(
                ax=ax2,
                column='total_energy_mwh',
                cmap=self.intensity_cmap,
                legend=True,
                legend_kwds={
                    'label': 'Total Fire Energy (MWh)',
                    'shrink': 0.7,
                    'pad': 0.01,
                    'format': '%.0f'
                },
                norm=plt.matplotlib.colors.LogNorm(
                    vmin=max(1, energy_watersheds['total_energy_mwh'].min()),
                    vmax=energy_watersheds['total_energy_mwh'].max()
                )
            )
        
        ax2.set_title('(b) Cumulative Fire Energy', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'Figure_2_Fire_Intensity.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")
        
    def create_figure_3_hsbf(self):
        """Figure 3: Hydrologically Significant Burn Fraction"""
        print("Creating Figure 3: HSBF...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Background watersheds
        self.watersheds.boundary.plot(ax=ax, linewidth=0.1, color='gray', alpha=0.3)
        
        # HSBF values
        hsbf_watersheds = self.watersheds[self.watersheds['hsbf'] > 0].copy()
        
        if len(hsbf_watersheds) > 0:
            # Define meaningful bins for HSBF
            bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
            labels = ['<10%', '10-20%', '20-30%', '30-50%', '50-70%', '>70%']
            
            hsbf_watersheds['hsbf_category'] = pd.cut(
                hsbf_watersheds['hsbf'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            
            # Plot with categorical colors
            hsbf_watersheds.plot(
                ax=ax,
                column='hsbf_category',
                cmap=self.hsbf_cmap,
                categorical=True,
                legend=True,
                legend_kwds={
                    'title': 'HSBF (%)',
                    'loc': 'lower right',
                    'bbox_to_anchor': (0.98, 0.02),
                    'frameon': True,
                    'fancybox': True,
                    'shadow': True
                }
            )
        
        ax.set_title('Hydrologically Significant Burn Fraction by Watershed', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')
        
        # Add scale bar
        self._add_scale_bar(ax)
        
        # Add statistics box
        if len(hsbf_watersheds) > 0:
            stats_text = (
                f"Watersheds with burns: {len(hsbf_watersheds):,}\n"
                f"Max HSBF: {self.watersheds['hsbf'].max()*100:.1f}%\n"
                f"Mean HSBF: {hsbf_watersheds['hsbf'].mean()*100:.1f}%"
            )
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'Figure_3_HSBF.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")
        
    def create_figure_4_threshold_exceedance(self):
        """Figure 4: Threshold Exceedance Analysis"""
        print("Creating Figure 4: Threshold Exceedance...")
        
        # Select key thresholds to display
        thresholds = ['n_10pct_burns', 'n_30pct_burns', 'n_50pct_burns']
        titles = ['(a) Episodes >10% of Watershed', '(b) Episodes >30% of Watershed', 
                 '(c) Episodes >50% of Watershed']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for ax, col, title in zip(axes, thresholds, titles):
            # Background
            self.watersheds.boundary.plot(ax=ax, linewidth=0.1, color='gray', alpha=0.3)
            
            if col in self.watersheds.columns:
                threshold_watersheds = self.watersheds[self.watersheds[col] > 0]
                
                if len(threshold_watersheds) > 0:
                    # Define bins
                    max_val = threshold_watersheds[col].max()
                    if max_val > 10:
                        bins = [1, 2, 3, 5, 10, int(max_val) + 1]
                    elif max_val > 5:
                        bins = [1, 2, 3, 5, int(max_val) + 1]
                    else:
                        bins = list(range(1, int(max_val) + 2))
                    
                    # Ensure bins are unique and sorted
                    bins = sorted(list(set(bins)))
                    norm = BoundaryNorm(bins, self.threshold_cmap.N)
                    
                    threshold_watersheds.plot(
                        ax=ax,
                        column=col,
                        cmap=self.threshold_cmap,
                        norm=norm,
                        legend=True,
                        legend_kwds={
                            'label': 'Number of Episodes',
                            'shrink': 0.8,
                            'pad': 0.01,
                            'ticks': bins[:-1],
                            'format': '%.0f'
                        }
                    )
            
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'Figure_4_Threshold_Exceedance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")
        
    def create_figure_5_fire_severity(self):
        """Figure 5: Fire Severity and Persistence"""
        print("Creating Figure 5: Fire Severity and Persistence...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Subplot A: High Severity Episodes
        self.watersheds.boundary.plot(ax=ax1, linewidth=0.1, color='gray', alpha=0.3)
        
        if 'n_high_severity' in self.watersheds.columns:
            severity_watersheds = self.watersheds[self.watersheds['n_high_severity'] > 0]
            
            if len(severity_watersheds) > 0:
                # Use discrete color bins
                max_severity = severity_watersheds['n_high_severity'].max()
                if max_severity > 10:
                    bins = [1, 2, 3, 5, 10, int(max_severity) + 1]
                elif max_severity > 5:
                    bins = [1, 2, 3, 5, int(max_severity) + 1]
                else:
                    bins = list(range(1, int(max_severity) + 2))
                
                # Ensure bins are unique and sorted
                bins = sorted(list(set(bins)))
                norm = BoundaryNorm(bins, self.intensity_cmap.N)
                
                severity_watersheds.plot(
                    ax=ax1,
                    column='n_high_severity',
                    cmap=self.intensity_cmap,
                    norm=norm,
                    legend=True,
                    legend_kwds={
                        'label': 'High Severity Episodes',
                        'shrink': 0.7,
                        'pad': 0.01,
                        'ticks': bins[:-1],
                        'format': '%.0f'
                    }
                )
        
        ax1.set_title('(a) High Severity Fire Episodes', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Subplot B: Mean Persistence Score
        self.watersheds.boundary.plot(ax=ax2, linewidth=0.1, color='gray', alpha=0.3)
        
        if 'mean_persistence' in self.watersheds.columns:
            persistence_watersheds = self.watersheds[self.watersheds['mean_persistence'] > 0]
            
            if len(persistence_watersheds) > 0:
                persistence_watersheds.plot(
                    ax=ax2,
                    column='mean_persistence',
                    cmap='viridis',
                    legend=True,
                    legend_kwds={
                        'label': 'Mean Persistence Score',
                        'shrink': 0.7,
                        'pad': 0.01,
                        'format': '%.2f'
                    },
                    vmin=0,
                    vmax=1
                )
        
        ax2.set_title('(b) Fire Persistence Pattern', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'Figure_5_Fire_Severity.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")
        
    def create_figure_6_temporal_patterns(self):
        """Figure 6: Temporal Fire Patterns"""
        print("Creating Figure 6: Temporal Patterns...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Subplot A: Total Fire Days
        self.watersheds.boundary.plot(ax=ax1, linewidth=0.1, color='gray', alpha=0.3)
        
        if 'total_fire_days' in self.watersheds.columns:
            firedays_watersheds = self.watersheds[self.watersheds['total_fire_days'] > 0]
            
            if len(firedays_watersheds) > 0:
                firedays_watersheds.plot(
                    ax=ax1,
                    column='total_fire_days',
                    cmap=self.fire_cmap,
                    legend=True,
                    legend_kwds={
                        'label': 'Total Fire Days',
                        'shrink': 0.7,
                        'pad': 0.01,
                        'format': '%.0f'
                    },
                    norm=plt.matplotlib.colors.LogNorm(
                        vmin=max(1, firedays_watersheds['total_fire_days'].min()),
                        vmax=firedays_watersheds['total_fire_days'].max()
                    )
                )
        
        ax1.set_title('(a) Cumulative Fire Days', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Subplot B: Mean Episode Duration
        self.watersheds.boundary.plot(ax=ax2, linewidth=0.1, color='gray', alpha=0.3)
        
        if 'mean_episode_duration_days' in self.watersheds.columns:
            duration_watersheds = self.watersheds[self.watersheds['mean_episode_duration_days'] > 0]
            
            if len(duration_watersheds) > 0:
                duration_watersheds.plot(
                    ax=ax2,
                    column='mean_episode_duration_days',
                    cmap='plasma',
                    legend=True,
                    legend_kwds={
                        'label': 'Mean Episode Duration (days)',
                        'shrink': 0.7,
                        'pad': 0.01,
                        'format': '%.1f'
                    }
                )
        
        ax2.set_title('(b) Average Fire Episode Duration', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'Figure_6_Temporal_Patterns.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")
        
    def create_figure_7_composite_risk(self):
        """Figure 7: Composite Fire Risk Assessment"""
        print("Creating Figure 7: Composite Risk Assessment...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Calculate composite risk score
        self.watersheds['composite_risk'] = 0
        
        # Normalize and weight different factors
        if 'episode_count' in self.watersheds.columns:
            normalized_count = self.watersheds['episode_count'] / self.watersheds['episode_count'].max()
            self.watersheds['composite_risk'] += normalized_count * 0.2
        
        if 'hsbf' in self.watersheds.columns:
            self.watersheds['composite_risk'] += self.watersheds['hsbf'] * 0.3
        
        if 'n_high_severity' in self.watersheds.columns:
            normalized_severity = self.watersheds['n_high_severity'] / self.watersheds['n_high_severity'].max()
            self.watersheds['composite_risk'] += normalized_severity * 0.25
        
        if 'total_energy_mwh' in self.watersheds.columns:
            normalized_energy = self.watersheds['total_energy_mwh'] / self.watersheds['total_energy_mwh'].max()
            self.watersheds['composite_risk'] += normalized_energy * 0.25
        
        # Background
        self.watersheds.boundary.plot(ax=ax, linewidth=0.1, color='gray', alpha=0.3)
        
        # Plot composite risk
        risk_watersheds = self.watersheds[self.watersheds['composite_risk'] > 0]
        
        if len(risk_watersheds) > 0:
            # Define risk categories
            risk_watersheds['risk_category'] = pd.cut(
                risk_watersheds['composite_risk'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            )
            
            # Custom colormap for risk
            risk_colors = ['#2166ac', '#67a9cf', '#f7f7f7', '#fddbc7', '#b2182b']
            risk_cmap = LinearSegmentedColormap.from_list('risk', risk_colors)
            
            risk_watersheds.plot(
                ax=ax,
                column='risk_category',
                cmap=risk_cmap,
                categorical=True,
                legend=True,
                legend_kwds={
                    'title': 'Fire Risk Level',
                    'loc': 'lower right',
                    'bbox_to_anchor': (0.98, 0.02),
                    'frameon': True,
                    'fancybox': True,
                    'shadow': True
                }
            )
        
        ax.set_title('Composite Watershed Fire Risk Assessment', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')
        
        # Add scale bar
        self._add_scale_bar(ax)
        
        # Add methodology note
        method_text = (
            "Risk factors: Episode frequency (20%), HSBF (30%),\n"
            "High severity episodes (25%), Total energy (25%)"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.02, 0.02, method_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=props, style='italic')
        
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / 'Figure_7_Composite_Risk.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")
        
    def create_summary_statistics_table(self):
        """Create a summary statistics figure"""
        print("Creating Summary Statistics Table...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate summary statistics
        stats_data = []
        
        # Total watersheds
        total_watersheds = len(self.watersheds)
        watersheds_with_fires = (self.watersheds['episode_count'] > 0).sum() if 'episode_count' in self.watersheds.columns else 0
        
        stats_data.append(['Total Watersheds', f'{total_watersheds:,}'])
        stats_data.append(['Watersheds with Fires', f'{watersheds_with_fires:,} ({watersheds_with_fires/total_watersheds*100:.1f}%)'])
        
        if 'episode_count' in self.watersheds.columns:
            stats_data.append(['Total Fire Episodes', f'{self.watersheds["episode_count"].sum():,.0f}'])
            stats_data.append(['Mean Episodes per Affected Watershed', 
                             f'{self.watersheds[self.watersheds["episode_count"]>0]["episode_count"].mean():.1f}'])
        
        if 'total_burned_area_km2' in self.watersheds.columns:
            stats_data.append(['Total Burned Area', f'{self.watersheds["total_burned_area_km2"].sum():,.0f} km²'])
        
        if 'total_energy_mwh' in self.watersheds.columns:
            stats_data.append(['Total Fire Energy', f'{self.watersheds["total_energy_mwh"].sum():,.0f} MWh'])
        
        if 'hsbf' in self.watersheds.columns:
            hsbf_watersheds = self.watersheds[self.watersheds['hsbf'] > 0]
            stats_data.append(['Watersheds with HSBF >30%', f'{(self.watersheds["hsbf"] > 0.3).sum():,}'])
            stats_data.append(['Maximum HSBF', f'{self.watersheds["hsbf"].max()*100:.1f}%'])
        
        if 'n_high_severity' in self.watersheds.columns:
            stats_data.append(['High Severity Episodes', f'{self.watersheds["n_high_severity"].sum():,.0f}'])
        
        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#E7E7E7')
                    else:
                        cell.set_facecolor('white')
                
                cell.set_edgecolor('#4472C4')
                cell.set_linewidth(1)
        
        ax.set_title('Watershed Fire Statistics Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Save
        output_path = self.viz_dir / 'Summary_Statistics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")
        
    def _add_scale_bar(self, ax, length=100, location='lower left', pad=0.1):
        """Add a scale bar to the map"""
        # Get the extent of the axis in map units
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate position based on location
        if 'lower' in location:
            y_pos = ylim[0] + (ylim[1] - ylim[0]) * pad
        else:
            y_pos = ylim[1] - (ylim[1] - ylim[0]) * pad
            
        if 'left' in location:
            x_pos = xlim[0] + (xlim[1] - xlim[0]) * pad
        else:
            x_pos = xlim[1] - (xlim[1] - xlim[0]) * pad - length * 1000  # Convert km to m
        
        # Draw scale bar
        ax.plot([x_pos, x_pos + length * 1000], [y_pos, y_pos], 'k-', linewidth=3)
        ax.plot([x_pos, x_pos], [y_pos - 2000, y_pos + 2000], 'k-', linewidth=3)
        ax.plot([x_pos + length * 1000, x_pos + length * 1000], [y_pos - 2000, y_pos + 2000], 'k-', linewidth=3)
        
        # Add text
        ax.text(x_pos + length * 500, y_pos - 5000, f'{length} km', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    def create_all_figures(self):
        """Generate all research figures"""
        print("\n" + "="*60)
        print("GENERATING RESEARCH PAPER FIGURES")
        print("="*60)
        
        # Create individual figures with error handling
        figures = [
            ("Figure 1: Fire Frequency", self.create_figure_1_fire_frequency),
            ("Figure 2: Fire Intensity", self.create_figure_2_fire_intensity),
            ("Figure 3: HSBF", self.create_figure_3_hsbf),
            ("Figure 4: Threshold Exceedance", self.create_figure_4_threshold_exceedance),
            ("Figure 5: Fire Severity", self.create_figure_5_fire_severity),
            ("Figure 6: Temporal Patterns", self.create_figure_6_temporal_patterns),
            ("Figure 7: Composite Risk", self.create_figure_7_composite_risk),
            ("Summary Statistics", self.create_summary_statistics_table)
        ]
        
        for name, func in figures:
            try:
                func()
            except Exception as e:
                print(f"Error creating {name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("FIGURE GENERATION COMPLETE")
        print("="*60)
        print(f"All figures saved to: {self.viz_dir}")
        print("\nGenerated figures:")
        for file in sorted(self.viz_dir.glob('*.png')):
            print(f"  • {file.name}")


def get_latest_output_dir():
    """Find the most recent output directory"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        raise FileNotFoundError("No outputs directory found")
    
    # Find all run directories
    run_dirs = []
    for item in outputs_dir.iterdir():
        if item.is_dir() and ('_run_' in item.name):
            run_dirs.append(item)
    
    if not run_dirs:
        raise FileNotFoundError("No run directories found in outputs/")
    
    # Sort by modification time and return the latest
    latest_dir = sorted(run_dirs, key=lambda x: x.stat().st_mtime)[-1]
    return latest_dir


def main():
    """Main function to run watershed visualizations"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create research paper quality spatial visualizations of watershed fire metrics"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        nargs='?',
        help='Path to the output directory from fire_episode_clustering.py (optional)'
    )
    
    args = parser.parse_args()
    
    # Determine which directory to use
    if args.output_dir:
        output_path = Path(args.output_dir)
        if not output_path.exists():
            print(f"Error: Output directory not found: {output_path}")
            sys.exit(1)
    else:
        # Use latest directory
        try:
            output_path = get_latest_output_dir()
            print(f"Using latest output directory: {output_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Run visualization
    try:
        visualizer = WatershedSpatialVisualizer(output_path)
        visualizer.create_all_figures()
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()