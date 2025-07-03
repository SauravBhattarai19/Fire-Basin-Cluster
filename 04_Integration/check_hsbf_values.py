#!/usr/bin/env python3
"""
Detailed HSBF analysis to understand the distribution and identify specific issues
Place this in 04_Integration/ directory
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

def detailed_hsbf_analysis(output_dir):
    """Perform detailed analysis of HSBF values"""
    
    output_path = Path(output_dir)
    episodes_dir = output_path / 'episodes'
    
    print("="*60)
    print("DETAILED HSBF ANALYSIS")
    print("="*60)
    
    # Load watershed statistics
    watershed_path = episodes_dir / 'watershed_fire_statistics.geojson'
    if not watershed_path.exists():
        print(f"Error: Watershed statistics not found at {watershed_path}")
        return
    
    print(f"\nLoading watershed data...")
    watersheds = gpd.read_file(watershed_path)
    print(f"Loaded {len(watersheds)} watersheds")
    
    # Filter to watersheds with HSBF > 0
    watersheds_with_hsbf = watersheds[watersheds['hsbf'] > 0].copy()
    print(f"Watersheds with HSBF > 0: {len(watersheds_with_hsbf)}")
    
    # Convert HSBF to percentage for easier interpretation
    watersheds_with_hsbf['hsbf_percent'] = watersheds_with_hsbf['hsbf'] * 100
    
    # Detailed statistics
    print("\n" + "-"*60)
    print("HSBF DISTRIBUTION ANALYSIS (for watersheds with HSBF > 0)")
    print("-"*60)
    
    # Calculate various percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
    print("\nHSBF Percentiles:")
    for p in percentiles:
        value = np.percentile(watersheds_with_hsbf['hsbf_percent'], p)
        print(f"  {p:5.1f}th percentile: {value:8.1f}%")
    
    # Analyze by categories
    print("\n" + "-"*60)
    print("HSBF CATEGORIES")
    print("-"*60)
    
    categories = [
        (0, 1, "0-1%"),
        (1, 5, "1-5%"),
        (5, 10, "5-10%"),
        (10, 20, "10-20%"),
        (20, 30, "20-30%"),
        (30, 50, "30-50%"),
        (50, 70, "50-70%"),
        (70, 100, "70-100%"),
        (100, float('inf'), ">100% (INVALID)")
    ]
    
    print("\nWatersheds by HSBF category:")
    for min_val, max_val, label in categories:
        count = ((watersheds_with_hsbf['hsbf_percent'] > min_val) & 
                (watersheds_with_hsbf['hsbf_percent'] <= max_val)).sum()
        pct = count / len(watersheds_with_hsbf) * 100
        print(f"  {label:20s}: {count:6d} watersheds ({pct:5.1f}%)")
    
    # Focus on the problematic watersheds
    print("\n" + "-"*60)
    print("ANALYSIS OF PROBLEMATIC WATERSHEDS (HSBF > 100%)")
    print("-"*60)
    
    problematic = watersheds_with_hsbf[watersheds_with_hsbf['hsbf'] > 1].copy()
    print(f"\nTotal problematic watersheds: {len(problematic)}")
    
    if len(problematic) > 0:
        # Group by magnitude of error
        problematic['error_magnitude'] = problematic['hsbf_percent'] / 100
        
        print("\nError magnitude distribution:")
        error_ranges = [(1, 10), (10, 100), (100, 1000), (1000, 10000), (10000, float('inf'))]
        for min_mag, max_mag in error_ranges:
            count = ((problematic['error_magnitude'] >= min_mag) & 
                    (problematic['error_magnitude'] < max_mag)).sum()
            if count > 0:
                print(f"  {min_mag:5.0f}x - {max_mag:5.0f}x too large: {count} watersheds")
        
        # Analyze characteristics of problematic watersheds
        print("\n" + "-"*60)
        print("CHARACTERISTICS OF PROBLEMATIC WATERSHEDS")
        print("-"*60)
        
        print("\nWatershed area statistics (problematic vs normal):")
        normal = watersheds_with_hsbf[watersheds_with_hsbf['hsbf'] <= 1]
        
        print(f"\nProblematic watersheds (HSBF > 100%):")
        print(f"  Mean area: {problematic['area_km2'].mean():.2f} km²")
        print(f"  Median area: {problematic['area_km2'].median():.2f} km²")
        print(f"  Min area: {problematic['area_km2'].min():.2f} km²")
        print(f"  Max area: {problematic['area_km2'].max():.2f} km²")
        
        print(f"\nNormal watersheds (HSBF ≤ 100%):")
        print(f"  Mean area: {normal['area_km2'].mean():.2f} km²")
        print(f"  Median area: {normal['area_km2'].median():.2f} km²")
        print(f"  Min area: {normal['area_km2'].min():.2f} km²")
        print(f"  Max area: {normal['area_km2'].max():.2f} km²")
        
        # Check for very small watersheds
        tiny_problematic = problematic[problematic['area_km2'] < 10]
        print(f"\nProblematic watersheds < 10 km²: {len(tiny_problematic)} ({len(tiny_problematic)/len(problematic)*100:.1f}%)")
        
        # Detailed look at worst cases
        print("\n" + "-"*60)
        print("TOP 20 WORST CASES - DETAILED ANALYSIS")
        print("-"*60)
        
        worst_cases = problematic.nlargest(20, 'hsbf')
        for idx, row in worst_cases.iterrows():
            print(f"\nHUC12: {row.get('huc12', 'N/A')}")
            print(f"  HSBF: {row['hsbf_percent']:.1f}%")
            print(f"  Watershed area: {row['area_km2']:.2f} km²")
            print(f"  Max episode area: {row.get('max_episode_area_km2', 'N/A'):.2f} km²")
            print(f"  Total burned area: {row.get('total_burned_area_km2', 'N/A'):.2f} km²")
            print(f"  Episode count: {row.get('episode_count', 'N/A')}")
            
            # Calculate what the episode area would need to be
            implied_episode_area = row['hsbf'] * row['area_km2']
            print(f"  Implied episode area from HSBF: {implied_episode_area:.2f} km²")
            
            if 'max_episode_area_km2' in row:
                discrepancy = implied_episode_area / row['max_episode_area_km2']
                print(f"  Discrepancy factor: {discrepancy:.1f}x")
    
    # Create visualization
    print("\n" + "-"*60)
    print("CREATING DIAGNOSTIC PLOTS")
    print("-"*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. HSBF distribution histogram (log scale)
    ax1 = axes[0, 0]
    hsbf_values = watersheds_with_hsbf['hsbf_percent']
    bins = np.logspace(np.log10(0.01), np.log10(hsbf_values.max()), 50)
    ax1.hist(hsbf_values, bins=bins, edgecolor='black', alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_xlabel('HSBF (%)')
    ax1.set_ylabel('Number of Watersheds')
    ax1.set_title('HSBF Distribution (log scale)')
    ax1.axvline(x=100, color='red', linestyle='--', label='100% threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. HSBF vs Watershed Area scatter
    ax2 = axes[0, 1]
    colors = ['blue' if x <= 100 else 'red' for x in watersheds_with_hsbf['hsbf_percent']]
    ax2.scatter(watersheds_with_hsbf['area_km2'], watersheds_with_hsbf['hsbf_percent'], 
                c=colors, alpha=0.5, s=20)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Watershed Area (km²)')
    ax2.set_ylabel('HSBF (%)')
    ax2.set_title('HSBF vs Watershed Area')
    ax2.axhline(y=100, color='red', linestyle='--', label='100% threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Episode area vs Watershed area for problematic cases
    ax3 = axes[1, 0]
    if 'max_episode_area_km2' in problematic.columns:
        ax3.scatter(problematic['area_km2'], problematic['max_episode_area_km2'], 
                   c='red', alpha=0.6, label='Problematic')
        # Add 1:1 line
        min_val = min(problematic['area_km2'].min(), problematic['max_episode_area_km2'].min())
        max_val = max(problematic['area_km2'].max(), problematic['max_episode_area_km2'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Watershed Area (km²)')
        ax3.set_ylabel('Max Episode Area (km²)')
        ax3.set_title('Episode vs Watershed Area (Problematic Cases)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Summary statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""HSBF SUMMARY STATISTICS

Total watersheds: {len(watersheds):,}
Watersheds with fires: {len(watersheds_with_hsbf):,}

HSBF Statistics (fire-affected only):
• Mean: {watersheds_with_hsbf['hsbf_percent'].mean():.1f}%
• Median: {watersheds_with_hsbf['hsbf_percent'].median():.1f}%
• 95th percentile: {np.percentile(watersheds_with_hsbf['hsbf_percent'], 95):.1f}%
• 99th percentile: {np.percentile(watersheds_with_hsbf['hsbf_percent'], 99):.1f}%
• Maximum: {watersheds_with_hsbf['hsbf_percent'].max():.1f}%

Problematic watersheds (>100%): {len(problematic)}
• Mean watershed area: {problematic['area_km2'].mean():.1f} km²
• Watersheds <10 km²: {len(tiny_problematic)}

Likely cause: Fire episodes that span 
multiple watersheds are being assigned
their full area to each watershed."""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('HSBF Diagnostic Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / 'hsbf_diagnostic_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved diagnostic plot to: {plot_path}")
    
    # Final recommendations
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKEY FINDINGS:")
    print(f"1. The mean HSBF of {watersheds_with_hsbf['hsbf_percent'].mean():.1f}% is reasonable")
    print(f"2. {len(problematic)} watersheds ({len(problematic)/len(watersheds_with_hsbf)*100:.1f}%) have impossible HSBF values > 100%")
    print(f"3. The worst case is {watersheds_with_hsbf['hsbf_percent'].max():.1f}% (should be max 100%)")
    print(f"4. Most problematic watersheds are small (median {problematic['area_km2'].median():.1f} km²)")
    print("\nLIKELY CAUSE:")
    print("Large fire episodes that span multiple watersheds are being")
    print("assigned their full area to each watershed they touch, rather")
    print("than just the portion within each watershed.")
    print("\nRECOMMENDED FIX:")
    print("Clip fire episode polygons to watershed boundaries before")
    print("calculating areas, or use spatial intersection to determine")
    print("the actual burned area within each watershed.")


def get_latest_output_dir():
    """Find the most recent output directory"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        raise FileNotFoundError("No outputs directory found")
    
    run_dirs = []
    for item in outputs_dir.iterdir():
        if item.is_dir() and ('_run_' in item.name):
            run_dirs.append(item)
    
    if not run_dirs:
        raise FileNotFoundError("No run directories found in outputs/")
    
    latest_dir = sorted(run_dirs, key=lambda x: x.stat().st_mtime)[-1]
    return latest_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detailed HSBF analysis")
    parser.add_argument('output_dir', type=str, nargs='?', 
                       help='Path to output directory (optional)')
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_path = Path(args.output_dir)
        if not output_path.exists():
            print(f"Error: Output directory not found: {output_path}")
            sys.exit(1)
    else:
        try:
            output_path = get_latest_output_dir()
            print(f"Using latest output directory: {output_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    detailed_hsbf_analysis(output_path)