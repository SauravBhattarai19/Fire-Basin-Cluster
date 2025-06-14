#!/usr/bin/env python3
"""
Fire Episode Clustering System
Main entry point for spatiotemporal clustering of MODIS FIRMS fire data
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import (
    load_config, setup_logging, create_output_directory, 
    PerformanceMonitor, format_duration, save_checkpoint, load_checkpoint
)
from data_preparation import DataPreparation
from clustering import SpatioTemporalDBSCAN
from episode_characterization import EpisodeCharacterization
from validation import ValidationFramework

def main(config_path, resume_checkpoint=None):
    """
    Main pipeline for fire episode clustering
    
    Args:
        config_path: Path to configuration YAML file
        resume_checkpoint: Path to checkpoint file to resume from
    """
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("="*60)
    logger.info("FIRE EPISODE CLUSTERING SYSTEM")
    logger.info("="*60)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Test mode: {config['study_area']['test_mode']}")
    
    # Create output directory
    output_dir = create_output_directory(config)
    logger.info(f"Output directory: {output_dir}")
    
    # Copy config to output directory
    import shutil
    shutil.copy2(config_path, output_dir / 'config.yaml')
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(config['logging']['report_interval_seconds'])
    
    # Initialize pipeline results
    pipeline_results = {
        'start_time': datetime.now().isoformat(),
        'config_path': str(config_path),
        'output_dir': str(output_dir)
    }
    
    try:
        # Check for resume checkpoint
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            checkpoint_data, checkpoint_meta = load_checkpoint(resume_checkpoint)
            start_stage = checkpoint_meta.get('stage', 1)
        else:
            start_stage = 1
            checkpoint_data = None
        
        # Stage 1: Data Preparation
        if start_stage <= 1:
            logger.info("\n" + "="*40)
            logger.info("STAGE 1: DATA PREPARATION")
            logger.info("="*40)
            
            data_prep = DataPreparation(config)
            
            # Load fire data
            fire_df = data_prep.load_fire_data()
            
            # Load watershed data
            watershed_gdf = data_prep.load_watershed_data()
            
            # Get data summary
            data_summary = data_prep.get_data_summary(fire_df, watershed_gdf)
            pipeline_results['data_summary'] = data_summary
            
            # Save summary
            with open(output_dir / 'data_summary.json', 'w') as f:
                json.dump(data_summary, f, indent=2)
            
            # Prepare for clustering
            coords, times, features, metadata = data_prep.prepare_for_clustering(fire_df)
            
            # Save checkpoint
            if config['checkpoint']['enable_checkpointing']:
                checkpoint_path = output_dir / 'checkpoints' / 'stage1_data_prep.pkl'
                save_checkpoint(
                    {
                        'fire_df': fire_df,
                        'watershed_gdf': watershed_gdf,
                        'coords': coords,
                        'times': times,
                        'features': features,
                        'metadata': metadata
                    },
                    checkpoint_path,
                    {'stage': 1, 'completed': True}
                )
            
            monitor.log_resources(force=True)
        
        # Stage 2: Spatiotemporal Clustering
        if start_stage <= 2:
            logger.info("\n" + "="*40)
            logger.info("STAGE 2: SPATIOTEMPORAL CLUSTERING")
            logger.info("="*40)
            
            # Load from checkpoint if needed
            if start_stage == 2 and checkpoint_data:
                fire_df = checkpoint_data['fire_df']
                watershed_gdf = checkpoint_data['watershed_gdf']
                coords = checkpoint_data['coords']
                times = checkpoint_data['times']
                features = checkpoint_data['features']
                metadata = checkpoint_data['metadata']
            
            # Initialize clustering
            clusterer = SpatioTemporalDBSCAN(config)
            
            # Check if parameter optimization is requested
            if config.get('run_parameter_optimization', False):
                logger.info("Running parameter optimization...")
                param_ranges = config['clustering']['param_ranges']
                
                # Use subset for optimization
                subset_size = min(50000, len(coords))
                subset_idx = np.random.choice(len(coords), subset_size, replace=False)
                
                opt_results, best_params = clusterer.parameter_optimization(
                    coords[subset_idx],
                    times[subset_idx],
                    {k: v[subset_idx] for k, v in features.items()},
                    param_ranges
                )
                
                # Save optimization results
                opt_results.to_csv(output_dir / 'parameter_optimization.csv', index=False)
                
                # Update clustering parameters
                clusterer.spatial_eps = best_params['spatial_eps']
                clusterer.temporal_eps = best_params['temporal_eps']
                clusterer.min_samples = int(best_params['min_samples'])
                
                logger.info(f"Using optimized parameters: eps_s={clusterer.spatial_eps}m, "
                          f"eps_t={clusterer.temporal_eps}d, min_samples={clusterer.min_samples}")
            
            # Perform clustering
            cluster_start_time = time.time()
            labels = clusterer.fit_predict(coords, times, features)
            cluster_time = time.time() - cluster_start_time
            
            logger.info(f"Clustering completed in {format_duration(cluster_time)}")
            
            # Add labels to fire dataframe
            fire_df['episode_id'] = labels
            
            # Save checkpoint
            if config['checkpoint']['enable_checkpointing']:
                checkpoint_path = output_dir / 'checkpoints' / 'stage2_clustering.pkl'
                save_checkpoint(
                    {
                        'fire_df': fire_df,
                        'watershed_gdf': watershed_gdf,
                        'labels': labels,
                        'coords': coords,
                        'times': times,
                        'clustering_time': cluster_time
                    },
                    checkpoint_path,
                    {'stage': 2, 'completed': True}
                )
            
            monitor.log_resources(force=True)
        
        # Stage 3: Episode Characterization
        if start_stage <= 3:
            logger.info("\n" + "="*40)
            logger.info("STAGE 3: EPISODE CHARACTERIZATION")
            logger.info("="*40)
            
            # Load from checkpoint if needed
            if start_stage == 3 and checkpoint_data:
                fire_df = checkpoint_data['fire_df']
                watershed_gdf = checkpoint_data['watershed_gdf']
                labels = checkpoint_data['labels']
                coords = checkpoint_data['coords']
                times = checkpoint_data['times']
            
            # Initialize episode characterization
            episode_char = EpisodeCharacterization(config)
            
            # Generate episode records
            episodes_df = episode_char.characterize_episodes(fire_df, labels)
            
            # Save episode records
            if config['output']['save_episode_records']:
                # Save in multiple formats
                for fmt in config['output']['export_formats']:
                    if fmt == 'parquet':
                        episodes_df.to_parquet(
                            output_dir / 'episodes' / 'fire_episodes.parquet',
                            index=False
                        )
                    elif fmt == 'csv':
                        episodes_df.to_csv(
                            output_dir / 'episodes' / 'fire_episodes.csv',
                            index=False
                        )
                    elif fmt == 'geojson':
                        # Convert to GeoDataFrame
                        import geopandas as gpd
                        from shapely.geometry import Point
                        
                        geometry = [Point(row['centroid_lon'], row['centroid_lat']) 
                                  for _, row in episodes_df.iterrows()]
                        episodes_gdf = gpd.GeoDataFrame(
                            episodes_df, geometry=geometry, crs='EPSG:4326'
                        )
                        episodes_gdf.to_file(
                            output_dir / 'episodes' / 'fire_episodes.geojson',
                            driver='GeoJSON'
                        )
                
                logger.info(f"Saved {len(episodes_df)} episode records")
            
            # Aggregate to watersheds if configured
            if config['output']['save_watershed_stats']:
                watershed_stats = episode_char.aggregate_to_watersheds(
                    episodes_df, watershed_gdf
                )
                
                # Save watershed statistics
                watershed_stats.to_file(
                    output_dir / 'episodes' / 'watershed_fire_statistics.geojson',
                    driver='GeoJSON'
                )
                
                logger.info(f"Aggregated statistics for {len(watershed_stats)} watersheds")
            
            # Save checkpoint
            if config['checkpoint']['enable_checkpointing']:
                checkpoint_path = output_dir / 'checkpoints' / 'stage3_episodes.pkl'
                save_checkpoint(
                    {
                        'episodes_df': episodes_df,
                        'watershed_stats': watershed_stats if config['output']['save_watershed_stats'] else None
                    },
                    checkpoint_path,
                    {'stage': 3, 'completed': True}
                )
            
            monitor.log_resources(force=True)
        
        # Stage 4: Validation and Quality Assessment
        if start_stage <= 4:
            logger.info("\n" + "="*40)
            logger.info("STAGE 4: VALIDATION & QUALITY ASSESSMENT")
            logger.info("="*40)
            
            # Load from checkpoint if needed
            if start_stage == 4 and checkpoint_data:
                episodes_df = checkpoint_data['episodes_df']
                if 'fire_df' in checkpoint_data:
                    fire_df = checkpoint_data['fire_df']
                    labels = checkpoint_data['labels']
                    coords = checkpoint_data['coords']
                    times = checkpoint_data['times']
            
            # Initialize validation framework
            validator = ValidationFramework(config, output_dir)
            
            # Validate clustering
            clustering_validation = validator.validate_clustering(
                fire_df, labels, coords, times
            )
            pipeline_results['clustering_validation'] = clustering_validation
            
            # Validate episodes
            episode_validation = validator.validate_episodes(episodes_df)
            pipeline_results['episode_validation'] = episode_validation
            
            # Generate validation report
            validator.generate_validation_report(pipeline_results)
            
            logger.info("Validation completed")
            
            monitor.log_resources(force=True)
        
        # Final summary
        pipeline_results['end_time'] = datetime.now().isoformat()
        pipeline_results['total_duration_seconds'] = (
            datetime.fromisoformat(pipeline_results['end_time']) - 
            datetime.fromisoformat(pipeline_results['start_time'])
        ).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total duration: {format_duration(pipeline_results['total_duration_seconds'])}")
        logger.info(f"Episodes generated: {len(episodes_df)}")
        logger.info(f"Output directory: {output_dir}")
        
        # Save final results
        with open(output_dir / 'pipeline_results.json', 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        # Clean up old checkpoints if configured
        if config['checkpoint']['max_checkpoint_age_days'] > 0:
            _cleanup_old_checkpoints(output_dir / 'checkpoints', 
                                   config['checkpoint']['max_checkpoint_age_days'])
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        pipeline_results['error'] = str(e)
        pipeline_results['status'] = 'FAILED'
        
        # Save error report
        with open(output_dir / 'error_report.json', 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        raise

def _cleanup_old_checkpoints(checkpoint_dir, max_age_days):
    """Remove checkpoints older than max_age_days"""
    import shutil
    from datetime import datetime, timedelta
    
    if not checkpoint_dir.exists():
        return
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    
    for checkpoint_file in checkpoint_dir.glob('*.pkl'):
        if datetime.fromtimestamp(checkpoint_file.stat().st_mtime) < cutoff_time:
            checkpoint_file.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fire Episode Clustering System - Spatiotemporal clustering of MODIS FIRMS data"
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume from'
    )
    
    parser.add_argument(
        '--optimize-params',
        action='store_true',
        help='Run parameter optimization before clustering'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Add parameter optimization flag to config if specified
    if args.optimize_params:
        config = load_config(config_path)
        config['run_parameter_optimization'] = True
        
        # Create temporary config with optimization flag
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        try:
            main(temp_config_path, args.resume)
        finally:
            os.unlink(temp_config_path)
    else:
        main(config_path, args.resume) 