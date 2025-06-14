#!/usr/bin/env python3
"""
Data preparation module for fire episode clustering
Handles data loading, quality control, and preprocessing
"""

import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils import SpatialUtils, chunk_data_spatially

class DataPreparation:
    """Handle data loading and preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('DataPreparation')
        self.spatial_utils = SpatialUtils(config['study_area']['output_epsg'])
        
        # Set up data paths
        self.fire_path = Path(config['data']['fire_data_path'])
        self.watershed_path = Path(config['data']['watershed_data_path'])
        
        # Quality control parameters
        self.qc_params = config['quality_control']
        
    def load_fire_data(self):
        """Load and preprocess MODIS FIRMS fire data"""
        self.logger.info(f"Loading fire data from {self.fire_path}")
        
        # Check if file exists
        if not self.fire_path.exists():
            raise FileNotFoundError(f"Fire data file not found: {self.fire_path}")
        
        # Load JSON data
        with open(self.fire_path, 'r') as f:
            fire_data = json.load(f)
        
        # Convert to DataFrame
        fire_df = pd.DataFrame(fire_data)
        self.logger.info(f"Loaded {len(fire_df):,} fire detection records")
        
        # Convert data types
        fire_df['confidence'] = pd.to_numeric(fire_df['confidence'], errors='coerce')
        fire_df['frp'] = pd.to_numeric(fire_df['frp'], errors='coerce')
        fire_df['brightness'] = pd.to_numeric(fire_df['brightness'], errors='coerce')
        fire_df['latitude'] = pd.to_numeric(fire_df['latitude'], errors='coerce')
        fire_df['longitude'] = pd.to_numeric(fire_df['longitude'], errors='coerce')
        
        # Parse dates and times
        fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date'])
        
        # Create full datetime column
        fire_df['acq_time_str'] = fire_df['acq_time'].astype(str).str.zfill(4)
        fire_df['acq_datetime'] = pd.to_datetime(
            fire_df['acq_date'].astype(str) + ' ' + 
            fire_df['acq_time_str'].str[:2] + ':' + 
            fire_df['acq_time_str'].str[2:]
        )
        
        # Apply geographic filtering if in test mode
        if self.config['study_area']['test_mode']:
            fire_df = self._apply_geographic_filter(fire_df)
        
        # Apply quality control
        fire_df = self._apply_quality_control(fire_df)
        
        # Add additional computed fields
        fire_df = self._add_computed_fields(fire_df)
        
        # Sort by datetime
        fire_df = fire_df.sort_values('acq_datetime').reset_index(drop=True)
        
        self.logger.info(f"After preprocessing: {len(fire_df):,} records remain")
        
        return fire_df
    
    def _apply_geographic_filter(self, df):
        """Apply geographic bounding box filter"""
        bbox = self.config['study_area']['bounding_box']
        west, south, east, north = bbox
        
        self.logger.info(f"Applying geographic filter: {bbox}")
        
        mask = (
            (df['longitude'] >= west) &
            (df['longitude'] <= east) &
            (df['latitude'] >= south) &
            (df['latitude'] <= north)
        )
        
        filtered_df = df[mask].copy()
        self.logger.info(f"Geographic filter: {len(df):,} -> {len(filtered_df):,} records")
        
        return filtered_df
    
    def _apply_quality_control(self, df):
        """Apply quality control filters"""
        initial_count = len(df)
        
        # Filter by confidence
        min_conf = self.qc_params['min_confidence']
        df = df[df['confidence'] >= min_conf]
        self.logger.info(f"Confidence filter (>={min_conf}%): {initial_count:,} -> {len(df):,}")
        
        # Filter by valid instruments
        valid_instruments = self.qc_params['valid_instruments']
        if valid_instruments:
            df = df[df['instrument'].isin(valid_instruments)]
            self.logger.info(f"Instrument filter: -> {len(df):,}")
        
        # Filter by valid satellites
        valid_satellites = self.qc_params['valid_satellites']
        if valid_satellites:
            df = df[df['satellite'].isin(valid_satellites)]
            self.logger.info(f"Satellite filter: -> {len(df):,}")
        
        # Remove duplicates if configured
        if self.qc_params['remove_duplicates']:
            df = self._remove_duplicates(df)
        
        # Remove invalid coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Remove invalid FRP values
        df = df[df['frp'] > 0]
        
        self.logger.info(f"Total quality control: {initial_count:,} -> {len(df):,} records")
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove spatial-temporal duplicates"""
        radius_m = self.qc_params['duplicate_radius_m']
        time_hours = self.qc_params['duplicate_time_hours']
        
        self.logger.info(f"Removing duplicates within {radius_m}m and {time_hours}h")
        
        # Transform to projected coordinates for distance calculation
        x, y = self.spatial_utils.transform_points(
            df['longitude'].values, 
            df['latitude'].values,
            to_target=True
        )
        df['x_proj'] = x
        df['y_proj'] = y
        
        # Round coordinates to grid cells
        grid_size = radius_m
        df['x_grid'] = (df['x_proj'] / grid_size).round()
        df['y_grid'] = (df['y_proj'] / grid_size).round()
        
        # Round time to hours
        df['time_grid'] = df['acq_datetime'].dt.round(f'{time_hours}H')
        
        # Keep highest confidence detection in each grid cell
        df_dedup = df.sort_values('confidence', ascending=False).drop_duplicates(
            subset=['x_grid', 'y_grid', 'time_grid'], 
            keep='first'
        )
        
        # Clean up temporary columns
        df_dedup = df_dedup.drop(columns=['x_proj', 'y_proj', 'x_grid', 'y_grid', 'time_grid'])
        
        self.logger.info(f"Duplicate removal: {len(df):,} -> {len(df_dedup):,}")
        
        return df_dedup
    
    def _add_computed_fields(self, df):
        """Add computed fields for analysis"""
        # Day of year for seasonal analysis
        df['day_of_year'] = df['acq_datetime'].dt.dayofyear
        
        # Hour of day for diurnal analysis
        df['hour_of_day'] = df['acq_datetime'].dt.hour
        
        # Season (simple Northern Hemisphere)
        df['season'] = df['acq_date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Projected coordinates
        x, y = self.spatial_utils.transform_points(
            df['longitude'].values,
            df['latitude'].values,
            to_target=True
        )
        df['x_proj'] = x
        df['y_proj'] = y
        
        # Unique identifier
        df['point_id'] = range(len(df))
        
        return df
    
    def load_watershed_data(self):
        """Load watershed boundaries"""
        self.logger.info(f"Loading watershed data from {self.watershed_path}")
        
        # Configure GDAL for large files
        os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'
        
        # Load watershed data
        watershed_gdf = gpd.read_file(self.watershed_path)
        self.logger.info(f"Loaded {len(watershed_gdf):,} watersheds")
        
        # Apply geographic filter if in test mode
        if self.config['study_area']['test_mode']:
            bbox = self.config['study_area']['bounding_box']
            west, south, east, north = bbox
            
            # Create bounding box polygon
            from shapely.geometry import box
            bbox_poly = box(west, south, east, north)
            
            # Filter watersheds that intersect with bounding box
            watershed_gdf = watershed_gdf[watershed_gdf.intersects(bbox_poly)]
            self.logger.info(f"Geographic filter: -> {len(watershed_gdf):,} watersheds")
        
        # Reproject to target CRS
        target_epsg = self.config['study_area']['output_epsg']
        watershed_gdf = watershed_gdf.to_crs(f'EPSG:{target_epsg}')
        
        # Calculate basic properties
        watershed_gdf['area_km2'] = watershed_gdf.geometry.area / 1e6
        watershed_gdf['centroid'] = watershed_gdf.geometry.centroid
        
        return watershed_gdf
    
    def prepare_for_clustering(self, fire_df):
        """Prepare fire data for clustering algorithm"""
        self.logger.info("Preparing data for clustering")
        
        # Extract required arrays
        coords = fire_df[['x_proj', 'y_proj']].values
        times = fire_df['acq_datetime'].values
        
        # Additional features for weighted clustering
        features = {
            'confidence': fire_df['confidence'].values,
            'frp': fire_df['frp'].values,
            'brightness': fire_df['brightness'].values,
            'daynight': fire_df['daynight'].values,
            'satellite': fire_df['satellite'].values
        }
        
        # Create metadata dictionary
        metadata = {
            'total_points': len(fire_df),
            'date_range': (
                fire_df['acq_datetime'].min().isoformat(),
                fire_df['acq_datetime'].max().isoformat()
            ),
            'bbox_proj': [
                coords[:, 0].min(), coords[:, 1].min(),
                coords[:, 0].max(), coords[:, 1].max()
            ],
            'satellites': fire_df['satellite'].unique().tolist(),
            'projection': self.config['study_area']['output_epsg']
        }
        
        return coords, times, features, metadata
    
    def chunk_fire_data(self, fire_df):
        """Chunk fire data for parallel processing"""
        chunk_size = self.config['processing']['spatial_chunk_size']
        overlap = self.config['processing']['overlap_buffer']
        
        self.logger.info(f"Chunking data: {chunk_size}° tiles with {overlap}° overlap")
        
        chunks = chunk_data_spatially(fire_df, chunk_size, overlap)
        
        self.logger.info(f"Created {len(chunks)} spatial chunks")
        
        return chunks
    
    def get_data_summary(self, fire_df, watershed_gdf=None):
        """Generate summary statistics of loaded data"""
        summary = {
            'fire_data': {
                'total_records': len(fire_df),
                'date_range': {
                    'start': fire_df['acq_datetime'].min().isoformat(),
                    'end': fire_df['acq_datetime'].max().isoformat(),
                    'days': (fire_df['acq_datetime'].max() - fire_df['acq_datetime'].min()).days
                },
                'geographic_bounds': {
                    'west': float(fire_df['longitude'].min()),
                    'east': float(fire_df['longitude'].max()),
                    'south': float(fire_df['latitude'].min()),
                    'north': float(fire_df['latitude'].max())
                },
                'satellites': fire_df['satellite'].value_counts().to_dict(),
                'day_night': fire_df['daynight'].value_counts().to_dict(),
                'confidence_stats': {
                    'mean': float(fire_df['confidence'].mean()),
                    'std': float(fire_df['confidence'].std()),
                    'min': float(fire_df['confidence'].min()),
                    'max': float(fire_df['confidence'].max())
                },
                'frp_stats': {
                    'mean': float(fire_df['frp'].mean()),
                    'std': float(fire_df['frp'].std()),
                    'min': float(fire_df['frp'].min()),
                    'max': float(fire_df['frp'].max())
                }
            }
        }
        
        if watershed_gdf is not None:
            summary['watershed_data'] = {
                'total_watersheds': len(watershed_gdf),
                'area_stats': {
                    'mean': float(watershed_gdf['area_km2'].mean()),
                    'std': float(watershed_gdf['area_km2'].std()),
                    'min': float(watershed_gdf['area_km2'].min()),
                    'max': float(watershed_gdf['area_km2'].max())
                }
            }
        
        return summary 