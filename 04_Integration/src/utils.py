#!/usr/bin/env python3
"""
Utility functions for fire episode clustering
"""

import os
import sys
import yaml
import json
import logging
import psutil
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pyproj
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU libraries
try:
    import cupy as cp
    import cuspatial
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries not available, falling back to CPU processing")

class PerformanceMonitor:
    """Monitor system resource usage"""
    
    def __init__(self, log_interval=60):
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.logger = logging.getLogger('PerformanceMonitor')
        
    def log_resources(self, force=False):
        """Log current resource usage"""
        current_time = time.time()
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return
            
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU monitoring if available
        gpu_info = ""
        if GPU_AVAILABLE:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info = f" | GPU: {gpu_util.gpu}% util, {gpu_mem.used/1e9:.1f}/{gpu_mem.total/1e9:.1f} GB"
            except:
                pass
        
        elapsed = (current_time - self.start_time) / 60
        self.logger.info(
            f"Resources - Elapsed: {elapsed:.1f} min | "
            f"CPU: {cpu_percent}% | "
            f"RAM: {memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB ({memory.percent}%)"
            f"{gpu_info}"
        )
        self.last_log_time = current_time

class SpatialUtils:
    """Utilities for spatial operations"""
    
    def __init__(self, target_epsg=6933):
        self.target_epsg = target_epsg
        self.wgs84 = pyproj.CRS('EPSG:4326')
        self.target_crs = pyproj.CRS(f'EPSG:{target_epsg}')
        self.transformer_to_target = pyproj.Transformer.from_crs(
            self.wgs84, self.target_crs, always_xy=True
        )
        self.transformer_to_wgs84 = pyproj.Transformer.from_crs(
            self.target_crs, self.wgs84, always_xy=True
        )
    
    def transform_points(self, lons, lats, to_target=True):
        """Transform points between coordinate systems"""
        if to_target:
            return self.transformer_to_target.transform(lons, lats)
        else:
            return self.transformer_to_wgs84.transform(lons, lats)
    
    def calculate_distance_matrix(self, coords1, coords2=None, use_gpu=False):
        """Calculate pairwise distances between points"""
        if coords2 is None:
            coords2 = coords1
            
        if use_gpu and GPU_AVAILABLE:
            # GPU-accelerated distance calculation
            coords1_gpu = cp.asarray(coords1)
            coords2_gpu = cp.asarray(coords2)
            
            # Euclidean distance calculation on GPU
            diff = coords1_gpu[:, np.newaxis, :] - coords2_gpu[np.newaxis, :, :]
            distances = cp.sqrt(cp.sum(diff**2, axis=2))
            return cp.asnumpy(distances)
        else:
            # CPU-based calculation using broadcasting
            diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
            return np.sqrt(np.sum(diff**2, axis=2))
    
    def create_bounding_box(self, points, buffer_m=1000):
        """Create bounding box around points with buffer"""
        min_x = np.min(points[:, 0]) - buffer_m
        max_x = np.max(points[:, 0]) + buffer_m
        min_y = np.min(points[:, 1]) - buffer_m
        max_y = np.max(points[:, 1]) + buffer_m
        
        return Polygon([
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y)
        ])

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('log_level', 'INFO'))
    log_file = log_config.get('log_file', 'fire_clustering.log')
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    fh.setFormatter(file_formatter)
    root_logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(console_formatter)
    root_logger.addHandler(ch)
    
    return root_logger

def create_output_directory(config):
    """Create output directory structure"""
    base_dir = config['data']['output_base_dir']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if config['study_area']['test_mode']:
        run_dir = Path(base_dir) / f"test_run_{timestamp}"
    else:
        run_dir = Path(base_dir) / f"production_run_{timestamp}"
    
    # Create subdirectories
    subdirs = ['episodes', 'validation', 'checkpoints', 'logs', 'visualizations']
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return run_dir

def chunk_data_spatially(data, chunk_size_deg, overlap_deg=0.1):
    """Chunk data into spatial tiles with overlap"""
    min_lon = data['longitude'].min()
    max_lon = data['longitude'].max()
    min_lat = data['latitude'].min()
    max_lat = data['latitude'].max()
    
    chunks = []
    
    lon_start = min_lon
    while lon_start < max_lon:
        lon_end = min(lon_start + chunk_size_deg, max_lon + overlap_deg)
        
        lat_start = min_lat
        while lat_start < max_lat:
            lat_end = min(lat_start + chunk_size_deg, max_lat + overlap_deg)
            
            # Select data in chunk
            mask = (
                (data['longitude'] >= lon_start - overlap_deg) &
                (data['longitude'] <= lon_end) &
                (data['latitude'] >= lat_start - overlap_deg) &
                (data['latitude'] <= lat_end)
            )
            
            if mask.sum() > 0:
                chunk_info = {
                    'bounds': [lon_start, lat_start, lon_end, lat_end],
                    'data': data[mask].copy(),
                    'chunk_id': f"{lon_start:.1f}_{lat_start:.1f}"
                }
                chunks.append(chunk_info)
            
            lat_start += chunk_size_deg - overlap_deg
        
        lon_start += chunk_size_deg - overlap_deg
    
    return chunks

def calculate_temporal_distance(times1, times2, max_days=None):
    """Calculate temporal distance matrix in days"""
    # Convert to numpy datetime64 if needed
    if not isinstance(times1, np.ndarray):
        times1 = np.array(times1, dtype='datetime64[ns]')
    if not isinstance(times2, np.ndarray):
        times2 = np.array(times2, dtype='datetime64[ns]')
    
    # Calculate differences in days
    diff = np.abs(times1[:, np.newaxis] - times2[np.newaxis, :])
    days = diff / np.timedelta64(1, 'D')
    
    # Apply maximum threshold if specified
    if max_days is not None:
        days = np.minimum(days, max_days)
    
    return days

def merge_overlapping_clusters(clusters_list, overlap_threshold=0.8):
    """Merge clusters from overlapping spatial chunks"""
    # Implementation would handle merging clusters that span chunk boundaries
    # This is a placeholder for the actual implementation
    merged_clusters = {}
    cluster_id = 0
    
    for clusters in clusters_list:
        for old_id, points in clusters.items():
            # Check for overlap with existing clusters
            # For now, just assign new IDs
            merged_clusters[cluster_id] = points
            cluster_id += 1
    
    return merged_clusters

def save_checkpoint(data, checkpoint_path, metadata=None):
    """Save checkpoint for recovery"""
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'data': data,
        'metadata': metadata or {}
    }
    
    # Use pickle for complex objects or parquet for DataFrames
    import pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logging.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint for recovery"""
    import pickle
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint['data'], checkpoint['metadata']

def format_duration(seconds):
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s" 