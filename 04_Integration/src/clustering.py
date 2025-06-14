#!/usr/bin/env python3
"""
Spatiotemporal clustering module for fire episode detection
Implements DBSCAN with custom distance metrics
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from utils import SpatialUtils, calculate_temporal_distance, GPU_AVAILABLE

# Try to import RAPIDS for GPU acceleration
try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

class SpatioTemporalDBSCAN:
    """
    Custom DBSCAN implementation for spatiotemporal clustering of fire detections
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('SpatioTemporalDBSCAN')
        
        # Extract clustering parameters
        cluster_config = config['clustering']
        self.spatial_eps = cluster_config['spatial_eps_meters']
        self.temporal_eps = cluster_config['temporal_eps_days']
        self.min_samples = cluster_config['min_samples']
        self.handle_day_night = cluster_config['handle_day_night']
        
        # Advanced parameters
        self.day_night_weight = cluster_config.get('day_night_weight', 0.8)
        self.use_confidence_weight = cluster_config.get('confidence_weighting', True)
        self.use_frp_weight = cluster_config.get('frp_weighting', False)
        
        # Performance parameters
        self.use_gpu = config['processing']['use_gpu_acceleration'] and GPU_AVAILABLE
        self.n_jobs = config['processing']['max_cpu_cores']
        
        self.logger.info(f"Initialized with eps_spatial={self.spatial_eps}m, "
                        f"eps_temporal={self.temporal_eps}d, min_samples={self.min_samples}")
        
    def fit_predict(self, coords, times, features=None):
        """
        Perform spatiotemporal clustering on fire detections
        
        Args:
            coords: numpy array of shape (n_points, 2) with projected coordinates
            times: numpy array of datetime objects
            features: dict of additional features (confidence, frp, daynight, etc.)
            
        Returns:
            labels: numpy array of cluster labels (-1 for noise)
        """
        n_points = len(coords)
        self.logger.info(f"Clustering {n_points:,} points")
        
        # Handle different day/night strategies
        if self.handle_day_night == "separate":
            labels = self._cluster_separate_daynight(coords, times, features)
        elif self.handle_day_night == "weighted":
            labels = self._cluster_weighted(coords, times, features)
        else:  # "combined"
            labels = self._cluster_combined(coords, times)
        
        # Post-process clusters
        labels = self._post_process_clusters(labels, coords, times, features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        self.logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        return labels
    
    def _cluster_combined(self, coords, times):
        """Standard combined spatiotemporal clustering"""
        if self.use_gpu and RAPIDS_AVAILABLE:
            return self._cluster_gpu(coords, times)
        else:
            return self._cluster_cpu(coords, times)
    
    def _cluster_cpu(self, coords, times):
        """CPU-based clustering using custom distance metric"""
        # Calculate distance matrix
        self.logger.info("Calculating spatiotemporal distance matrix (CPU)")
        
        # For large datasets, use a chunked approach
        if len(coords) > 50000:
            return self._cluster_cpu_chunked(coords, times)
        
        # Calculate spatial distances
        spatial_dist = pairwise_distances(coords, metric='euclidean')
        
        # Calculate temporal distances
        temporal_dist = calculate_temporal_distance(times, times)
        
        # Normalize and combine distances
        spatial_normalized = spatial_dist / self.spatial_eps
        temporal_normalized = temporal_dist / self.temporal_eps
        
        # Combined distance (both conditions must be met)
        combined_dist = np.maximum(spatial_normalized, temporal_normalized)
        
        # Run DBSCAN with precomputed distances
        dbscan = DBSCAN(
            eps=1.0,  # Normalized threshold
            min_samples=self.min_samples,
            metric='precomputed',
            n_jobs=self.n_jobs
        )
        
        labels = dbscan.fit_predict(combined_dist)
        
        return labels
    
    def _cluster_cpu_chunked(self, coords, times):
        """Chunked CPU clustering for large datasets"""
        self.logger.info("Using chunked clustering for large dataset")
        
        # Create spatial index for efficient neighbor search
        tree = BallTree(coords, metric='euclidean')
        
        # Initialize labels
        labels = np.full(len(coords), -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(len(coords), dtype=bool)
        
        # Process points
        for i in tqdm(range(len(coords)), desc="Clustering"):
            if visited[i]:
                continue
                
            # Find spatial neighbors
            spatial_neighbors = tree.query_radius(
                [coords[i]], r=self.spatial_eps
            )[0]
            
            # Filter by temporal distance
            time_diffs = np.abs((times[spatial_neighbors] - times[i]) / np.timedelta64(1, 'D'))
            neighbors = spatial_neighbors[time_diffs <= self.temporal_eps]
            
            if len(neighbors) >= self.min_samples:
                # Start new cluster
                labels[i] = cluster_id
                visited[i] = True
                
                # Expand cluster
                seed_set = list(neighbors)
                j = 0
                
                while j < len(seed_set):
                    q = seed_set[j]
                    
                    if not visited[q]:
                        visited[q] = True
                        
                        # Find neighbors of q
                        q_spatial_neighbors = tree.query_radius(
                            [coords[q]], r=self.spatial_eps
                        )[0]
                        
                        q_time_diffs = np.abs((times[q_spatial_neighbors] - times[q]) / np.timedelta64(1, 'D'))
                        q_neighbors = q_spatial_neighbors[q_time_diffs <= self.temporal_eps]
                        
                        if len(q_neighbors) >= self.min_samples:
                            for neighbor in q_neighbors:
                                if neighbor not in seed_set:
                                    seed_set.append(neighbor)
                        
                        if labels[q] == -1:
                            labels[q] = cluster_id
                    
                    j += 1
                
                cluster_id += 1
            else:
                visited[i] = True
        
        return labels
    
    def _cluster_gpu(self, coords, times):
        """GPU-accelerated clustering using RAPIDS"""
        self.logger.info("Using GPU-accelerated clustering")
        
        # Convert times to numeric for GPU processing
        times_numeric = (times - times.min()) / np.timedelta64(1, 'D')
        
        # Combine spatial and temporal features
        features_combined = np.column_stack([
            coords / self.spatial_eps,
            times_numeric.reshape(-1, 1) / self.temporal_eps
        ])
        
        # Use RAPIDS DBSCAN
        clusterer = cuDBSCAN(
            eps=1.0,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(features_combined)
        
        return labels
    
    def _cluster_separate_daynight(self, coords, times, features):
        """Cluster day and night detections separately then merge"""
        self.logger.info("Clustering day and night detections separately")
        
        daynight = features['daynight']
        labels = np.full(len(coords), -1, dtype=int)
        
        # Cluster day detections
        day_mask = daynight == 'D'
        if day_mask.sum() > 0:
            day_labels = self._cluster_combined(
                coords[day_mask], times[day_mask]
            )
            labels[day_mask] = day_labels
        
        # Cluster night detections
        night_mask = daynight == 'N'
        if night_mask.sum() > 0:
            night_labels = self._cluster_combined(
                coords[night_mask], times[night_mask]
            )
            # Offset night cluster IDs
            night_labels[night_labels >= 0] += labels.max() + 1
            labels[night_mask] = night_labels
        
        # Merge nearby day/night clusters
        labels = self._merge_day_night_clusters(labels, coords, times, daynight)
        
        return labels
    
    def _cluster_weighted(self, coords, times, features):
        """Weighted clustering considering confidence and other features"""
        self.logger.info("Using weighted clustering with additional features")
        
        # Build feature matrix
        feature_list = [coords / self.spatial_eps]
        
        # Add temporal feature
        times_numeric = (times - times.min()) / np.timedelta64(1, 'D')
        feature_list.append(times_numeric.reshape(-1, 1) / self.temporal_eps)
        
        # Add confidence weighting if enabled
        if self.use_confidence_weight and 'confidence' in features:
            confidence_normalized = (features['confidence'] - 50) / 50  # Normalize around 50%
            feature_list.append(confidence_normalized.reshape(-1, 1) * 0.5)  # Weight factor
        
        # Add FRP weighting if enabled
        if self.use_frp_weight and 'frp' in features:
            frp_log = np.log1p(features['frp'])
            frp_normalized = (frp_log - frp_log.mean()) / frp_log.std()
            feature_list.append(frp_normalized.reshape(-1, 1) * 0.3)  # Weight factor
        
        # Combine features
        features_combined = np.hstack(feature_list)
        
        # Run DBSCAN
        dbscan = DBSCAN(
            eps=1.0,
            min_samples=self.min_samples,
            metric='euclidean',
            n_jobs=self.n_jobs
        )
        
        labels = dbscan.fit_predict(features_combined)
        
        return labels
    
    def _merge_day_night_clusters(self, labels, coords, times, daynight):
        """Merge day and night clusters that are close in space and time"""
        self.logger.info("Merging adjacent day/night clusters")
        
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) < 2:
            return labels
        
        # Calculate cluster properties
        cluster_props = {}
        for label in unique_labels:
            mask = labels == label
            cluster_props[label] = {
                'centroid': coords[mask].mean(axis=0),
                'time_mean': times[mask].mean(),
                'time_min': times[mask].min(),
                'time_max': times[mask].max(),
                'is_day': (daynight[mask] == 'D').any(),
                'is_night': (daynight[mask] == 'N').any()
            }
        
        # Find clusters to merge
        merge_pairs = []
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                props1 = cluster_props[label1]
                props2 = cluster_props[label2]
                
                # Check if one is day and other is night
                if not ((props1['is_day'] and props2['is_night']) or 
                       (props1['is_night'] and props2['is_day'])):
                    continue
                
                # Check spatial distance
                spatial_dist = np.linalg.norm(props1['centroid'] - props2['centroid'])
                if spatial_dist > self.spatial_eps * 2:
                    continue
                
                # Check temporal overlap or proximity
                time_gap = max(0, (min(props2['time_min'], props1['time_min']) - 
                                 max(props2['time_max'], props1['time_max'])) / np.timedelta64(1, 'D'))
                
                if time_gap <= 1:  # Within 1 day
                    merge_pairs.append((label1, label2))
        
        # Merge clusters
        if merge_pairs:
            self.logger.info(f"Merging {len(merge_pairs)} day/night cluster pairs")
            
            # Create mapping
            label_map = {i: i for i in unique_labels}
            for label1, label2 in merge_pairs:
                # Map higher label to lower
                label_map[max(label1, label2)] = min(label1, label2)
            
            # Apply transitive closure
            changed = True
            while changed:
                changed = False
                for k, v in label_map.items():
                    if label_map[v] != v:
                        label_map[k] = label_map[v]
                        changed = True
            
            # Apply mapping
            new_labels = labels.copy()
            for old_label, new_label in label_map.items():
                new_labels[labels == old_label] = new_label
            
            return new_labels
        
        return labels
    
    def _post_process_clusters(self, labels, coords, times, features):
        """Post-process clusters to handle edge cases"""
        # Remove very small clusters
        min_cluster_size = max(self.min_samples, 3)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
        small_clusters = unique_labels[counts < min_cluster_size]
        if len(small_clusters) > 0:
            self.logger.info(f"Removing {len(small_clusters)} small clusters")
            for label in small_clusters:
                labels[labels == label] = -1
        
        # Renumber clusters sequentially
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) > 0:
            label_map = {old: new for new, old in enumerate(unique_labels)}
            new_labels = labels.copy()
            for old_label, new_label in label_map.items():
                new_labels[labels == old_label] = new_label
            labels = new_labels
        
        return labels
    
    def parameter_optimization(self, coords, times, features, param_ranges):
        """
        Optimize DBSCAN parameters using grid search
        """
        self.logger.info("Starting parameter optimization")
        
        results = []
        
        # Grid search over parameter ranges
        for spatial_eps in param_ranges['spatial_eps']:
            for temporal_eps in param_ranges['temporal_eps']:
                for min_samples in param_ranges['min_samples']:
                    # Update parameters
                    self.spatial_eps = spatial_eps
                    self.temporal_eps = temporal_eps
                    self.min_samples = min_samples
                    
                    # Run clustering
                    labels = self.fit_predict(coords, times, features)
                    
                    # Calculate metrics
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = (labels == -1).sum()
                    noise_ratio = n_noise / len(labels)
                    
                    # Calculate silhouette score for non-noise points
                    if n_clusters > 1 and n_noise < len(labels) - 10:
                        from sklearn.metrics import silhouette_score
                        non_noise_mask = labels >= 0
                        if non_noise_mask.sum() > 10:
                            try:
                                # Combine spatial and temporal features for silhouette
                                times_numeric = (times - times.min()) / np.timedelta64(1, 'D')
                                combined_features = np.column_stack([
                                    coords / 1000,  # km
                                    times_numeric.reshape(-1, 1)
                                ])
                                silhouette = silhouette_score(
                                    combined_features[non_noise_mask],
                                    labels[non_noise_mask]
                                )
                            except:
                                silhouette = -1
                        else:
                            silhouette = -1
                    else:
                        silhouette = -1
                    
                    result = {
                        'spatial_eps': spatial_eps,
                        'temporal_eps': temporal_eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette_score': silhouette
                    }
                    
                    results.append(result)
                    self.logger.info(f"Params: eps_s={spatial_eps}, eps_t={temporal_eps}, "
                                   f"min_pts={min_samples} -> {n_clusters} clusters, "
                                   f"{noise_ratio:.1%} noise, silhouette={silhouette:.3f}")
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal parameters (balance between cluster count and noise)
        # Prefer solutions with reasonable noise ratio and good silhouette score
        results_df['score'] = (
            results_df['silhouette_score'] * 0.5 +
            (1 - results_df['noise_ratio']) * 0.3 +
            np.clip(results_df['n_clusters'] / 100, 0, 1) * 0.2
        )
        
        best_params = results_df.loc[results_df['score'].idxmax()]
        
        self.logger.info(f"Best parameters: eps_spatial={best_params['spatial_eps']}m, "
                       f"eps_temporal={best_params['temporal_eps']}d, "
                       f"min_samples={best_params['min_samples']}")
        
        return results_df, best_params 