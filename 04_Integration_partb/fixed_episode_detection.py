#!/usr/bin/env python3
"""
Fixed Fire Episode Detection System
Scientifically sound spatiotemporal clustering for MODIS fire data
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union, transform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class ImprovedFireEpisodeDetector:
    """
    Scientifically sound fire episode detection
    Based on fire behavior principles
    """
    
    def __init__(self, config=None):
        """Initialize with science-based parameters"""
        
        # Science-based clustering parameters
        # Based on: Archibald & Roy (2009), Andela et al. (2019)
        self.spatial_threshold_m = 2500  # 2.5km - accounts for MODIS pixel size + fire spread
        self.temporal_threshold_days = 11  # 11 days - based on African fire studies
        self.min_detections = 2  # Minimum detections to form episode
        
        # Fire behavior parameters
        self.max_spread_rate_kmh = 10  # Maximum realistic fire spread rate
        self.pixel_size_m = 1000  # MODIS pixel size
        self.revisit_time_hours = 6  # Approximate satellite revisit
        
        print("Initialized Improved Fire Episode Detector")
        print(f"Spatial threshold: {self.spatial_threshold_m}m")
        print(f"Temporal threshold: {self.temporal_threshold_days} days")
    
    def detect_episodes(self, fire_df):
        """
        Detect fire episodes using improved algorithm
        
        Args:
            fire_df: DataFrame with fire detections (must have x_proj, y_proj, acq_datetime)
        
        Returns:
            fire_df with episode_id column
            episodes_gdf with proper geometries
        """
        print(f"\nDetecting fire episodes from {len(fire_df)} detections...")
        
        # Step 1: Prepare arrays
        coords = fire_df[['x_proj', 'y_proj']].values.astype('float64')
        times = pd.to_datetime(fire_df['acq_datetime']).values
        time_days = times.astype('datetime64[s]').astype(float) / 86400.0
        
        # Step 2: Custom DBSCAN using spatial KDTree and exact temporal + spread constraints
        print("Clustering with custom DBSCAN (KDTree + time and spread constraints)...")
        tree = KDTree(coords)  # Euclidean in projected meters
        spatial_eps = float(self.spatial_threshold_m)
        temporal_eps_days = float(self.temporal_threshold_days)
        max_speed_km_per_day = float(self.max_spread_rate_kmh) * 24.0

        n = coords.shape[0]
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        from collections import deque

        def region_query(idx):
            # Spatial neighbors within spatial_eps
            neighbor_idx = tree.query_radius(coords[idx].reshape(1, -1), r=spatial_eps, return_distance=False)[0]
            if neighbor_idx.size == 0:
                return np.array([], dtype=int)
            # Filter by temporal window and spread-rate
            dt = np.abs(time_days[neighbor_idx] - time_days[idx])
            mask_time = dt <= temporal_eps_days
            candidate = neighbor_idx[mask_time]
            if candidate.size == 0:
                return np.array([], dtype=int)
            # Spread-rate constraint
            dx = coords[candidate, 0] - coords[idx, 0]
            dy = coords[candidate, 1] - coords[idx, 1]
            dist_m = np.hypot(dx, dy)
            # Avoid division by zero: allow if dt=0 (same time)
            allowed_m = (max_speed_km_per_day * np.maximum(dt[mask_time], 0.0)) * 1000.0
            keep = (dt[mask_time] == 0.0) | (dist_m <= allowed_m)
            return candidate[keep]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = region_query(i)
            if neighbors.size < self.min_detections:
                labels[i] = -1
                continue
            # Start new cluster
            labels[i] = cluster_id
            queue = deque([j for j in neighbors if j != i])
            while queue:
                j = queue.popleft()
                if not visited[j]:
                    visited[j] = True
                    neighbors_j = region_query(j)
                    if neighbors_j.size >= self.min_detections:
                        # Merge neighbor sets
                        for nb in neighbors_j:
                            if nb != j:
                                queue.append(nb)
                if labels[j] == -1:
                    labels[j] = cluster_id
                elif labels[j] == -1 or labels[j] == cluster_id:
                    labels[j] = cluster_id
            cluster_id += 1
        
        # Step 4: Post-process clusters
        labels = self._post_process_clusters(fire_df, labels, coords, times)
        
        # Step 5: Add to dataframe
        fire_df['episode_id'] = labels
        
        # Step 6: Create episode geometries (CRITICAL FIX)
        episodes_gdf = self._create_episode_geometries(fire_df)
        
        # Statistics
        n_episodes = len(episodes_gdf)
        n_noise = (labels == -1).sum()
        print(f"\nResults: {n_episodes} episodes detected, {n_noise} noise points")
        
        return fire_df, episodes_gdf
    
    def _compute_spatiotemporal_distance(self, coords, times):
        """
        Compute normalized spatiotemporal distance matrix
        Key fix: Proper normalization and thresholding
        """
        n_points = len(coords)
        
        # Spatial distances
        spatial_dist = cdist(coords, coords, metric='euclidean')
        
        # Temporal distances in days
        time_days = times.astype('datetime64[s]').astype(float) / 86400
        temporal_dist = np.abs(time_days[:, np.newaxis] - time_days[np.newaxis, :])
        
        # Normalize distances
        spatial_norm = spatial_dist / self.spatial_threshold_m
        temporal_norm = temporal_dist / self.temporal_threshold_days
        
        # Combined distance: both conditions must be met
        # Use L-infinity norm (maximum) to ensure both constraints
        combined_dist = np.maximum(spatial_norm, temporal_norm)
        
        # Apply fire spread rate constraint
        # Points can't be connected if spread rate would be unrealistic
        for i in range(n_points):
            for j in range(i+1, n_points):
                if temporal_dist[i, j] > 0:
                    required_speed_kmh = (spatial_dist[i, j] / 1000) / (temporal_dist[i, j] * 24)
                    if required_speed_kmh > self.max_spread_rate_kmh:
                        combined_dist[i, j] = combined_dist[j, i] = np.inf
        
        return combined_dist
    
    def _post_process_clusters(self, fire_df, labels, coords, times):
        """Post-process clusters to ensure quality"""
        
        unique_labels = np.unique(labels[labels >= 0])
        final_labels = labels.copy()
        
        for label in unique_labels:
            mask = labels == label
            cluster_points = coords[mask]
            cluster_times = times[mask]
            
            # Check cluster quality
            if len(cluster_points) < self.min_detections:
                final_labels[mask] = -1
                continue
            
            # Check temporal continuity
            time_range = (cluster_times.max() - cluster_times.min()) / np.timedelta64(1, 'D')
            unique_days = len(np.unique(cluster_times.astype('datetime64[D]')))
            
            if unique_days < 2 and time_range > 1:
                # Single day detection spread over time - likely false cluster
                final_labels[mask] = -1
                continue
            
            # Check spatial coherence
            if len(cluster_points) > 3:
                # Calculate spatial standard deviation
                spatial_std = np.sqrt(np.var(cluster_points[:, 0]) + np.var(cluster_points[:, 1]))
                
                # If too dispersed relative to number of points, split or mark as noise
                max_expected_std = self.spatial_threshold_m * np.sqrt(len(cluster_points))
                if spatial_std > max_expected_std:
                    final_labels[mask] = -1
        
        # Renumber clusters sequentially
        unique_final = np.unique(final_labels[final_labels >= 0])
        label_map = {old: new for new, old in enumerate(unique_final)}
        
        for old_label, new_label in label_map.items():
            final_labels[final_labels == old_label] = new_label
        
        return final_labels

    
    def _create_episode_geometries(self, fire_df):
        """
        Create accurate fire episode geometries
        CRITICAL: Use convex hull or alpha shape, not bounding box!
        """
        episodes = []
        
        # Group by episode
        grouped = fire_df[fire_df['episode_id'] >= 0].groupby('episode_id')
        
        for episode_id, group in grouped:
            # Get fire points
            points = [Point(row['longitude'], row['latitude']) 
                     for _, row in group.iterrows()]
            
            # Create geometry based on number of points
            if len(points) == 1:
                # Single point - buffer by MODIS pixel size
                geometry = points[0].buffer(0.01)  # ~1km in degrees
            elif len(points) == 2:
                # Two points - create line buffer
                line = MultiPoint(points).convex_hull
                geometry = line.buffer(0.01)
            else:
                # Multiple points - use convex hull
                multipoint = MultiPoint(points)
                geometry = multipoint.convex_hull
                
                # Add small buffer to account for pixel size
                geometry = geometry.buffer(0.005)  # ~500m buffer
            
            # Calculate episode metrics
            episode_record = {
                'episode_id': episode_id,
                'geometry': geometry,
                'start_date': group['acq_datetime'].min(),
                'end_date': group['acq_datetime'].max(),
                'duration_days': (group['acq_datetime'].max() - group['acq_datetime'].min()).days + 1,
                'n_detections': len(group),
                'mean_frp': group['frp'].mean(),
                'max_frp': group['frp'].max(),
                'total_frp': group['frp'].sum(),
                'mean_confidence': group['confidence'].mean(),
                'unique_days': group['acq_date'].nunique(),
                'day_detections': (group['daynight'] == 'D').sum(),
                'night_detections': (group['daynight'] == 'N').sum()
            }
            
            episodes.append(episode_record)
        
        # Create GeoDataFrame
        episodes_gdf = gpd.GeoDataFrame(episodes, crs='EPSG:4326')
        
        # Project to equal area for accurate area calculation
        episodes_gdf_proj = episodes_gdf.to_crs('EPSG:6933')
        episodes_gdf['area_km2'] = episodes_gdf_proj.geometry.area / 1e6
        
        return episodes_gdf


class FireRegimeMetrics:
    """Calculate comprehensive fire regime metrics for watersheds"""
    
    def __init__(self):
        """Initialize fire regime calculator"""
        self.metrics = {}
    
    def calculate_watershed_fire_metrics(self, episodes_gdf, watersheds_gdf):
        """
        Calculate scientifically sound fire metrics for each watershed
        
        Returns:
            watersheds_gdf with fire regime metrics
        """
        print("\nCalculating watershed fire regime metrics...")
        
        # Ensure CRS match
        if episodes_gdf.crs != watersheds_gdf.crs:
            episodes_gdf = episodes_gdf.to_crs(watersheds_gdf.crs)
        
        # Memory-safe intersection: iterate watersheds and intersect only nearby episodes
        print("Performing spatial intersection (streaming, memory-safe)...")
        if episodes_gdf.empty:
            print("No fire episodes available")
            return self._add_empty_metrics(watersheds_gdf)
        
        # Spatial index on episodes
        episodes_sindex = episodes_gdf.sindex
        
        # Projector for accurate area calculation
        try:
            import pyproj
            projector = pyproj.Transformer.from_crs(watersheds_gdf.crs, 'EPSG:6933', always_xy=True).transform
        except Exception:
            projector = None
        
        metrics_list = []
        
        # Iterate watersheds one by one to keep memory bounded
        for _, watershed in watersheds_gdf.iterrows():
            huc12 = watershed['huc12']
            watershed_geom = watershed.geometry
            watershed_area = float(watershed['area_km2'])
            
            # Candidate episodes by bbox intersect
            candidate_idx = list(episodes_sindex.query(watershed_geom, predicate='intersects'))
            if len(candidate_idx) == 0:
                metrics = self._get_empty_metrics()
                metrics['huc12'] = huc12
                metrics_list.append(metrics)
                continue
            
            candidates = episodes_gdf.iloc[candidate_idx].copy()
            # Exact intersection test and area
            episode_rows = []
            for _, ep in candidates.iterrows():
                if not ep.geometry.intersects(watershed_geom):
                    continue
                inter_geom = ep.geometry.intersection(watershed_geom)
                if inter_geom.is_empty:
                    continue
                # Compute area in equal-area projection
                if projector is not None:
                    inter_area_m2 = transform(projector, inter_geom).area
                    inter_area_km2 = inter_area_m2 / 1e6
                else:
                    # Fallback: approximate using current CRS area
                    inter_area_km2 = inter_geom.area / 1e6
                if inter_area_km2 <= 0:
                    continue
                episode_rows.append({
                    'intersection_area_km2': inter_area_km2,
                    'n_detections': ep.get('n_detections', 0),
                    'total_frp': ep.get('total_frp', 0.0),
                    'max_frp': ep.get('max_frp', 0.0),
                    'mean_frp': ep.get('mean_frp', 0.0),
                    'duration_days': ep.get('duration_days', 0),
                    'start_date': ep.get('start_date', pd.NaT),
                    'day_detections': ep.get('day_detections', 0)
                })
            
            if len(episode_rows) == 0:
                metrics = self._get_empty_metrics()
                metrics['huc12'] = huc12
            else:
                eps_df = pd.DataFrame(episode_rows)
                metrics = self._calculate_metrics_for_watershed(eps_df, watershed_area)
                metrics['huc12'] = huc12
            metrics_list.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        watersheds_with_fire = watersheds_gdf.merge(metrics_df, on='huc12', how='left')
        watersheds_with_fire = watersheds_with_fire.fillna(0)
        watersheds_with_fire['hsbf'] = (
            watersheds_with_fire['max_single_fire_area_km2'] /
            watersheds_with_fire['area_km2']
        ).clip(upper=1.0)
        
        print(f"Calculated metrics for {len(watersheds_gdf)} watersheds")
        print(f"Watersheds with fires: {(watersheds_with_fire['episode_count'] > 0).sum()}")
        print(f"Maximum HSBF: {watersheds_with_fire['hsbf'].max():.3f}")
        
        return watersheds_with_fire
    
    def _calculate_metrics_for_watershed(self, episodes, watershed_area):
        """Calculate comprehensive fire regime metrics"""
        
        metrics = {
            # Basic counts
            'episode_count': len(episodes),
            'total_detections': episodes['n_detections'].sum(),
            
            # Area metrics (using intersection areas!)
            'total_burned_area_km2': episodes['intersection_area_km2'].sum(),
            'max_single_fire_area_km2': episodes['intersection_area_km2'].max(),
            'mean_fire_area_km2': episodes['intersection_area_km2'].mean(),
            
            # Intensity metrics
            'total_frp_mw': episodes['total_frp'].sum(),
            'max_frp_mw': episodes['max_frp'].max(),
            'mean_frp_mw': episodes['mean_frp'].mean(),
            
            # Temporal metrics
            'total_fire_days': episodes['duration_days'].sum(),
            'max_duration_days': episodes['duration_days'].max(),
            'mean_duration_days': episodes['duration_days'].mean(),
            
            # Fire return interval (years between fires)
            'fire_return_interval_years': self._calculate_return_interval(episodes),
            
            # Seasonality
            'peak_fire_month': self._get_peak_month(episodes),
            'seasonality_index': self._calculate_seasonality(episodes),
            
            # Day/night activity
            'day_detection_ratio': episodes['day_detections'].sum() / 
                                  max(1, episodes['n_detections'].sum()),
            
            # Fire size distribution parameters
            'fire_size_cv': episodes['intersection_area_km2'].std() / 
                           max(0.001, episodes['intersection_area_km2'].mean()),
            
            # Proportion metrics
            'area_burned_fraction': min(1.0, episodes['intersection_area_km2'].sum() / watershed_area),
            
            # Severity proxy (high FRP fires)
            'high_intensity_count': (episodes['max_frp'] > episodes['max_frp'].quantile(0.9)).sum(),
        }
        
        return metrics
    
    def _calculate_return_interval(self, episodes):
        """Calculate mean fire return interval"""
        if len(episodes) < 2:
            return np.nan
        
        # Get unique fire dates
        fire_dates = pd.to_datetime(episodes['start_date']).sort_values()
        
        # Calculate intervals in years
        intervals = np.diff(fire_dates) / np.timedelta64(365, 'D')
        
        return intervals.mean() if len(intervals) > 0 else np.nan
    
    def _get_peak_month(self, episodes):
        """Get month with most fire activity"""
        if len(episodes) == 0:
            return 0
        
        months = pd.to_datetime(episodes['start_date']).dt.month
        if len(months) > 0:
            return months.mode()[0] if len(months.mode()) > 0 else months.iloc[0]
        return 0
    
    def _calculate_seasonality(self, episodes):
        """Calculate seasonality index (0=uniform, 1=highly seasonal)"""
        if len(episodes) < 3:
            return 0
        
        months = pd.to_datetime(episodes['start_date']).dt.month
        month_counts = months.value_counts()
        
        # Fill missing months with zeros
        all_months = pd.Series(0, index=range(1, 13))
        all_months.update(month_counts)
        
        # Calculate coefficient of variation
        cv = all_months.std() / max(0.001, all_months.mean())
        
        # Normalize to 0-1 scale
        return min(1.0, cv / 3.46)  # Max CV for monthly data is ~3.46
    
    def _get_empty_metrics(self):
        """Return empty metrics dictionary"""
        return {
            'episode_count': 0,
            'total_detections': 0,
            'total_burned_area_km2': 0,
            'max_single_fire_area_km2': 0,
            'mean_fire_area_km2': 0,
            'total_frp_mw': 0,
            'max_frp_mw': 0,
            'mean_frp_mw': 0,
            'total_fire_days': 0,
            'max_duration_days': 0,
            'mean_duration_days': 0,
            'fire_return_interval_years': np.nan,
            'peak_fire_month': 0,
            'seasonality_index': 0,
            'day_detection_ratio': 0,
            'fire_size_cv': 0,
            'area_burned_fraction': 0,
            'high_intensity_count': 0
        }
    
    def _add_empty_metrics(self, watersheds_gdf):
        """Add empty metrics to watersheds with no fires"""
        empty_metrics = self._get_empty_metrics()
        
        for col, value in empty_metrics.items():
            watersheds_gdf[col] = value
        
        watersheds_gdf['hsbf'] = 0
        
        return watersheds_gdf