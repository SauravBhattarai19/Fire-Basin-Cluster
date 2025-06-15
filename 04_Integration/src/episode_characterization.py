#!/usr/bin/env python3
"""
Fire episode characterization module
Generates comprehensive episode records from clustered fire detections
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from scipy.stats import circmean
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils import SpatialUtils

class EpisodeCharacterization:
    """
    Generate comprehensive fire episode records from clustered detections
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('EpisodeCharacterization')
        self.spatial_utils = SpatialUtils(config['study_area']['output_epsg'])
        
        # Episode metric settings
        self.metrics_config = config['episode_metrics']
        self.dormancy_threshold = self.metrics_config['dormancy_threshold_days']
        self.rekindle_max_days = self.metrics_config['rekindle_max_days']
        
    def characterize_episodes(self, fire_df, labels):
        """
        Generate episode records from clustered fire detections
        
        Args:
            fire_df: DataFrame with fire detections
            labels: Cluster labels from DBSCAN
            
        Returns:
            episodes_df: DataFrame with episode characteristics
        """
        self.logger.info("Characterizing fire episodes")
        
        # Add cluster labels to dataframe
        fire_df['cluster_id'] = labels
        
        # Filter out noise points
        clustered_df = fire_df[fire_df['cluster_id'] >= 0].copy()
        
        if len(clustered_df) == 0:
            self.logger.warning("No clusters found")
            return pd.DataFrame()
        
        # Group by cluster and generate episode records
        episodes = []
        unique_clusters = clustered_df['cluster_id'].unique()
        
        for cluster_id in tqdm(unique_clusters, desc="Characterizing episodes"):
            cluster_data = clustered_df[clustered_df['cluster_id'] == cluster_id]
            episode = self._characterize_single_episode(cluster_data, cluster_id)
            episodes.append(episode)
        
        episodes_df = pd.DataFrame(episodes)
        
        # Post-process episodes
        episodes_df = self._post_process_episodes(episodes_df)
        
        self.logger.info(f"Generated {len(episodes_df)} episode records")
        
        return episodes_df
    
    def _characterize_single_episode(self, cluster_data, episode_id):
        """Characterize a single fire episode"""
        
        # Basic temporal metrics
        temporal_metrics = self._calculate_temporal_metrics(cluster_data)
        
        # Spatial metrics
        spatial_metrics = self._calculate_spatial_metrics(cluster_data)
        
        # Intensity metrics
        intensity_metrics = self._calculate_intensity_metrics(cluster_data)
        
        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(cluster_data)
        
        # Behavior patterns
        behavior_metrics = self._calculate_behavior_metrics(cluster_data)
        
        # Combine all metrics
        episode = {
            'episode_id': episode_id,
            **temporal_metrics,
            **spatial_metrics,
            **intensity_metrics,
            **quality_metrics,
            **behavior_metrics
        }
        
        return episode
    
    def _calculate_temporal_metrics(self, cluster_data):
        """Calculate temporal characteristics of episode"""
        
        # Basic temporal extent
        start_time = cluster_data['acq_datetime'].min()
        end_time = cluster_data['acq_datetime'].max()
        duration = end_time - start_time
        
        # Active days calculation
        unique_dates = cluster_data['acq_date'].unique()
        active_days = len(unique_dates)
        
        # Dormancy analysis
        if active_days > 1:
            dates_sorted = pd.Series(pd.to_datetime(unique_dates)).sort_values()
            date_gaps = np.diff(dates_sorted) / np.timedelta64(1, 'D')
            dormancy_periods = (date_gaps > self.dormancy_threshold).sum()
            max_dormancy = date_gaps.max() if len(date_gaps) > 0 else 0
        else:
            dormancy_periods = 0
            max_dormancy = 0
        
        # Daily detection pattern
        daily_counts = cluster_data.groupby('acq_date').size()
        
        metrics = {
            'start_datetime': start_time,
            'end_datetime': end_time,
            'duration_hours': duration.total_seconds() / 3600,
            'duration_days': duration.days + duration.seconds / 86400,
            'active_days': active_days,
            'dormancy_periods': dormancy_periods,
            'max_dormancy_days': max_dormancy,
            'mean_daily_detections': daily_counts.mean(),
            'max_daily_detections': daily_counts.max(),
            'detection_consistency': active_days / (duration.days + 1) if duration.days >= 0 else 1.0
        }
        
        return metrics
    
    def _calculate_spatial_metrics(self, cluster_data):
        """Calculate spatial characteristics of episode"""
        
        # Get coordinates
        coords = cluster_data[['x_proj', 'y_proj']].values
        coords_wgs84 = cluster_data[['longitude', 'latitude']].values
        
        # Centroid
        centroid = coords.mean(axis=0)
        centroid_lon, centroid_lat = self.spatial_utils.transform_points(
            [centroid[0]], [centroid[1]], to_target=False
        )
        
        # Bounding box
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # Area estimation using convex hull
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                hull_area = hull.volume  # In 2D, volume is area
                hull_points = coords[hull.vertices]
                
                # Shape elongation
                if len(hull_points) >= 3:
                    # Use eigenvalues of covariance matrix
                    cov_matrix = np.cov(hull_points.T)
                    eigenvalues = np.linalg.eigvals(cov_matrix)
                    elongation = np.sqrt(eigenvalues.max() / eigenvalues.min()) if eigenvalues.min() > 0 else 1.0
                else:
                    elongation = 1.0
            except:
                hull_area = 0
                elongation = 1.0
        else:
            hull_area = 0
            elongation = 1.0
        
        # Maximum spread distance
        if len(coords) > 1:
            from scipy.spatial.distance import pdist
            max_distance = pdist(coords).max()
        else:
            max_distance = 0
        
        # Convert bounding box corners to WGS84
        bbox_corners = [
            (min_x, min_y), (max_x, min_y),
            (max_x, max_y), (min_x, max_y)
        ]
        bbox_x = [c[0] for c in bbox_corners]
        bbox_y = [c[1] for c in bbox_corners]
        bbox_lon, bbox_lat = self.spatial_utils.transform_points(
            bbox_x, bbox_y, to_target=False
        )
        
        metrics = {
            'centroid_lat': float(centroid_lat[0]),
            'centroid_lon': float(centroid_lon[0]),
            'bounding_box': [
                float(min(bbox_lon)), float(min(bbox_lat)),
                float(max(bbox_lon)), float(max(bbox_lat))
            ],
            'areasqm': hull_area / 1e6,  # Convert from m² to km²
            'perimeter_km': bbox_width * 2 + bbox_height * 2 / 1000,  # Approximate
            'max_spread_km': max_distance / 1000,
            'shape_elongation': elongation,
            'bbox_width_km': bbox_width / 1000,
            'bbox_height_km': bbox_height / 1000
        }
        
        # Spread direction and rate if configured
        if self.metrics_config['compute_spread_metrics']:
            spread_metrics = self._calculate_spread_metrics(cluster_data, coords)
            metrics.update(spread_metrics)
        
        return metrics
    
    def _calculate_spread_metrics(self, cluster_data, coords):
        """Calculate fire spread direction and rate"""
        
        # Sort by time
        sorted_data = cluster_data.sort_values('acq_datetime')
        
        if len(sorted_data) < 2:
            return {
                'spread_direction_deg': np.nan,
                'mean_spread_rate_kmh': 0,
                'max_spread_rate_kmh': 0
            }
        
        # Calculate progressive centroids
        window_size = min(10, len(sorted_data) // 4)
        if window_size < 2:
            window_size = 2
        
        spread_vectors = []
        spread_rates = []
        
        for i in range(0, len(sorted_data) - window_size, window_size // 2):
            window1 = sorted_data.iloc[i:i+window_size]
            window2 = sorted_data.iloc[i+window_size:i+2*window_size]
            
            if len(window2) == 0:
                continue
            
            # Calculate centroids
            centroid1 = window1[['x_proj', 'y_proj']].mean().values
            centroid2 = window2[['x_proj', 'y_proj']].mean().values
            
            # Vector from centroid1 to centroid2
            vector = centroid2 - centroid1
            distance = np.linalg.norm(vector)
            
            # Time difference
            time1 = window1['acq_datetime'].mean()
            time2 = window2['acq_datetime'].mean()
            time_diff_hours = (time2 - time1).total_seconds() / 3600
            
            if time_diff_hours > 0 and distance > 0:
                # Direction (0-360 degrees, 0=North)
                direction = np.degrees(np.arctan2(vector[0], vector[1])) % 360
                spread_vectors.append(direction)
                
                # Rate
                rate_kmh = (distance / 1000) / time_diff_hours
                spread_rates.append(rate_kmh)
        
        if spread_vectors:
            # Mean direction using circular statistics
            mean_direction = circmean(np.radians(spread_vectors)) * 180 / np.pi % 360
            mean_rate = np.mean(spread_rates)
            max_rate = np.max(spread_rates)
        else:
            mean_direction = np.nan
            mean_rate = 0
            max_rate = 0
        
        return {
            'spread_direction_deg': mean_direction,
            'mean_spread_rate_kmh': mean_rate,
            'max_spread_rate_kmh': max_rate
        }
    
    def _calculate_intensity_metrics(self, cluster_data):
        """Calculate fire intensity metrics"""
        
        # FRP statistics
        frp_values = cluster_data['frp'].values
        total_energy = frp_values.sum()  # MW
        
        # Convert to MWh by assuming each detection represents ~15 minutes
        # (approximate satellite revisit time)
        detection_duration_hours = 0.25
        total_energy_mwh = total_energy * detection_duration_hours
        
        # Brightness statistics
        brightness_values = cluster_data['brightness'].values
        
        # Temporal intensity profile
        if self.metrics_config['compute_intensity_profiles']:
            daily_frp = cluster_data.groupby('acq_date')['frp'].agg(['sum', 'mean', 'max'])
            intensity_trend = self._calculate_trend(
                daily_frp.index, daily_frp['mean'].values
            )
        else:
            intensity_trend = 0
        
        metrics = {
            'total_energy_mwh': total_energy_mwh,
            'peak_frp': frp_values.max(),
            'mean_frp': frp_values.mean(),
            'std_frp': frp_values.std(),
            'percentile_90_frp': np.percentile(frp_values, 90),
            'peak_brightness': brightness_values.max(),
            'mean_brightness': brightness_values.mean(),
            'intensity_trend': intensity_trend,
            'high_intensity_detections': (frp_values > np.percentile(frp_values, 75)).sum()
        }
        
        # Confidence-weighted FRP if configured
        if self.config['clustering']['confidence_weighting']:
            confidence_weights = cluster_data['confidence'].values / 100
            weighted_frp = (frp_values * confidence_weights).sum() / confidence_weights.sum()
            metrics['confidence_weighted_frp'] = weighted_frp
        else:
            metrics['confidence_weighted_frp'] = metrics['mean_frp']
        
        return metrics
    
    def _calculate_quality_metrics(self, cluster_data):
        """Calculate data quality and completeness metrics"""
        
        # Detection count
        detection_count = len(cluster_data)
        
        # Satellite coverage
        satellites = cluster_data['satellite'].unique()
        satellite_coverage = len(satellites) / 2.0  # Assuming 2 satellites (Terra, Aqua)
        
        # Confidence statistics
        confidence_values = cluster_data['confidence'].values
        mean_confidence = confidence_values.mean()
        high_confidence_ratio = (confidence_values >= 80).sum() / len(confidence_values)
        
        # Data completeness estimation
        # Based on expected satellite revisits
        duration_days = (cluster_data['acq_datetime'].max() - 
                        cluster_data['acq_datetime'].min()).days + 1
        expected_detections = duration_days * 4  # ~4 passes per day (2 satellites, day+night)
        completeness_score = min(1.0, detection_count / expected_detections)
        
        # Spatial coherence (how clustered the detections are)
        if len(cluster_data) > 3:
            coords = cluster_data[['x_proj', 'y_proj']].values
            centroid = coords.mean(axis=0)
            distances = np.linalg.norm(coords - centroid, axis=1)
            spatial_coherence = 1 / (1 + distances.std() / distances.mean())
        else:
            spatial_coherence = 1.0
        
        metrics = {
            'detection_count': detection_count,
            'satellite_coverage': satellite_coverage,
            'satellites_list': ','.join(satellites),
            'mean_confidence': mean_confidence,
            'high_confidence_ratio': high_confidence_ratio,
            'min_confidence': confidence_values.min(),
            'data_completeness_score': completeness_score,
            'spatial_coherence_score': spatial_coherence
        }
        
        return metrics
    
    def _calculate_behavior_metrics(self, cluster_data):
        """Calculate fire behavior patterns"""
        
        # Day/night activity
        daynight_counts = cluster_data['daynight'].value_counts()
        day_detections = daynight_counts.get('D', 0)
        night_detections = daynight_counts.get('N', 0)
        total_detections = day_detections + night_detections
        
        if total_detections > 0:
            day_night_ratio = day_detections / total_detections
        else:
            day_night_ratio = 0.5
        
        # Diurnal pattern analysis
        hour_counts = cluster_data['hour_of_day'].value_counts()
        
        # Peak activity hours
        if len(hour_counts) > 0:
            peak_hour = hour_counts.idxmax()
            
            # Diurnal variation coefficient
            hourly_mean = hour_counts.mean()
            hourly_std = hour_counts.std()
            diurnal_variation = hourly_std / hourly_mean if hourly_mean > 0 else 0
        else:
            peak_hour = 12
            diurnal_variation = 0
        
        # Persistence analysis
        daily_data = cluster_data.groupby('acq_date').size()
        if len(daily_data) > 1:
            # Check for consistent daily detections
            date_range = pd.date_range(daily_data.index.min(), daily_data.index.max(), freq='D')
            persistence_score = len(daily_data) / len(date_range)
        else:
            persistence_score = 1.0
        
        # Rekindle analysis
        if len(daily_data) > 2:
            dates = pd.to_datetime(daily_data.index)
            date_gaps = np.diff(dates) / np.timedelta64(1, 'D')
            rekindle_events = ((date_gaps > self.dormancy_threshold) & 
                             (date_gaps <= self.rekindle_max_days)).sum()
        else:
            rekindle_events = 0
        
        metrics = {
            'day_detections': day_detections,
            'night_detections': night_detections,
            'day_night_ratio': day_night_ratio,
            'peak_activity_hour': peak_hour,
            'diurnal_variation_coef': diurnal_variation,
            'persistence_score': persistence_score,
            'rekindle_events': rekindle_events,
            'is_persistent': persistence_score > 0.5,
            'is_day_active': day_night_ratio > 0.6,
            'is_night_active': day_night_ratio < 0.4
        }
        
        return metrics
    
    def _calculate_trend(self, dates, values):
        """Calculate linear trend in values over time"""
        if len(values) < 2:
            return 0
        
        # Convert dates to numeric
        x = np.arange(len(values))
        
        # Linear regression
        coeffs = np.polyfit(x, values, 1)
        trend = coeffs[0]
        
        # Normalize by mean value
        mean_val = values.mean()
        if mean_val > 0:
            trend_normalized = trend / mean_val
        else:
            trend_normalized = 0
        
        return trend_normalized
    
    def _post_process_episodes(self, episodes_df):
        """Post-process episode records"""
        
        if len(episodes_df) == 0:
            return episodes_df
        
        # Add derived fields
        episodes_df['episode_type'] = episodes_df.apply(self._classify_episode, axis=1)
        
        # Sort by start time
        episodes_df = episodes_df.sort_values('start_datetime').reset_index(drop=True)
        
        # Add quality flags
        validation_config = self.config['validation']
        
        episodes_df['is_valid'] = (
            (episodes_df['duration_hours'] >= validation_config['min_episode_duration_hours']) &
            (episodes_df['duration_days'] <= validation_config['max_episode_duration_days']) &
            (episodes_df['areasqm'] >= validation_config['min_episode_area_km2']) &
            (episodes_df['max_spread_rate_kmh'] <= validation_config['max_episode_spread_rate_kmh']) &
            (episodes_df['spatial_coherence_score'] >= validation_config['min_spatial_coherence']) &
            (episodes_df['data_completeness_score'] >= validation_config['min_data_completeness'])
        )
        
        # Round numeric fields
        numeric_cols = episodes_df.select_dtypes(include=[np.number]).columns
        episodes_df[numeric_cols] = episodes_df[numeric_cols].round(4)
        
        return episodes_df
    
    def _classify_episode(self, row):
        """Classify episode type based on characteristics"""
        
        # Simple classification based on duration and intensity
        if row['duration_days'] < 1:
            if row['peak_frp'] > 1000:
                return 'explosive'
            else:
                return 'short'
        elif row['duration_days'] < 7:
            if row['persistence_score'] > 0.8:
                return 'persistent_moderate'
            else:
                return 'intermittent'
        else:
            if row['rekindle_events'] > 0:
                return 'complex_rekindle'
            elif row['persistence_score'] > 0.7:
                return 'long_persistent'
            else:
                return 'long_intermittent'
    
    def aggregate_to_watersheds(self, episodes_df, watershed_gdf):
        """Aggregate episode statistics to watershed level"""
        
        self.logger.info("Aggregating episodes to watersheds")
        
        # Convert episodes to GeoDataFrame
        geometry = [Point(row['centroid_lon'], row['centroid_lat']) 
                   for _, row in episodes_df.iterrows()]
        episodes_gdf = gpd.GeoDataFrame(episodes_df, geometry=geometry, crs='EPSG:4326')
        
        # Reproject to match watersheds
        episodes_gdf = episodes_gdf.to_crs(watershed_gdf.crs)
        
        # Spatial join
        episodes_in_watersheds = gpd.sjoin(episodes_gdf, watershed_gdf, 
                                         how='inner', predicate='within')
        
        # Calculate 95th percentile FRP for high severity threshold
        frp_95th_percentile = episodes_df['peak_frp'].quantile(0.95)
        self.logger.info(f"95th percentile FRP for high severity: {frp_95th_percentile:.1f} MW")
        
        # Create custom aggregation functions for threshold exceedance metrics
        def count_area_threshold(series, watershed_area, threshold):
            """Count episodes exceeding area threshold"""
            return (series / watershed_area > threshold).sum()
        
        def count_high_severity(series, threshold):
            """Count high severity episodes"""
            return (series > threshold).sum()
        
        # Aggregate statistics - first do the basic aggregations
        watershed_stats = episodes_in_watersheds.groupby('huc12').agg({
            'episode_id': 'count',
            'total_energy_mwh': 'sum',
            'areasqm': ['sum', 'max'],  # Added max for HSBF calculation
            'duration_days': ['mean', 'max', 'sum'],
            'peak_frp': 'max',
            'mean_frp': 'mean',
            'detection_count': 'sum',
            'day_night_ratio': 'mean',
            'persistence_score': 'mean'
        })
        
        # Flatten column names
        watershed_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                  for col in watershed_stats.columns]
        
        # Rename columns
        watershed_stats = watershed_stats.rename(columns={
            'episode_id_count': 'episode_count',
            'total_energy_mwh_sum': 'total_energy_mwh',
            'areasqm_sum': 'total_burned_area_km2',
            'areasqm_max': 'max_episode_area_km2',
            'duration_days_mean': 'mean_episode_duration_days',
            'duration_days_max': 'max_episode_duration_days',
            'duration_days_sum': 'total_fire_days',
            'peak_frp_max': 'watershed_peak_frp',
            'mean_frp_mean': 'watershed_mean_frp',
            'detection_count_sum': 'total_detections',
            'day_night_ratio_mean': 'mean_day_activity',
            'persistence_score_mean': 'mean_persistence'
        })
        
        # Add threshold exceedance metrics
        # For each watershed, calculate counts for area thresholds
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        
        for threshold in thresholds:
            threshold_pct = int(threshold * 100)
            col_name = f'n_{threshold_pct}pct_burns'
            
            # Calculate for each watershed
            threshold_counts = {}
            for huc12 in episodes_in_watersheds['huc12'].unique():
                # Get watershed area
                watershed_area = watershed_gdf[watershed_gdf['huc12'] == huc12]['area_km2'].iloc[0]
                # Get episodes in this watershed
                episodes_in_ws = episodes_in_watersheds[episodes_in_watersheds['huc12'] == huc12]
                # Count episodes exceeding threshold
                count = (episodes_in_ws['areasqm'] / watershed_area > threshold).sum()
                threshold_counts[huc12] = count
            
            # Add to watershed stats
            watershed_stats[col_name] = pd.Series(threshold_counts)
        
        # Add high severity count
        high_severity_counts = {}
        for huc12 in episodes_in_watersheds['huc12'].unique():
            episodes_in_ws = episodes_in_watersheds[episodes_in_watersheds['huc12'] == huc12]
            count = (episodes_in_ws['peak_frp'] > frp_95th_percentile).sum()
            high_severity_counts[huc12] = count
        
        watershed_stats['n_high_severity'] = pd.Series(high_severity_counts)
        
        # Add to watershed GeoDataFrame
        watershed_fire_stats = watershed_gdf.merge(watershed_stats, 
                                                  left_on='huc12', 
                                                  right_index=True, 
                                                  how='left')
        
        # Calculate HSBF (Hydrologically Significant Burn Fraction)
        # HSBF = max(episode_area) / watershed_area
        watershed_fire_stats['hsbf'] = (
            watershed_fire_stats['max_episode_area_km2'] / watershed_fire_stats['area_km2']
        )
        
        # Fill NaN values for watersheds with no fires
        fill_values = {
            'episode_count': 0,
            'total_energy_mwh': 0,
            'total_burned_area_km2': 0,
            'max_episode_area_km2': 0,
            'mean_episode_duration_days': 0,
            'max_episode_duration_days': 0,
            'total_fire_days': 0,
            'watershed_peak_frp': 0,
            'watershed_mean_frp': 0,
            'total_detections': 0,
            'mean_day_activity': 0,
            'mean_persistence': 0,
            'hsbf': 0,
            'n_high_severity': 0
        }
        
        # Fill threshold exceedance columns
        for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
            threshold_pct = int(threshold * 100)
            col_name = f'n_{threshold_pct}pct_burns'
            fill_values[col_name] = 0
        
        watershed_fire_stats = watershed_fire_stats.fillna(fill_values)
        
        if 'centroid' in watershed_fire_stats.columns:
            watershed_fire_stats['centroid_wkt'] = watershed_fire_stats['centroid'].to_wkt()
            watershed_fire_stats = watershed_fire_stats.drop(columns=['centroid'])
        
        # Ensure only one geometry column remains
        geom_cols = [col for col in watershed_fire_stats.columns 
                    if isinstance(watershed_fire_stats[col].dtype, object) 
                    and hasattr(watershed_fire_stats[col].iloc[0] if len(watershed_fire_stats) > 0 else None, 'geom_type')]
        
        if len(geom_cols) > 1:
            # Keep the main geometry column, convert others to WKT
            main_geom_col = 'geometry'
            for col in geom_cols:
                if col != main_geom_col:
                    watershed_fire_stats[f'{col}_wkt'] = watershed_fire_stats[col].to_wkt()
                    watershed_fire_stats = watershed_fire_stats.drop(columns=[col])
        
        self.logger.info(f"Aggregated to {len(watershed_stats)} watersheds with fires")
        
        return watershed_fire_stats