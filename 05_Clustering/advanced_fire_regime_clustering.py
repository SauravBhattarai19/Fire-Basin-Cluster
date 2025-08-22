#!/usr/bin/env python3
"""
Advanced Fire Regime Clustering Framework
Addresses bias issues and leverages high-performance computing
Author: Fire Analysis Research Team
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering, 
                           SpectralClustering, OPTICS, Birch)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                           davies_bouldin_score, adjusted_rand_score)
from sklearn.model_selection import GridSearchCV
import hdbscan
from scipy.stats import zscore, chi2
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set optimal number of threads for BLAS operations
import os
n_cores = min(32, mp.cpu_count())  # Cap at 32 for BLAS operations
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)

class AdvancedFireRegimeClustering:
    """
    Scientifically robust fire regime clustering that addresses bias
    and leverages high-performance computing
    """
    
    def __init__(self, watersheds_gdf, n_jobs=-1):
        """
        Initialize advanced clustering framework
        
        Args:
            watersheds_gdf: GeoDataFrame with fire metrics
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.watersheds = watersheds_gdf.copy()
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        print(f"Initialized with {self.n_jobs} parallel workers")
        
        # Store results
        self.clustering_results = {}
        self.evaluation_metrics = {}
        
    def stratified_fire_analysis(self):
        """
        Stratify watersheds by fire activity level to address bias
        This is CRITICAL for unbiased clustering!
        """
        print("\n" + "="*60)
        print("STRATIFIED FIRE ACTIVITY ANALYSIS")
        print("="*60)
        
        # Calculate fire activity percentiles
        fire_activity = self.watersheds['episode_count'].fillna(0)
        
        # Define strata based on fire activity
        self.watersheds['fire_strata'] = pd.cut(
            fire_activity,
            bins=[-0.1, 0, 5, 20, 50, np.inf],
            labels=['No Fire', 'Rare Fire', 'Occasional Fire', 
                   'Frequent Fire', 'Very Frequent Fire']
        )
        
        # Print stratification results
        print("\nFire Activity Stratification:")
        print("-" * 40)
        strata_counts = self.watersheds['fire_strata'].value_counts()
        for strata, count in strata_counts.items():
            pct = count / len(self.watersheds) * 100
            print(f"{strata}: {count:,} watersheds ({pct:.1f}%)")
        
        # Identify fire-affected watersheds needing clustering
        self.fire_watersheds = self.watersheds[
            self.watersheds['episode_count'] > 0
        ].copy()
        
        print(f"\nFire-affected watersheds: {len(self.fire_watersheds):,}")
        
        # Further stratify fire-affected watersheds
        if len(self.fire_watersheds) > 100:
            # Use quantiles for balanced strata
            quantiles = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
            self.fire_watersheds['fire_intensity_strata'] = pd.qcut(
                self.fire_watersheds['hsbf'].fillna(0),
                q=quantiles,
                labels=['Very Low', 'Low', 'Moderate', 'High', 'Extreme'],
                duplicates='drop'
            )
            
            print("\nFire Intensity Stratification (HSBF-based):")
            intensity_counts = self.fire_watersheds['fire_intensity_strata'].value_counts()
            for strata, count in intensity_counts.items():
                print(f"  {strata}: {count:,} watersheds")
        
        return self.fire_watersheds
    
    def prepare_multiscale_features(self, include_ecological=False):
        """
        Prepare features at multiple scales with proper transformations
        """
        print("\n" + "="*60)
        print("MULTISCALE FEATURE ENGINEERING")
        print("="*60)
        
        # Core fire regime features
        core_features = [
            'episode_count',
            'total_burned_area_km2',
            'max_single_fire_area_km2',
            'mean_fire_area_km2',
            'hsbf',
            'total_frp_mw',
            'max_frp_mw',
            'mean_frp_mw',
            'mean_duration_days',
            'fire_return_interval_years',
            'seasonality_index',
            'day_detection_ratio',
            'fire_size_cv',
            'area_burned_fraction',
            'high_intensity_count'
        ]
        
        # Get available features
        available_features = [f for f in core_features 
                            if f in self.fire_watersheds.columns]
        
        print(f"Using {len(available_features)} core features")
        
        # Extract feature matrix
        X = self.fire_watersheds[available_features].copy()
        
        # Handle missing values strategically
        # Fire return interval: use large value for single fires
        if 'fire_return_interval_years' in X.columns:
            X['fire_return_interval_years'].fillna(50, inplace=True)
        
        # Fill other NaNs with 0
        X.fillna(0, inplace=True)
        
        # Create derived features
        print("Creating derived features...")
        
        # Fire complexity index
        if 'fire_size_cv' in X.columns and 'seasonality_index' in X.columns:
            X['fire_complexity'] = X['fire_size_cv'] * X['seasonality_index']
        
        # Fire pressure index (frequency * intensity)
        if 'episode_count' in X.columns and 'mean_frp_mw' in X.columns:
            X['fire_pressure'] = np.log1p(X['episode_count']) * np.log1p(X['mean_frp_mw'])
        
        # Burn severity proxy
        if 'hsbf' in X.columns and 'max_frp_mw' in X.columns:
            X['burn_severity_proxy'] = X['hsbf'] * np.log1p(X['max_frp_mw'])
        
        # Add spatial features if available
        if 'centroid_lat' in self.fire_watersheds.columns:
            X['latitude'] = self.fire_watersheds['centroid_lat']
            X['longitude'] = self.fire_watersheds['centroid_lon']
            print("Added spatial coordinates")
        
        # Add ecological features if requested and available
        if include_ecological:
            # Prefer variables listed in 06_Ecological_Context/variable_descriptions.json if present
            try:
                from pathlib import Path
                import json
                var_desc_path = Path('06_Ecological_Context') / 'variable_descriptions.json'
                ecological_vars = []
                if var_desc_path.exists():
                    with open(var_desc_path, 'r') as f:
                        ecological_vars = list(json.load(f).keys())
                # Fallback: all numeric columns not already included and not identifiers
                non_feature_cols = set([
                    'cluster', 'final_cluster', 'cluster_name', 'fire_strata',
                    'fire_intensity_strata', 'geometry', 'latitude', 'longitude',
                    'centroid_lat', 'centroid_lon'
                ] + available_features)
                candidate_cols = []
                if ecological_vars:
                    candidate_cols = [c for c in ecological_vars if c in self.fire_watersheds.columns]
                else:
                    candidate_cols = [
                        c for c in self.fire_watersheds.columns
                        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(self.fire_watersheds[c])
                    ]
                added = 0
                for feat in candidate_cols:
                    if feat not in X.columns and feat in self.fire_watersheds.columns:
                        if pd.api.types.is_numeric_dtype(self.fire_watersheds[feat]):
                            X[feat] = self.fire_watersheds[feat]
                            added += 1
                if added:
                    print(f"Added {added} ecological feature(s)")
            except Exception as _:
                pass
        
        # Final cleaning: handle infinities/NaNs introduced by added features
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        # Drop columns that are entirely NaN or constant (no variance)
        drop_cols = []
        for col in list(numeric_cols):
            if X[col].isna().all():
                drop_cols.append(col)
            elif X[col].nunique(dropna=True) <= 1:
                drop_cols.append(col)
        if drop_cols:
            X.drop(columns=drop_cols, inplace=True)
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if drop_cols:
                print(f"Dropped {len(drop_cols)} non-informative ecological/core feature(s)")
        # Impute remaining NaNs with column medians (numeric only)
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        self.feature_matrix = X
        print(f"\nFinal feature matrix shape: {X.shape}")
        
        # Apply multiple transformation strategies
        self.transformed_features = self._apply_transformations(X)
        
        return self.transformed_features
    
    def _apply_transformations(self, X):
        """
        Apply multiple transformation strategies for robustness
        """
        print("\nApplying feature transformations...")
        
        transformations = {}
        
        # 1. Standard scaling (z-score normalization)
        scaler_standard = StandardScaler()
        transformations['standard'] = scaler_standard.fit_transform(X)
        
        # 2. Robust scaling (median and IQR)
        scaler_robust = RobustScaler()
        transformations['robust'] = scaler_robust.fit_transform(X)
        
        # 3. Quantile transformation (uniform distribution)
        scaler_quantile = QuantileTransformer(output_distribution='uniform', 
                                              n_quantiles=min(1000, len(X)))
        transformations['quantile'] = scaler_quantile.fit_transform(X)
        
        # 4. Log transformation for skewed features
        X_log = X.copy()
        skewed_cols = ['episode_count', 'total_burned_area_km2', 
                      'total_frp_mw', 'max_frp_mw']
        for col in skewed_cols:
            if col in X_log.columns:
                X_log[col] = np.log1p(X_log[col])
        transformations['log_robust'] = RobustScaler().fit_transform(X_log)
        
        print(f"Created {len(transformations)} transformation variants")
        
        # Store transformers for inverse transform
        self.transformers = {
            'standard': scaler_standard,
            'robust': scaler_robust,
            'quantile': scaler_quantile
        }
        
        return transformations
    
    def parallel_clustering_suite(self, X_dict, n_clusters_range=range(3, 11)):
        """
        Run multiple clustering algorithms in parallel
        Leverages all available cores efficiently
        """
        print("\n" + "="*60)
        print("PARALLEL MULTI-ALGORITHM CLUSTERING")
        print("="*60)
        
        algorithms = self._get_clustering_algorithms(n_clusters_range)
        
        # Prepare all combinations
        tasks = []
        for transform_name, X_transformed in X_dict.items():
            for algo_name, algo_params in algorithms.items():
                tasks.append((transform_name, algo_name, algo_params, X_transformed))
        
        print(f"Running {len(tasks)} clustering experiments in parallel...")
        # Dynamically reduce workers for very large problems to avoid OOM
        n_samples = len(self.fire_watersheds)
        dynamic_n_jobs = self.n_jobs
        if n_samples > 30000:
            dynamic_n_jobs = min(self.n_jobs, 12)
        print(f"Using {dynamic_n_jobs} workers")
        
        # Run parallel clustering
        results = Parallel(n_jobs=dynamic_n_jobs, backend='loky', verbose=1)(
            delayed(self._run_single_clustering)(
                transform_name, algo_name, algo_params, X_transformed
            )
            for transform_name, algo_name, algo_params, X_transformed in tasks
        )
        
        # Organize results
        for result in results:
            if result is not None:
                key = f"{result['transform']}_{result['algorithm']}_{result['n_clusters']}"
                self.clustering_results[key] = result
        
        print(f"\nCompleted {len(self.clustering_results)} clustering experiments")
        
        return self.clustering_results
    
    def _get_clustering_algorithms(self, n_clusters_range):
        """
        Define comprehensive set of clustering algorithms
        """
        algorithms = {}
        
        for n_clusters in n_clusters_range:
            # K-Means variants
            algorithms[f'kmeans_{n_clusters}'] = {
                'algorithm': 'kmeans',
                'n_clusters': n_clusters,
                'params': {'n_init': 20, 'max_iter': 500, 'random_state': 42}
            }
            
            # Gaussian Mixture Model
            # GMM with reduced iterations for memory safety
            algorithms[f'gmm_{n_clusters}'] = {
                'algorithm': 'gmm',
                'n_clusters': n_clusters,
                'params': {'covariance_type': 'diag', 'n_init': 1,
                          'random_state': 42, 'max_iter': 100}
            }
            
            # Bayesian GMM (automatically determines optimal components)
            algorithms[f'bgmm_{n_clusters}'] = {
                'algorithm': 'bgmm',
                'n_clusters': n_clusters,
                'params': {'n_components': n_clusters * 2,  # Upper bound
                          'covariance_type': 'full', 'random_state': 42}
            }
            
            # Spectral Clustering (good for non-convex clusters)
            # Spectral can be very heavy; skip for large datasets
            if len(self.fire_watersheds) < 5000:
                algorithms[f'spectral_{n_clusters}'] = {
                    'algorithm': 'spectral',
                    'n_clusters': n_clusters,
                    'params': {'affinity': 'nearest_neighbors', 
                             'n_neighbors': 10, 'random_state': 42}
                }
            
            # Agglomerative (Hierarchical)
            algorithms[f'hierarchical_{n_clusters}'] = {
                'algorithm': 'hierarchical',
                'n_clusters': n_clusters,
                'params': {'linkage': 'ward'}
            }
        
        # Density-based algorithms (determine n_clusters automatically)
        
        # HDBSCAN (Hierarchical DBSCAN) - BEST for varying densities
        algorithms['hdbscan_auto'] = {
            'algorithm': 'hdbscan',
            'n_clusters': 'auto',
            'params': {'min_cluster_size': max(5, len(self.fire_watersheds) // 100),
                      'min_samples': 5, 'cluster_selection_epsilon': 0.0,
                      'cluster_selection_method': 'eom'}
        }
        
        # OPTICS (Ordering Points To Identify Clustering Structure)
        algorithms['optics_auto'] = {
            'algorithm': 'optics',
            'n_clusters': 'auto',
            'params': {'min_samples': 5, 'xi': 0.05, 'min_cluster_size': 0.05}
        }
        
        # BIRCH (good for large datasets)
        for n_clusters in n_clusters_range:
            algorithms[f'birch_{n_clusters}'] = {
                'algorithm': 'birch',
                'n_clusters': n_clusters,
                'params': {'threshold': 0.5, 'branching_factor': 50}
            }
        
        return algorithms
    
    def _run_single_clustering(self, transform_name, algo_name, algo_params, X):
        """
        Run a single clustering experiment
        """
        try:
            algorithm = algo_params['algorithm']
            n_clusters = algo_params['n_clusters']
            params = algo_params['params']
            
            # Initialize clusterer
            if algorithm == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, **params)
                labels = clusterer.fit_predict(X)
                
            elif algorithm == 'gmm':
                clusterer = GaussianMixture(n_components=n_clusters, **params)
                labels = clusterer.fit_predict(X)
                
            elif algorithm == 'bgmm':
                clusterer = BayesianGaussianMixture(**params)
                labels = clusterer.fit_predict(X)
                n_clusters = len(np.unique(labels))  # Actual clusters found
                
            elif algorithm == 'spectral':
                clusterer = SpectralClustering(n_clusters=n_clusters, **params)
                labels = clusterer.fit_predict(X)
                
            elif algorithm == 'hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, **params)
                labels = clusterer.fit_predict(X)
                
            elif algorithm == 'hdbscan':
                clusterer = hdbscan.HDBSCAN(**params)
                labels = clusterer.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
            elif algorithm == 'optics':
                clusterer = OPTICS(**params)
                labels = clusterer.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
            elif algorithm == 'birch':
                clusterer = Birch(n_clusters=n_clusters, **params)
                labels = clusterer.fit_predict(X)
            
            else:
                return None
            
            # Evaluate clustering
            metrics = self._evaluate_clustering(X, labels)
            
            return {
                'transform': transform_name,
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'labels': labels,
                'metrics': metrics,
                'clusterer': clusterer
            }
            
        except Exception as e:
            print(f"Error in {algo_name} with {transform_name}: {str(e)}")
            return None
    
    def _evaluate_clustering(self, X, labels):
        """
        Comprehensive clustering evaluation
        """
        metrics = {}
        
        # Filter out noise points for metrics
        mask = labels >= 0
        if mask.sum() < 2:
            return metrics
        
        X_clean = X[mask]
        labels_clean = labels[mask]
        
        # Only calculate if we have meaningful clusters
        n_clusters = len(np.unique(labels_clean))
        if n_clusters < 2 or n_clusters >= len(labels_clean) - 1:
            return metrics
        
        try:
            # Internal validation metrics (subsample for memory safety)
            max_samples_for_silhouette = 20000
            if len(X_clean) > max_samples_for_silhouette:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_clean), max_samples_for_silhouette, replace=False)
                X_eval = X_clean[idx]
                labels_eval = labels_clean[idx]
            else:
                X_eval = X_clean
                labels_eval = labels_clean

            metrics['silhouette'] = silhouette_score(X_eval, labels_eval)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_clean, labels_clean)
            metrics['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
            
            # Cluster statistics
            metrics['n_clusters'] = n_clusters
            metrics['n_noise'] = (labels == -1).sum()
            metrics['noise_ratio'] = metrics['n_noise'] / len(labels)
            
            # Cluster size statistics
            cluster_sizes = pd.Series(labels_clean).value_counts()
            metrics['min_cluster_size'] = cluster_sizes.min()
            metrics['max_cluster_size'] = cluster_sizes.max()
            metrics['cluster_size_std'] = cluster_sizes.std()
            metrics['cluster_balance'] = cluster_sizes.min() / cluster_sizes.max()
            
        except Exception as e:
            print(f"Evaluation error: {e}")
        
        return metrics
    
    def select_best_clustering(self, criteria='balanced'):
        """
        Select best clustering based on multiple criteria
        """
        print("\n" + "="*60)
        print("SELECTING OPTIMAL CLUSTERING")
        print("="*60)
        
        # Convert results to DataFrame for analysis
        results_df = []
        for key, result in self.clustering_results.items():
            if result['metrics']:
                row = {
                    'key': key,
                    'transform': result['transform'],
                    'algorithm': result['algorithm'],
                    'n_clusters': result['n_clusters']
                }
                row.update(result['metrics'])
                results_df.append(row)
        
        results_df = pd.DataFrame(results_df)
        
        if len(results_df) == 0:
            print("No valid clustering results found!")
            return None
        
        # Filter out poor clusterings
        results_df = results_df[
            (results_df['n_clusters'] >= 3) & 
            (results_df['n_clusters'] <= 15) &
            (results_df['noise_ratio'] < 0.3)
        ]
        
        if len(results_df) == 0:
            print("No clusterings meet quality criteria!")
            return None
        
        print(f"\nEvaluating {len(results_df)} valid clusterings...")
        
        # Normalize metrics (higher is better)
        results_df['silhouette_norm'] = (results_df['silhouette'] - results_df['silhouette'].min()) / (results_df['silhouette'].max() - results_df['silhouette'].min() + 1e-10)
        results_df['ch_norm'] = (results_df['calinski_harabasz'] - results_df['calinski_harabasz'].min()) / (results_df['calinski_harabasz'].max() - results_df['calinski_harabasz'].min() + 1e-10)
        results_df['db_norm'] = 1 - (results_df['davies_bouldin'] - results_df['davies_bouldin'].min()) / (results_df['davies_bouldin'].max() - results_df['davies_bouldin'].min() + 1e-10)
        results_df['balance_norm'] = (results_df['cluster_balance'] - results_df['cluster_balance'].min()) / (results_df['cluster_balance'].max() - results_df['cluster_balance'].min() + 1e-10)
        
        # Calculate composite scores based on criteria
        if criteria == 'balanced':
            # Balance between all metrics
            results_df['score'] = (
                0.3 * results_df['silhouette_norm'] +
                0.2 * results_df['ch_norm'] +
                0.2 * results_df['db_norm'] +
                0.3 * results_df['balance_norm']
            )
        elif criteria == 'separation':
            # Prioritize cluster separation
            results_df['score'] = (
                0.5 * results_df['silhouette_norm'] +
                0.3 * results_df['ch_norm'] +
                0.2 * results_df['db_norm']
            )
        elif criteria == 'stability':
            # Prioritize balanced, stable clusters
            results_df['score'] = (
                0.2 * results_df['silhouette_norm'] +
                0.1 * results_df['ch_norm'] +
                0.2 * results_df['db_norm'] +
                0.5 * results_df['balance_norm']
            )
        
        # Select best
        best_idx = results_df['score'].idxmax()
        best_result = results_df.loc[best_idx]
        
        print("\nBest clustering:")
        print("-" * 40)
        print(f"Algorithm: {best_result['algorithm']}")
        print(f"Transform: {best_result['transform']}")
        print(f"N_clusters: {best_result['n_clusters']}")
        print(f"Silhouette: {best_result['silhouette']:.3f}")
        print(f"Calinski-Harabasz: {best_result['calinski_harabasz']:.1f}")
        print(f"Davies-Bouldin: {best_result['davies_bouldin']:.3f}")
        print(f"Cluster balance: {best_result['cluster_balance']:.3f}")
        
        # Get full result
        self.best_clustering = self.clustering_results[best_result['key']]
        
        # Print top 5 alternatives
        print("\nTop 5 alternatives:")
        top5 = results_df.nlargest(5, 'score')[['algorithm', 'transform', 'n_clusters', 'silhouette', 'score']]
        print(top5.to_string(index=False))
        
        return self.best_clustering
    
    def ensemble_clustering(self, top_n=5):
        """
        Create ensemble clustering from top N results
        More robust than single algorithm
        """
        print("\n" + "="*60)
        print("ENSEMBLE CLUSTERING")
        print("="*60)
        
        # Get top N clusterings
        results_df = []
        for key, result in self.clustering_results.items():
            if result['metrics'] and 'silhouette' in result['metrics']:
                results_df.append({
                    'key': key,
                    'silhouette': result['metrics']['silhouette']
                })
        
        results_df = pd.DataFrame(results_df)
        top_keys = results_df.nlargest(top_n, 'silhouette')['key'].values
        
        # Create co-association matrix
        n_samples = len(self.fire_watersheds)
        co_association = np.zeros((n_samples, n_samples))
        
        print(f"Building ensemble from top {top_n} clusterings...")
        for key in top_keys:
            labels = self.clustering_results[key]['labels']
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if labels[i] == labels[j] and labels[i] >= 0:
                        co_association[i, j] += 1
                        co_association[j, i] += 1
        
        # Normalize
        co_association /= top_n
        # Self-agreement should be perfect
        np.fill_diagonal(co_association, 1.0)
        
        # Final clustering on co-association matrix
        print("Performing consensus clustering...")
        
        # Convert co-association to distance matrix
        distance_matrix = 1 - co_association
        # Ensure a valid distance matrix: symmetric with zero diagonal
        # Numerical safety: clamp to [0, 1], symmetrize, and zero the diagonal
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Use hierarchical clustering on consensus matrix
        from scipy.cluster.hierarchy import fcluster, linkage
        Z = linkage(squareform(distance_matrix), method='average')
        
        # Determine optimal cut
        max_d = np.percentile(Z[:, 2], 90)  # Cut at 90th percentile of distances
        ensemble_labels = fcluster(Z, max_d, criterion='distance') - 1
        
        print(f"Ensemble produced {len(np.unique(ensemble_labels))} clusters")
        
        # Evaluate ensemble
        ensemble_metrics = self._evaluate_clustering(
            self.transformed_features['robust'],
            ensemble_labels
        )
        
        print(f"Ensemble silhouette: {ensemble_metrics.get('silhouette', 0):.3f}")
        
        self.ensemble_labels = ensemble_labels
        self.ensemble_metrics = ensemble_metrics
        
        return ensemble_labels
    
    def analyze_and_characterize_clusters(self):
        """
        Comprehensive cluster analysis and characterization
        """
        print("\n" + "="*60)
        print("CLUSTER CHARACTERIZATION")
        print("="*60)
        
        # Use best clustering or ensemble
        if hasattr(self, 'ensemble_labels'):
            labels = self.ensemble_labels
            method = "Ensemble"
        elif hasattr(self, 'best_clustering'):
            labels = self.best_clustering['labels']
            method = self.best_clustering['algorithm']
        else:
            print("No clustering results available!")
            return None
        
        print(f"Analyzing {method} clustering results...")
        
        # Add labels to watersheds
        self.fire_watersheds['cluster'] = labels
        
        # Analyze each cluster
        cluster_profiles = {}
        
        for cluster_id in np.unique(labels[labels >= 0]):
            mask = labels == cluster_id
            cluster_data = self.fire_watersheds[mask]
            
            profile = {
                'size': mask.sum(),
                'episode_count': {
                    'mean': cluster_data['episode_count'].mean(),
                    'std': cluster_data['episode_count'].std(),
                    'median': cluster_data['episode_count'].median()
                },
                'hsbf': {
                    'mean': cluster_data['hsbf'].mean(),
                    'std': cluster_data['hsbf'].std(),
                    'max': cluster_data['hsbf'].max()
                },
                'fire_size': {
                    'mean_area': cluster_data['mean_fire_area_km2'].mean(),
                    'max_area': cluster_data['max_single_fire_area_km2'].mean()
                },
                'intensity': {
                    'mean_frp': cluster_data['mean_frp_mw'].mean(),
                    'max_frp': cluster_data['max_frp_mw'].mean()
                },
                'temporal': {
                    'duration': cluster_data['mean_duration_days'].mean(),
                    'seasonality': cluster_data['seasonality_index'].mean(),
                    'return_interval': cluster_data['fire_return_interval_years'].mean()
                }
            }
            
            # Assign descriptive name
            name = self._generate_cluster_name(profile)
            profile['name'] = name
            
            cluster_profiles[cluster_id] = profile
            
            print(f"\nCluster {cluster_id}: {name}")
            print(f"  Size: {profile['size']} watersheds")
            print(f"  Episodes: {profile['episode_count']['mean']:.1f} ± {profile['episode_count']['std']:.1f}")
            print(f"  HSBF: {profile['hsbf']['mean']:.3f} (max: {profile['hsbf']['max']:.3f})")
            print(f"  Mean fire area: {profile['fire_size']['mean_area']:.1f} km²")
            print(f"  Mean FRP: {profile['intensity']['mean_frp']:.1f} MW")
        
        self.cluster_profiles = cluster_profiles
        
        return cluster_profiles
    
    def _generate_cluster_name(self, profile):
        """
        Generate interpretable cluster name based on characteristics
        """
        # Frequency classification
        episodes = profile['episode_count']['mean']
        if episodes < 3:
            freq = "Rare"
        elif episodes < 10:
            freq = "Occasional"
        elif episodes < 25:
            freq = "Frequent"
        else:
            freq = "Very-Frequent"
        
        # Impact classification (based on HSBF)
        hsbf = profile['hsbf']['mean']
        if hsbf < 0.05:
            impact = "Minimal-Impact"
        elif hsbf < 0.15:
            impact = "Low-Impact"
        elif hsbf < 0.30:
            impact = "Moderate-Impact"
        elif hsbf < 0.50:
            impact = "High-Impact"
        else:
            impact = "Extreme-Impact"
        
        # Intensity classification
        frp = profile['intensity']['mean_frp']
        if frp < 30:
            intensity = "Low-Intensity"
        elif frp < 70:
            intensity = "Moderate-Intensity"
        else:
            intensity = "High-Intensity"
        
        # Size classification
        area = profile['fire_size']['mean_area']
        if area < 5:
            size = "Small"
        elif area < 20:
            size = "Medium"
        else:
            size = "Large"
        
        return f"{freq}_{impact}_{size}_{intensity}"
    
    def validate_clustering_stability(self, n_iterations=10):
        """
        Validate clustering stability through bootstrap resampling
        """
        print("\n" + "="*60)
        print("CLUSTERING STABILITY VALIDATION")
        print("="*60)
        
        if not hasattr(self, 'best_clustering'):
            print("No best clustering selected!")
            return None
        
        best_config = self.best_clustering
        X = self.transformed_features[best_config['transform']]
        algorithm = best_config['algorithm']
        n_clusters = best_config['n_clusters']
        
        print(f"Validating {algorithm} with {n_iterations} bootstrap samples...")
        
        # Store results
        stability_results = []
        
        for i in range(n_iterations):
            # Bootstrap sample
            n_samples = len(X)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            
            # Recluster
            if algorithm == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=i)
            elif algorithm == 'gmm':
                clusterer = GaussianMixture(n_components=n_clusters, random_state=i)
            elif algorithm == 'hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                continue
            
            labels_boot = clusterer.fit_predict(X_boot)
            
            # Evaluate
            try:
                sil = silhouette_score(X_boot, labels_boot)
                stability_results.append(sil)
            except:
                pass
        
        if stability_results:
            mean_sil = np.mean(stability_results)
            std_sil = np.std(stability_results)
            cv_sil = std_sil / mean_sil if mean_sil > 0 else np.inf
            
            print(f"\nStability Results:")
            print(f"  Mean Silhouette: {mean_sil:.3f}")
            print(f"  Std Silhouette: {std_sil:.3f}")
            print(f"  CV: {cv_sil:.3f}")
            
            if cv_sil < 0.1:
                print("  → Highly stable clustering")
            elif cv_sil < 0.2:
                print("  → Moderately stable clustering")
            else:
                print("  → Unstable clustering - consider ensemble approach")
        
        return stability_results
    
    def export_results(self, output_dir='05_Clustering'):
        """
        Export comprehensive clustering results
        """
        print("\n" + "="*60)
        print("EXPORTING RESULTS")
        print("="*60)
        
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export clustered watersheds
        if hasattr(self, 'ensemble_labels'):
            self.fire_watersheds['final_cluster'] = self.ensemble_labels
            method = 'ensemble'
        elif hasattr(self, 'best_clustering'):
            self.fire_watersheds['final_cluster'] = self.best_clustering['labels']
            method = self.best_clustering['algorithm']
        else:
            print("No clustering results to export!")
            return
        
        # Add cluster names
        if hasattr(self, 'cluster_profiles'):
            cluster_names = {k: v['name'] for k, v in self.cluster_profiles.items()}
            self.fire_watersheds['cluster_name'] = self.fire_watersheds['final_cluster'].map(cluster_names)
        
        # Export GeoPackage (ensure GeoDataFrame)
        from shapely.geometry import shape
        output_file = output_dir / f'fire_regime_clusters_{method}.gpkg'
        gdf_to_export = self.fire_watersheds
        if not isinstance(gdf_to_export, gpd.GeoDataFrame):
            if 'geometry' in gdf_to_export.columns:
                try:
                    # If geometry is WKB/WKT/GeoJSON-like, let GeoPandas parse
                    gdf_to_export = gpd.GeoDataFrame(gdf_to_export, geometry='geometry', crs=self.watersheds.crs if hasattr(self.watersheds, 'crs') else None)
                except Exception:
                    pass
        try:
            if isinstance(gdf_to_export, gpd.GeoDataFrame) and 'geometry' in gdf_to_export.columns:
                gdf_to_export.to_file(output_file, driver='GPKG')
                print(f"Exported clustered watersheds to {output_file}")
            else:
                # Fallback to CSV if geometry is unavailable
                csv_fallback = output_dir / f'fire_regime_clusters_{method}.csv'
                self.fire_watersheds.to_csv(csv_fallback, index=False)
                print(f"Geometry not available; exported CSV to {csv_fallback}")
        except Exception as e:
            # Final fallback to CSV on any export error
            csv_fallback = output_dir / f'fire_regime_clusters_{method}.csv'
            self.fire_watersheds.to_csv(csv_fallback, index=False)
            print(f"GeoPackage export failed ({e}); exported CSV to {csv_fallback}")
        
        # Export cluster statistics
        if hasattr(self, 'cluster_profiles'):
            profiles_df = pd.DataFrame(self.cluster_profiles).T
            profiles_file = output_dir / 'cluster_profiles.csv'
            profiles_df.to_csv(profiles_file)
            print(f"Exported cluster profiles to {profiles_file}")
        
        # Export evaluation metrics
        metrics_data = []
        for key, result in self.clustering_results.items():
            if result['metrics']:
                row = {
                    'method': f"{result['transform']}_{result['algorithm']}",
                    'n_clusters': result['n_clusters']
                }
                row.update(result['metrics'])
                metrics_data.append(row)
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_file = output_dir / 'clustering_evaluation_metrics.csv'
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Exported evaluation metrics to {metrics_file}")
        
        print("\nExport complete!")
        
        return output_dir