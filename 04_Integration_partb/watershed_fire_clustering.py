#!/usr/bin/env python3
"""
Watershed Fire Regime Clustering
Clusters watersheds based on fire characteristics
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class WatershedFireRegimeClustering:
    """
    Scientifically sound watershed clustering based on fire regime characteristics
    """
    
    def __init__(self, watersheds_gdf):
        """
        Initialize with watershed data containing fire metrics
        
        Args:
            watersheds_gdf: GeoDataFrame with fire regime metrics
        """
        self.watersheds = watersheds_gdf.copy()
        self.features = None
        self.scaler = None
        self.clusters = None
        self.cluster_metrics = {}
        
    def prepare_features(self, min_fire_count=1):
        """
        Prepare and select features for clustering
        
        Args:
            min_fire_count: Minimum number of fires to include watershed
        """
        print("\n" + "="*60)
        print("PREPARING FEATURES FOR CLUSTERING")
        print("="*60)
        
        # Filter to watersheds with sufficient fire activity
        self.watersheds_with_fire = self.watersheds[
            self.watersheds['episode_count'] >= min_fire_count
        ].copy()
        
        print(f"Watersheds with ≥{min_fire_count} fires: {len(self.watersheds_with_fire)}")
        
        if len(self.watersheds_with_fire) < 30:
            print("Warning: Too few watersheds for robust clustering")
            return None
        
        # Select fire regime features for clustering
        feature_columns = [
            # Fire frequency
            'episode_count',
            'fire_return_interval_years',
            
            # Fire size/extent
            'mean_fire_area_km2',
            'max_single_fire_area_km2',
            'area_burned_fraction',
            'hsbf',
            
            # Fire intensity
            'mean_frp_mw',
            'max_frp_mw',
            
            # Temporal patterns
            'mean_duration_days',
            'seasonality_index',
            'day_detection_ratio',
            
            # Variability
            'fire_size_cv'
        ]
        
        # Check which features are available
        available_features = [col for col in feature_columns 
                             if col in self.watersheds_with_fire.columns]
        
        print(f"\nUsing {len(available_features)} features:")
        for feat in available_features:
            print(f"  • {feat}")
        
        # Extract features
        self.feature_matrix = self.watersheds_with_fire[available_features].copy()
        
        # Handle missing values
        # For fire return interval, use a large value for single fires
        if 'fire_return_interval_years' in self.feature_matrix.columns:
            self.feature_matrix['fire_return_interval_years'].fillna(20, inplace=True)
        
        # Fill other NaNs with 0
        self.feature_matrix.fillna(0, inplace=True)
        
        # Handle infinite values
        self.feature_matrix.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Log transform skewed features
        skewed_features = ['episode_count', 'mean_fire_area_km2', 
                          'max_single_fire_area_km2', 'mean_frp_mw', 'max_frp_mw']
        
        for feat in skewed_features:
            if feat in self.feature_matrix.columns:
                self.feature_matrix[f'{feat}_log'] = np.log1p(self.feature_matrix[feat])
                self.feature_matrix.drop(feat, axis=1, inplace=True)
        
        # Standardize features
        self.scaler = RobustScaler()  # Robust to outliers
        self.features_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        print(f"\nFeature matrix shape: {self.features_scaled.shape}")
        
        # Perform PCA for visualization
        self._perform_pca()
        
        return self.features_scaled
    
    def _perform_pca(self):
        """Perform PCA for dimensionality reduction and visualization"""
        pca = PCA(n_components=min(10, self.features_scaled.shape[1]))
        self.pca_features = pca.fit_transform(self.features_scaled)
        
        # Print explained variance
        print("\nPCA Explained Variance:")
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumsum)):
            print(f"  PC{i+1}: {var:.3f} (cumulative: {cum:.3f})")
        
        self.pca = pca
    
    def determine_optimal_clusters(self, max_k=10):
        """
        Determine optimal number of clusters using multiple methods
        """
        print("\n" + "="*60)
        print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
        print("="*60)
        
        if self.features_scaled is None:
            print("Error: Features not prepared. Run prepare_features() first.")
            return None
        
        # Test range of k values
        k_range = range(2, min(max_k + 1, len(self.watersheds_with_fire) // 10))
        
        metrics = {
            'k': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'inertia': []
        }
        
        print("\nTesting k values:")
        for k in k_range:
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features_scaled)
            
            # Calculate metrics
            metrics['k'].append(k)
            # Silhouette: full for small n, sampled for large n
            try:
                n_samples = self.features_scaled.shape[0]
                if n_samples <= 20000:
                    sil = silhouette_score(self.features_scaled, labels)
                else:
                    sil = silhouette_score(
                        self.features_scaled, labels,
                        sample_size=min(10000, n_samples), random_state=42
                    )
            except Exception:
                sil = np.nan
            metrics['silhouette'].append(sil)
            metrics['calinski_harabasz'].append(calinski_harabasz_score(self.features_scaled, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(self.features_scaled, labels))
            metrics['inertia'].append(kmeans.inertia_)
            
            print(f"  k={k}: Silhouette={metrics['silhouette'][-1]:.3f}, "
                  f"CH={metrics['calinski_harabasz'][-1]:.1f}, "
                  f"DB={metrics['davies_bouldin'][-1]:.3f}")
        
        # Convert to DataFrame
        self.cluster_metrics = pd.DataFrame(metrics)
        
        # Find optimal k
        # Best k has high silhouette, high CH, low DB
        # Combine scores; if silhouette is NaN (too large), ignore it in ranking
        sil = self.cluster_metrics['silhouette']
        ch = self.cluster_metrics['calinski_harabasz']
        db = self.cluster_metrics['davies_bouldin']

        # Normalize safely and keep Series types
        if sil.notna().any():
            sil_norm = sil / sil.max(skipna=True)
        else:
            sil_norm = pd.Series(0.0, index=self.cluster_metrics.index)

        ch_norm = ch / (ch.max() if ch.max() != 0 else 1)
        db_norm = db / (db.max() if db.max() != 0 else 1)

        scores = sil_norm.fillna(0) + ch_norm.fillna(0) - db_norm.fillna(0)
        
        optimal_k = self.cluster_metrics.loc[scores.idxmax(), 'k']
        
        print(f"\nOptimal number of clusters: {int(optimal_k)}")
        
        # Create elbow plot
        self._plot_elbow_curve()
        
        return int(optimal_k)
    
    def _plot_elbow_curve(self):
        """Plot elbow curve and metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Elbow curve
        axes[0, 0].plot(self.cluster_metrics['k'], self.cluster_metrics['inertia'], 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Curve')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[0, 1].plot(self.cluster_metrics['k'], self.cluster_metrics['silhouette'], 'go-')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score (higher is better)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz
        axes[1, 0].plot(self.cluster_metrics['k'], self.cluster_metrics['calinski_harabasz'], 'ro-')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Score (higher is better)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Davies-Bouldin
        axes[1, 1].plot(self.cluster_metrics['k'], self.cluster_metrics['davies_bouldin'], 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].set_title('Davies-Bouldin Score (lower is better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Cluster Validation Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cluster_validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_clustering(self, n_clusters=None, method='kmeans'):
        """
        Perform watershed clustering
        
        Args:
            n_clusters: Number of clusters (if None, determine optimal)
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
        """
        print("\n" + "="*60)
        print("PERFORMING WATERSHED CLUSTERING")
        print("="*60)
        
        if self.features_scaled is None:
            print("Error: Features not prepared. Run prepare_features() first.")
            return None
        
        # Determine optimal clusters if not specified
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters()
        
        print(f"\nClustering with {method} (k={n_clusters})...")
        
        if method == 'kmeans':
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            self.clusters = self.clusterer.fit_predict(self.features_scaled)
            
        elif method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            self.clusters = self.clusterer.fit_predict(self.features_scaled)
            
            # Create dendrogram
            self._create_dendrogram()
            
        elif method == 'dbscan':
            # For DBSCAN, need to determine eps
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors_fit = neighbors.fit(self.features_scaled)
            distances, indices = neighbors_fit.kneighbors(self.features_scaled)
            distances = np.sort(distances[:, -1], axis=0)
            
            # Use knee point as eps
            eps = np.percentile(distances, 90)
            
            self.clusterer = DBSCAN(eps=eps, min_samples=5)
            self.clusters = self.clusterer.fit_predict(self.features_scaled)
            
            # Adjust cluster labels
            unique_labels = np.unique(self.clusters)
            n_clusters = len(unique_labels[unique_labels >= 0])
            n_noise = (self.clusters == -1).sum()
            
            print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add clusters to watersheds
        self.watersheds_with_fire['fire_regime_cluster'] = self.clusters
        
        # Analyze clusters
        self._analyze_clusters()
        
        # Create visualizations
        self._visualize_clusters()
        
        return self.clusters
    
    def _analyze_clusters(self):
        """Analyze and characterize each cluster"""
        print("\n" + "="*60)
        print("CLUSTER CHARACTERIZATION")
        print("="*60)
        
        unique_clusters = np.unique(self.clusters[self.clusters >= 0])
        
        for cluster_id in unique_clusters:
            mask = self.clusters == cluster_id
            n_watersheds = mask.sum()
            
            print(f"\nCluster {cluster_id}: {n_watersheds} watersheds")
            print("-" * 40)
            
            # Get cluster statistics
            cluster_data = self.watersheds_with_fire[mask]
            
            # Fire frequency
            print(f"Fire frequency:")
            print(f"  Episodes: {cluster_data['episode_count'].mean():.1f} ± {cluster_data['episode_count'].std():.1f}")
            
            # Fire size
            print(f"Fire size:")
            print(f"  Mean area: {cluster_data['mean_fire_area_km2'].mean():.1f} ± {cluster_data['mean_fire_area_km2'].std():.1f} km²")
            print(f"  HSBF: {cluster_data['hsbf'].mean():.3f} ± {cluster_data['hsbf'].std():.3f}")
            
            # Fire intensity
            print(f"Fire intensity:")
            print(f"  Mean FRP: {cluster_data['mean_frp_mw'].mean():.1f} ± {cluster_data['mean_frp_mw'].std():.1f} MW")
            
            # Temporal patterns
            print(f"Temporal patterns:")
            print(f"  Duration: {cluster_data['mean_duration_days'].mean():.1f} ± {cluster_data['mean_duration_days'].std():.1f} days")
            print(f"  Seasonality: {cluster_data['seasonality_index'].mean():.2f} ± {cluster_data['seasonality_index'].std():.2f}")
            
            # Assign cluster name based on characteristics
            self._name_cluster(cluster_id, cluster_data)
    
    def _name_cluster(self, cluster_id, cluster_data):
        """Assign descriptive name to cluster based on characteristics"""
        
        # Determine fire frequency category
        episodes = cluster_data['episode_count'].mean()
        if episodes < 5:
            freq = "Low"
        elif episodes < 15:
            freq = "Moderate"
        else:
            freq = "High"
        
        # Determine fire size category
        area = cluster_data['mean_fire_area_km2'].mean()
        if area < 10:
            size = "Small"
        elif area < 50:
            size = "Medium"
        else:
            size = "Large"
        
        # Determine intensity category
        frp = cluster_data['mean_frp_mw'].mean()
        if frp < 50:
            intensity = "Low-Intensity"
        elif frp < 100:
            intensity = "Moderate-Intensity"
        else:
            intensity = "High-Intensity"
        
        # Combine for name
        name = f"{freq}-Frequency {size}-{intensity}"
        
        print(f"\nCluster {cluster_id} Name: {name}")
        
        # Store in dictionary
        if not hasattr(self, 'cluster_names'):
            self.cluster_names = {}
        self.cluster_names[cluster_id] = name
    
    def _create_dendrogram(self):
        """Create hierarchical clustering dendrogram"""
        plt.figure(figsize=(12, 8))
        
        # Calculate linkage
        Z = linkage(self.features_scaled, method='ward')
        
        # Create dendrogram
        dendrogram(Z, truncate_mode='level', p=5)
        
        plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        plt.xlabel('Watershed Index')
        plt.ylabel('Distance')
        plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_clusters(self):
        """Create cluster visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. PCA visualization
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.pca_features[:, 0], self.pca_features[:, 1], 
                            c=self.clusters, cmap='viridis', s=50, alpha=0.7)
        ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%})')
        ax1.set_title('Clusters in PCA Space')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Fire frequency vs HSBF
        ax2 = axes[0, 1]
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            mask = self.clusters == cluster_id
            cluster_data = self.watersheds_with_fire[mask]
            ax2.scatter(cluster_data['episode_count'], cluster_data['hsbf'], 
                       label=f'Cluster {cluster_id}', s=30, alpha=0.6)
        ax2.set_xlabel('Fire Episode Count')
        ax2.set_ylabel('HSBF')
        ax2.set_title('Fire Frequency vs Impact')
        ax2.legend()
        ax2.set_xscale('log')
        
        # 3. Fire size vs intensity
        ax3 = axes[1, 0]
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            mask = self.clusters == cluster_id
            cluster_data = self.watersheds_with_fire[mask]
            ax3.scatter(cluster_data['mean_fire_area_km2'], cluster_data['mean_frp_mw'], 
                       label=f'Cluster {cluster_id}', s=30, alpha=0.6)
        ax3.set_xlabel('Mean Fire Area (km²)')
        ax3.set_ylabel('Mean FRP (MW)')
        ax3.set_title('Fire Size vs Intensity')
        ax3.legend()
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # 4. Cluster sizes
        ax4 = axes[1, 1]
        cluster_sizes = pd.Series(self.clusters[self.clusters >= 0]).value_counts().sort_index()
        ax4.bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', alpha=0.7)
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Number of Watersheds')
        ax4.set_title('Cluster Sizes')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Fire Regime Cluster Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_validation(self):
        """Perform statistical validation of clusters"""
        print("\n" + "="*60)
        print("STATISTICAL VALIDATION")
        print("="*60)
        
        if self.clusters is None:
            print("Error: No clusters found. Run perform_clustering() first.")
            return None
        
        # ANOVA for continuous variables
        print("\nANOVA Results (p-values):")
        print("-" * 40)
        
        continuous_vars = ['episode_count', 'mean_fire_area_km2', 'hsbf', 
                          'mean_frp_mw', 'mean_duration_days', 'seasonality_index']
        
        for var in continuous_vars:
            if var in self.watersheds_with_fire.columns:
                groups = []
                for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
                    mask = self.clusters == cluster_id
                    groups.append(self.watersheds_with_fire[mask][var].values)
                
                f_stat, p_value = f_oneway(*groups)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {var}: p={p_value:.4f} {significance}")
        
        # Silhouette analysis (guard for large n)
        try:
            if self.features_scaled.shape[0] <= 20000:
                print(f"\nOverall Silhouette Score: {silhouette_score(self.features_scaled, self.clusters):.3f}")
            else:
                print("\nOverall Silhouette Score: skipped for large dataset (>20k) to avoid high memory use")
        except Exception:
            print("\nOverall Silhouette Score: unavailable")
        
        # Cluster separation
        # Memory-safe cluster separation: use centroid distances instead of full pairwise matrix
        print("\nInter-cluster Centroid Distances:")
        unique_clusters = np.unique(self.clusters[self.clusters >= 0])
        centroids = {}
        for cid in unique_clusters:
            centroids[cid] = self.features_scaled[self.clusters == cid].mean(axis=0)
        # Compute pairwise distances between centroids
        centroid_ids = list(centroids.keys())
        centroid_matrix = np.vstack([centroids[cid] for cid in centroid_ids])
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(centroid_matrix))
        for i in range(len(centroid_ids)):
            for j in range(i+1, len(centroid_ids)):
                print(f"  Cluster {centroid_ids[i]} - Cluster {centroid_ids[j]}: {dists[i, j]:.2f}")
    
    def export_results(self, output_path='watershed_fire_regimes.gpkg'):
        """Export clustered watersheds to file"""
        print(f"\nExporting results to {output_path}...")
        
        # Add cluster names
        if hasattr(self, 'cluster_names'):
            self.watersheds_with_fire['fire_regime_name'] = self.watersheds_with_fire['fire_regime_cluster'].map(self.cluster_names)
        
        # Export
        self.watersheds_with_fire.to_file(output_path, driver='GPKG')
        
        # Also export summary statistics
        summary = self.watersheds_with_fire.groupby('fire_regime_cluster').agg({
            'episode_count': ['mean', 'std', 'min', 'max'],
            'hsbf': ['mean', 'std', 'min', 'max'],
            'mean_fire_area_km2': ['mean', 'std'],
            'mean_frp_mw': ['mean', 'std'],
            'seasonality_index': ['mean', 'std']
        }).round(3)
        
        summary.to_csv('fire_regime_summary_statistics.csv')
        
        print("Export complete!")
        
        return self.watersheds_with_fire