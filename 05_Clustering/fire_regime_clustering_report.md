# Fire Regime Clustering Analysis Report

Generated: 2025-08-20 12:57:02

## Executive Summary

- Total watersheds analyzed: 85,840
- Watersheds with fire activity: 48,853 (56.9%)
- Fire regime clusters identified: 13

## Methodology

### Addressing Bias in Fire Regime Clustering

The analysis employs several strategies to address bias from watersheds with minimal fire activity:

1. **Stratified Analysis**: Watersheds are first stratified by fire activity level
2. **Focused Clustering**: Only fire-affected watersheds are clustered
3. **Multi-Algorithm Ensemble**: Multiple clustering algorithms are tested
4. **Robust Transformations**: Multiple feature transformations are applied
5. **Parallel Processing**: Leverages high-performance computing for comprehensive analysis

### Fire Activity Stratification

| Strata | Count | Percentage |
|--------|-------|------------|
| No Fire | 36,987 | 43.1% |
| Rare Fire | 33,530 | 39.1% |
| Occasional Fire | 12,067 | 14.1% |
| Frequent Fire | 2,728 | 3.2% |
| Very Frequent Fire | 528 | 0.6% |

## Cluster Characteristics

### Cluster 0: Occasional_High-Impact_Large_High-Intensity

- **Size**: 38 watersheds
- **Fire Episodes**: 3.5 ± 2.9
- **HSBF**: 0.487 (max: 1.000)
- **Mean Fire Area**: 21.3 km²
- **Mean FRP**: 186.7 MW
- **Duration**: 11.1 days
- **Seasonality**: 0.32
- **Return Interval**: 4.3 years

### Cluster 1: Occasional_Low-Impact_Small_High-Intensity

- **Size**: 45520 watersheds
- **Fire Episodes**: 6.7 ± 11.0
- **HSBF**: 0.084 (max: 1.000)
- **Mean Fire Area**: 3.9 km²
- **Mean FRP**: 74.3 MW
- **Duration**: 2.4 days
- **Seasonality**: 0.29
- **Return Interval**: 2.7 years

### Cluster 2: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 811 watersheds
- **Fire Episodes**: 5.5 ± 4.7
- **HSBF**: 0.643 (max: 1.000)
- **Mean Fire Area**: 21.7 km²
- **Mean FRP**: 162.8 MW
- **Duration**: 14.4 days
- **Seasonality**: 0.40
- **Return Interval**: 3.9 years

### Cluster 3: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 1461 watersheds
- **Fire Episodes**: 5.5 ± 23.6
- **HSBF**: 0.561 (max: 1.000)
- **Mean Fire Area**: 22.1 km²
- **Mean FRP**: 175.6 MW
- **Duration**: 11.9 days
- **Seasonality**: 0.38
- **Return Interval**: 3.7 years

### Cluster 4: Frequent_Extreme-Impact_Large_High-Intensity

- **Size**: 15 watersheds
- **Fire Episodes**: 10.8 ± 29.7
- **HSBF**: 0.787 (max: 1.000)
- **Mean Fire Area**: 43.0 km²
- **Mean FRP**: 194.4 MW
- **Duration**: 28.7 days
- **Seasonality**: 0.38
- **Return Interval**: 4.6 years

### Cluster 5: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 19 watersheds
- **Fire Episodes**: 6.2 ± 3.4
- **HSBF**: 0.730 (max: 1.000)
- **Mean Fire Area**: 22.5 km²
- **Mean FRP**: 156.3 MW
- **Duration**: 26.7 days
- **Seasonality**: 0.51
- **Return Interval**: 3.7 years

### Cluster 6: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 93 watersheds
- **Fire Episodes**: 3.8 ± 2.2
- **HSBF**: 0.806 (max: 1.000)
- **Mean Fire Area**: 27.9 km²
- **Mean FRP**: 126.2 MW
- **Duration**: 30.9 days
- **Seasonality**: 0.33
- **Return Interval**: 4.1 years

### Cluster 7: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 116 watersheds
- **Fire Episodes**: 4.6 ± 2.3
- **HSBF**: 0.828 (max: 1.000)
- **Mean Fire Area**: 28.5 km²
- **Mean FRP**: 175.7 MW
- **Duration**: 29.4 days
- **Seasonality**: 0.44
- **Return Interval**: 4.2 years

### Cluster 8: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 2 watersheds
- **Fire Episodes**: 6.0 ± 1.4
- **HSBF**: 0.573 (max: 0.674)
- **Mean Fire Area**: 22.0 km²
- **Mean FRP**: 329.7 MW
- **Duration**: 15.9 days
- **Seasonality**: 0.40
- **Return Interval**: 3.4 years

### Cluster 9: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 10 watersheds
- **Fire Episodes**: 3.8 ± 2.0
- **HSBF**: 0.952 (max: 1.000)
- **Mean Fire Area**: 36.1 km²
- **Mean FRP**: 267.5 MW
- **Duration**: 31.4 days
- **Seasonality**: 0.36
- **Return Interval**: 7.1 years

### Cluster 10: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 2 watersheds
- **Fire Episodes**: 9.0 ± 7.1
- **HSBF**: 0.595 (max: 0.707)
- **Mean Fire Area**: 25.7 km²
- **Mean FRP**: 107.1 MW
- **Duration**: 18.2 days
- **Seasonality**: 0.63
- **Return Interval**: 2.9 years

### Cluster 11: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 11 watersheds
- **Fire Episodes**: 4.3 ± 2.2
- **HSBF**: 0.789 (max: 1.000)
- **Mean Fire Area**: 25.7 km²
- **Mean FRP**: 263.9 MW
- **Duration**: 19.0 days
- **Seasonality**: 0.42
- **Return Interval**: 3.4 years

### Cluster 12: Occasional_Extreme-Impact_Large_High-Intensity

- **Size**: 755 watersheds
- **Fire Episodes**: 5.6 ± 4.2
- **HSBF**: 0.737 (max: 1.000)
- **Mean Fire Area**: 22.3 km²
- **Mean FRP**: 154.9 MW
- **Duration**: 22.7 days
- **Seasonality**: 0.42
- **Return Interval**: 3.9 years

## Model Performance

### Best Clustering Configuration

- **Algorithm**: kmeans
- **Transform**: robust
- **Number of Clusters**: 3
- **Silhouette Score**: 0.871
- **Davies-Bouldin Score**: 0.450
- **Calinski-Harabasz Score**: 130831.8

## Recommendations

1. **Validation**: Validate clusters against known fire-prone regions
2. **Ecological Context**: Incorporate vegetation and climate data
3. **Temporal Analysis**: Examine cluster stability over time
4. **Management Applications**: Develop cluster-specific fire management strategies
5. **Prediction Models**: Use clusters as basis for fire risk prediction

## Technical Notes

- Parallel workers used: 32
- Clustering experiments conducted: 168
- Feature transformations: standard, robust, quantile, log-robust
- Algorithms tested: K-Means, GMM, Bayesian GMM, HDBSCAN, OPTICS, Spectral, Hierarchical, BIRCH

