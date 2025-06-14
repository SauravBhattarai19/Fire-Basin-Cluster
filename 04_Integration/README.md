# Fire Episode Clustering System

A scalable, configuration-driven system for spatiotemporal clustering of MODIS FIRMS fire data to generate fire episode records for watershed fire regime analysis.

## Overview

This system implements advanced spatiotemporal clustering using DBSCAN to identify fire episodes from MODIS satellite detections. It's designed to handle continental-scale datasets while being testable on geographic subsets.

### Key Features

- **Configuration-driven architecture** - All parameters controlled via YAML config
- **Spatiotemporal DBSCAN clustering** - Custom distance metrics for fire behavior
- **GPU acceleration support** - Optional RAPIDS integration for large datasets  
- **Comprehensive episode characterization** - Temporal, spatial, intensity metrics
- **Built-in validation framework** - Quality assessment and parameter optimization
- **Checkpoint/resume capability** - For long-running processes
- **Multiple output formats** - Parquet, GeoJSON, CSV

## Installation

### Basic Installation

```bash
# Clone or navigate to the project directory
cd 04_Integration

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

For GPU acceleration with NVIDIA GPUs:

```bash
# Install RAPIDS (adjust CUDA version as needed)
conda install -c rapidsai -c nvidia -c conda-forge \
    cuml=23.12 cuspatial=23.12 python=3.9 cudatoolkit=11.8
```

## Quick Start

### 1. Configure the System

Edit `config/config.yaml` to set your parameters:

```yaml
study_area:
  test_mode: true  # Start with test mode
  bounding_box: [-125, 32, -114, 42]  # California

clustering:
  spatial_eps_meters: 2000  # 2km spatial threshold
  temporal_eps_days: 3      # 3 day temporal threshold
  min_samples: 2            # Minimum detections
```

### 2. Run the Pipeline

```bash
# Basic run
python fire_episode_clustering.py config/config.yaml

# With parameter optimization
python fire_episode_clustering.py config/config.yaml --optimize-params

# Resume from checkpoint
python fire_episode_clustering.py config/config.yaml --resume outputs/test_run_*/checkpoints/stage2_clustering.pkl
```

### 3. Check Results

Results are saved in timestamped directories under `outputs/`:

```
outputs/test_run_20240115_143022/
├── episodes/
│   ├── fire_episodes.parquet      # Main episode records
│   ├── fire_episodes.geojson      # GeoJSON format
│   └── watershed_fire_statistics.geojson
├── validation/
│   ├── validation_report.html     # Human-readable report
│   ├── validation_report.json     # Detailed metrics
│   └── clustering_validation.png  # Diagnostic plots
├── config.yaml                    # Copy of config used
└── pipeline_results.json          # Overall run summary
```

## Configuration Guide

### Key Parameters

#### Study Area
- `test_mode`: Boolean to switch between test/production
- `bounding_box`: [west, south, east, north] in decimal degrees

#### Clustering Parameters
- `spatial_eps_meters`: Maximum distance between fire detections (meters)
- `temporal_eps_days`: Maximum time gap between detections (days)
- `min_samples`: Minimum detections to form a cluster
- `handle_day_night`: How to handle day/night detections
  - `"combined"`: Cluster together (default)
  - `"separate"`: Cluster separately then merge
  - `"weighted"`: Use weighted distance

#### Performance Settings
- `max_cpu_cores`: Number of CPU cores to use (default: 48)
- `use_gpu_acceleration`: Enable GPU if available
- `chunk_size_mb`: Memory chunk size for processing

### Parameter Optimization

The system includes automated parameter optimization:

```bash
python fire_episode_clustering.py config/config.yaml --optimize-params
```

This tests combinations of parameters specified in:

```yaml
clustering:
  param_ranges:
    spatial_eps: [1000, 2000, 3000, 5000]  # meters
    temporal_eps: [1, 2, 3, 5, 7]          # days
    min_samples: [2, 3, 5]                 # points
```

## Output Description

### Episode Records

Each fire episode has these attributes:

**Temporal Metrics:**
- `start_datetime`, `end_datetime`: Episode time bounds
- `duration_hours`, `duration_days`: Total duration
- `active_days`: Days with detections
- `dormancy_periods`: Number of gaps in detection

**Spatial Metrics:**
- `centroid_lat`, `centroid_lon`: Episode center
- `bounding_box`: Geographic extent [west, south, east, north]
- `area_km2`: Estimated burned area
- `shape_elongation`: Shape complexity measure
- `spread_direction_deg`: Mean spread direction (0-360°)
- `spread_rate_kmh`: Fire spread velocity

**Intensity Metrics:**
- `total_energy_mwh`: Cumulative fire radiative energy
- `peak_frp`, `mean_frp`: Fire radiative power statistics
- `peak_brightness`: Maximum brightness temperature

**Quality Metrics:**
- `detection_count`: Number of satellite detections
- `mean_confidence`: Average detection confidence
- `spatial_coherence_score`: Clustering quality (0-1)
- `data_completeness_score`: Temporal coverage (0-1)

## Workflow Stages

1. **Data Preparation**
   - Load MODIS fire data
   - Apply quality filters (confidence ≥70%)
   - Geographic clipping
   - Remove duplicates

2. **Spatiotemporal Clustering**
   - Transform to equal-area projection
   - Apply DBSCAN with custom metrics
   - Handle day/night detections
   - Post-process clusters

3. **Episode Characterization**
   - Calculate comprehensive metrics
   - Classify episode types
   - Validate against physical constraints

4. **Validation & Assessment**
   - Clustering quality metrics
   - Episode validation
   - Generate reports and visualizations

## Performance Considerations

### Memory Usage

For large datasets:
- Enable disk caching in config
- Use spatial chunking for >1M points
- Reduce `chunk_size_mb` if running out of memory

### Processing Time

Typical performance on test hardware:
- California subset (~100K points): 5-10 minutes
- Western US (~500K points): 30-60 minutes  
- Full CONUS (~3M points): 3-6 hours

### GPU Acceleration

GPU provides 5-10x speedup for clustering:
- Requires NVIDIA GPU with ≥8GB VRAM
- Install RAPIDS libraries
- Set `use_gpu_acceleration: true`

## Troubleshooting

### Common Issues

1. **Memory errors**
   - Reduce `chunk_size_mb` in config
   - Enable `enable_disk_caching`
   - Process smaller geographic areas

2. **Too many/few clusters**
   - Run parameter optimization
   - Adjust `spatial_eps_meters` and `temporal_eps_days`
   - Check data quality filters

3. **Invalid episodes**
   - Review validation criteria in config
   - Check `max_episode_duration_days`
   - Verify coordinate system compatibility

### Debug Mode

Enable debug logging:

```yaml
logging:
  log_level: "DEBUG"
```

## Citation

If using this system in research, please cite:

```
Fire Episode Clustering System (2024)
Advanced spatiotemporal clustering for MODIS FIRMS fire data
[Your institution/repository]
```

## License

[Specify your license]

## Contact

[Your contact information] 