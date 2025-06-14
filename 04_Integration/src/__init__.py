"""
Fire Episode Clustering System
Spatiotemporal clustering of MODIS FIRMS fire data
"""

__version__ = "1.0.0"
__author__ = "Fire Analysis Team"

# Import main components
from .data_preparation import DataPreparation
from .clustering import SpatioTemporalDBSCAN
from .episode_characterization import EpisodeCharacterization
from .validation import ValidationFramework
from .utils import (
    load_config,
    setup_logging,
    create_output_directory,
    PerformanceMonitor,
    SpatialUtils
)

__all__ = [
    'DataPreparation',
    'SpatioTemporalDBSCAN',
    'EpisodeCharacterization',
    'ValidationFramework',
    'load_config',
    'setup_logging',
    'create_output_directory',
    'PerformanceMonitor',
    'SpatialUtils'
] 