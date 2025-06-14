#!/bin/bash
# Example commands for running the Fire Episode Clustering System

echo "=========================================="
echo "Fire Episode Clustering - Example Commands"
echo "=========================================="

# Basic test run (California subset)
echo "1. Basic test run (California subset):"
echo "python fire_episode_clustering.py config/config.yaml"
echo ""

# Run with parameter optimization
echo "2. Run with parameter optimization:"
echo "python fire_episode_clustering.py config/config.yaml --optimize-params"
echo ""

# Resume from checkpoint
echo "3. Resume from checkpoint:"
echo "python fire_episode_clustering.py config/config.yaml --resume outputs/test_run_*/checkpoints/stage2_clustering.pkl"
echo ""

# Production run (modify config first)
echo "4. Production run (full dataset):"
echo "# First, edit config/config.yaml and set:"
echo "#   study_area:"
echo "#     test_mode: false"
echo "#     bounding_box: [-125, 25, -66, 50]  # Full CONUS"
echo "python fire_episode_clustering.py config/config.yaml"
echo ""

# Custom configuration
echo "5. Run with custom configuration:"
echo "cp config/config.yaml config/my_config.yaml"
echo "# Edit my_config.yaml as needed"
echo "python fire_episode_clustering.py config/my_config.yaml"
echo ""

# Installation test
echo "6. Test installation:"
echo "python test_installation.py"
echo ""

# View results
echo "7. View results:"
echo "# Results are saved in timestamped directories under outputs/"
echo "ls -la outputs/"
echo "# View validation report:"
echo "open outputs/test_run_*/validation/validation_report.html"
echo ""

echo "=========================================="
echo "For more information, see README.md"
echo "==========================================" 