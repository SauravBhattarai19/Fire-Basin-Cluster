#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('geopandas', 'GeoPandas'),
        ('shapely', 'Shapely'),
        ('pyproj', 'PyProj'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('yaml', 'PyYAML'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('psutil', 'PSUtil'),
        ('tqdm', 'TQDM'),
        ('rtree', 'Rtree'),
        ('pyarrow', 'PyArrow')
    ]
    
    failed = []
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            failed.append(name)
    
    # Test optional GPU packages
    print("\nTesting optional GPU packages...")
    gpu_packages = [
        ('cupy', 'CuPy'),
        ('cuml', 'cuML (RAPIDS)'),
        ('cuspatial', 'cuSpatial (RAPIDS)'),
        ('pynvml', 'PyNVML')
    ]
    
    gpu_available = False
    for package, name in gpu_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {name}")
            gpu_available = True
        except ImportError:
            print(f"- {name} - Not installed (optional)")
    
    if failed:
        print(f"\n❌ Missing required packages: {', '.join(failed)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages installed successfully!")
        if gpu_available:
            print("✅ GPU acceleration available")
        else:
            print("ℹ️  GPU acceleration not available (optional)")
        return True

def test_fire_clustering_modules():
    """Test that custom modules can be imported"""
    print("\nTesting fire clustering modules...")
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent / 'src'))
    
    modules = [
        ('utils', 'Utilities'),
        ('data_preparation', 'Data Preparation'),
        ('clustering', 'Clustering'),
        ('episode_characterization', 'Episode Characterization'),
        ('validation', 'Validation Framework')
    ]
    
    failed = []
    
    for module, name in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {name} module")
        except ImportError as e:
            print(f"✗ {name} module - Error: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed to import modules: {', '.join(failed)}")
        return False
    else:
        print("\n✅ All fire clustering modules loaded successfully!")
        return True

def test_data_files():
    """Check if data files exist"""
    print("\nChecking data files...")
    
    data_files = [
        ('../Json_files/fire_modis_us.json', 'MODIS fire data'),
        ('../Json_files/huc12_conus.geojson', 'HUC12 watershed data'),
        ('config/config.yaml', 'Configuration file')
    ]
    
    missing = []
    
    for file_path, name in data_files:
        path = Path(__file__).parent / file_path
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {name} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {name} - NOT FOUND at {path}")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing data files: {', '.join(missing)}")
        print("Note: The system can still run, but you'll need to update data paths in config.yaml")
        return False
    else:
        print("\n✅ All data files found!")
        return True

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    
    if not config_path.exists():
        print("✗ Configuration file not found")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key sections
        required_sections = ['study_area', 'clustering', 'processing', 'output']
        missing = []
        
        for section in required_sections:
            if section in config:
                print(f"✓ Config section: {section}")
            else:
                print(f"✗ Config section: {section} - MISSING")
                missing.append(section)
        
        if missing:
            print(f"\n❌ Missing configuration sections: {', '.join(missing)}")
            return False
        else:
            print("\n✅ Configuration file is valid!")
            
            # Print current settings
            print(f"\nCurrent configuration:")
            print(f"  Test mode: {config['study_area']['test_mode']}")
            print(f"  Bounding box: {config['study_area']['bounding_box']}")
            print(f"  Spatial eps: {config['clustering']['spatial_eps_meters']}m")
            print(f"  Temporal eps: {config['clustering']['temporal_eps_days']} days")
            
            return True
            
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

def test_output_directory():
    """Test output directory creation"""
    print("\nTesting output directory creation...")
    
    try:
        from src.utils import create_output_directory, load_config
        
        config_path = Path(__file__).parent / 'config' / 'config.yaml'
        config = load_config(str(config_path))
        
        # Temporarily set test mode
        config['study_area']['test_mode'] = True
        config['data']['output_base_dir'] = 'test_outputs'
        
        output_dir = create_output_directory(config)
        
        if output_dir.exists():
            print(f"✓ Created test output directory: {output_dir}")
            
            # Check subdirectories
            subdirs = ['episodes', 'validation', 'checkpoints', 'logs', 'visualizations']
            for subdir in subdirs:
                if (output_dir / subdir).exists():
                    print(f"  ✓ {subdir}/")
            
            # Clean up
            import shutil
            shutil.rmtree(output_dir.parent)
            print("✓ Cleaned up test directory")
            
            return True
        else:
            print("✗ Failed to create output directory")
            return False
            
    except Exception as e:
        print(f"✗ Error testing output directory: {e}")
        return False

def run_all_tests():
    """Run all installation tests"""
    print("="*60)
    print("FIRE EPISODE CLUSTERING - INSTALLATION TEST")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Module Imports", test_fire_clustering_modules),
        ("Data Files", test_data_files),
        ("Configuration", test_configuration),
        ("Output Directory", test_output_directory)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "✅" if success else "❌"
        print(f"{symbol} {test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Review and adjust config/config.yaml for your needs")
        print("2. Run: python fire_episode_clustering.py config/config.yaml")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")
        print("Most common fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Update data file paths in config/config.yaml")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 