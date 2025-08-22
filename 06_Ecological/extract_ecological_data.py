#!/usr/bin/env python3
"""
Extract Ecological Variables using Google Earth Engine
For fire-watershed clustering enhancement
Project ID: ee-jsuhydrolabenb
"""

import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
import time

# Initialize Earth Engine with your project
PROJECT_ID = 'ee-jsuhydrolabenb'

def initialize_gee():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"✓ Google Earth Engine initialized with project: {PROJECT_ID}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize GEE: {e}")
        print("\nTrying to authenticate...")
        try:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)
            print(f"✓ Successfully authenticated and initialized")
            return True
        except:
            print("Please run: earthengine authenticate")
            return False

class EcologicalDataExtractor:
    """
    Extract comprehensive ecological variables from GEE
    """
    
    def __init__(self, watersheds_gdf, year_range=(2015, 2023)):
        """
        Initialize extractor
        
        Args:
            watersheds_gdf: GeoDataFrame with watershed boundaries
            year_range: Years for climate averaging
        """
        self.watersheds = watersheds_gdf.copy()
        self.start_year = year_range[0]
        self.end_year = year_range[1]
        
        # Pre-process geometries to ensure they're valid
        print("Validating and fixing geometries...")
        invalid_count = 0
        for idx in self.watersheds.index:
            geom = self.watersheds.at[idx, 'geometry']
            if not geom.is_valid:
                # Try to fix invalid geometry
                self.watersheds.at[idx, 'geometry'] = geom.buffer(0)
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"   Fixed {invalid_count} invalid geometries")
        
        # Initialize GEE
        if not initialize_gee():
            raise RuntimeError("Failed to initialize Google Earth Engine")
        
        print(f"\n📍 Ecological Data Extractor initialized")
        print(f"   Watersheds: {len(self.watersheds)}")
        print(f"   Time period: {self.start_year}-{self.end_year}")
    
    def prepare_datasets(self):
        """
        Prepare all GEE datasets for extraction
        """
        print("\n" + "="*60)
        print("PREPARING GEE DATASETS")
        print("="*60)
        
        self.datasets = {}
        
        # 1. Elevation data (SRTM 30m)
        print("Loading elevation data...")
        self.datasets['elevation'] = ee.Image('USGS/SRTMGL1_003')
        
        # 2. Land cover (MODIS - updated to current version)
        print("Loading land cover data...")
        self.datasets['landcover'] = ee.ImageCollection('MODIS/061/MCD12Q1') \
            .filter(ee.Filter.date(f'{self.end_year}-01-01', f'{self.end_year}-12-31')) \
            .first() \
            .select('LC_Type1')
        
        # 3. Vegetation indices (MODIS NDVI/EVI - updated to current version)
        print("Loading vegetation indices...")
        self.datasets['vegetation'] = ee.ImageCollection('MODIS/061/MOD13A2') \
            .filter(ee.Filter.date(f'{self.start_year}-01-01', f'{self.end_year}-12-31'))
        
        # 4. Climate data (TerraClimate)
        print("Loading climate data...")
        self.datasets['climate'] = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
            .filter(ee.Filter.date(f'{self.start_year}-01-01', f'{self.end_year}-12-31'))
        
        # 5. Soil properties (OpenLandMap)
        print("Loading soil data...")
        self.datasets['soil'] = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
        
        # 6. Distance to roads (Global Roads from GRIP)
        print("Loading infrastructure data...")
        # Using TIGER roads for US
        self.datasets['roads'] = ee.FeatureCollection('TIGER/2016/Roads')
        
        # 7. Population density (WorldPop)
        print("Loading population data...")
        self.datasets['population'] = ee.ImageCollection('WorldPop/GP/100m/pop') \
            .filter(ee.Filter.date(f'{self.end_year}-01-01', f'{self.end_year}-12-31')) \
            .mean()
        
        # 8. Forest cover and change (Hansen Global Forest Change - updated version)
        print("Loading forest data...")
        self.datasets['forest'] = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
        
        print("\n✓ All datasets prepared")
        
        return self.datasets
    
    def extract_topographic_variables(self, feature):
        """
        Extract topographic variables for a watershed
        """
        try:
            # Get elevation
            elevation = self.datasets['elevation'].select('elevation')
            
            # Calculate slope and aspect
            terrain = ee.Terrain.products(elevation)
            
            # Create composite with proper band names
            topo_composite = ee.Image.cat([
                elevation.rename('elevation'),
                terrain.select('slope').rename('slope'),
                terrain.select('aspect').rename('aspect')
            ])
            
            # Create reducers
            reducers = ee.Reducer.mean() \
                .combine(ee.Reducer.stdDev(), '', True) \
                .combine(ee.Reducer.min(), '', True) \
                .combine(ee.Reducer.max(), '', True)
            
            # Extract statistics
            stats = topo_composite.reduceRegion(
                reducer=reducers,
                geometry=feature.geometry(),
                scale=30,
                maxPixels=1e9,
                bestEffort=True
            )
            
            # Calculate ruggedness safely
            elev_max = stats.get('elevation_max', 0)
            elev_min = stats.get('elevation_min', 0)
            ruggedness = ee.Number(elev_max).subtract(ee.Number(elev_min))
            
            return {
                'elevation_mean': stats.get('elevation_mean', 0),
                'elevation_std': stats.get('elevation_stdDev', 0),
                'elevation_min': stats.get('elevation_min', 0),
                'elevation_max': stats.get('elevation_max', 0),
                'slope_mean': stats.get('slope_mean', 0),
                'slope_std': stats.get('slope_stdDev', 0),
                'aspect_mean': stats.get('aspect_mean', 0),
                'ruggedness': ruggedness
            }
            
        except Exception as e:
            print(f"Error in topographic extraction: {e}")
            return {
                'elevation_mean': 0,
                'elevation_std': 0,
                'elevation_min': 0,
                'elevation_max': 0,
                'slope_mean': 0,
                'slope_std': 0,
                'aspect_mean': 0,
                'ruggedness': 0
            }
    
    def extract_vegetation_variables(self, feature):
        """
        Extract vegetation and land cover variables
        """
        try:
            # Mean NDVI and EVI
            ndvi_mean = self.datasets['vegetation'].select('NDVI').mean()
            evi_mean = self.datasets['vegetation'].select('EVI').mean()
            
            # Vegetation seasonality (std of NDVI)
            ndvi_std = self.datasets['vegetation'].select('NDVI').reduce(ee.Reducer.stdDev())
            
            # Create composite image with proper band names
            veg_composite = ee.Image.cat([
                ndvi_mean.rename('ndvi_mean'),
                evi_mean.rename('evi_mean'),
                ndvi_std.rename('ndvi_std')
            ])
            
            # Extract stats
            veg_stats = veg_composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=feature.geometry(),
                scale=250,
                maxPixels=1e9,
                bestEffort=True
            )
            
            # Land cover composition
            landcover = self.datasets['landcover']
            
            # Calculate fractions using simpler approach
            pixel_area = ee.Image.pixelArea()
            
            # Forest types (classes 1-5)
            forest_mask = landcover.gte(1).And(landcover.lte(5))
            forest_area = pixel_area.updateMask(forest_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=feature.geometry(),
                scale=500,
                maxPixels=1e9,
                bestEffort=True
            )
            
            # Total area
            total_area = pixel_area.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=feature.geometry(),
                scale=500,
                maxPixels=1e9,
                bestEffort=True
            )
            
            # Shrublands (classes 6-7)
            shrub_mask = landcover.gte(6).And(landcover.lte(7))
            shrub_area = pixel_area.updateMask(shrub_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=feature.geometry(),
                scale=500,
                maxPixels=1e9,
                bestEffort=True
            )
            
            # Grasslands (class 10)
            grass_mask = landcover.eq(10)
            grass_area = pixel_area.updateMask(grass_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=feature.geometry(),
                scale=500,
                maxPixels=1e9,
                bestEffort=True
            )
            
            # Safe division function
            def safe_divide(numerator, denominator):
                return ee.Algorithms.If(
                    ee.Number(denominator).gt(0),
                    ee.Number(numerator).divide(ee.Number(denominator)),
                    0
                )
            
            return {
                'ndvi_mean': veg_stats.get('ndvi_mean', 0),
                'evi_mean': veg_stats.get('evi_mean', 0),
                'ndvi_seasonality': veg_stats.get('ndvi_std', 0),
                'forest_fraction': safe_divide(forest_area.get('area', 0), total_area.get('area', 1)),
                'shrub_fraction': safe_divide(shrub_area.get('area', 0), total_area.get('area', 1)),
                'grass_fraction': safe_divide(grass_area.get('area', 0), total_area.get('area', 1))
            }
            
        except Exception as e:
            print(f"Error in vegetation extraction: {e}")
            return {
                'ndvi_mean': 0,
                'evi_mean': 0,
                'ndvi_seasonality': 0,
                'forest_fraction': 0,
                'shrub_fraction': 0,
                'grass_fraction': 0
            }
    
    def extract_climate_variables(self, feature):
        """
        Extract climate variables
        """
        try:
            climate = self.datasets['climate']
            
            # Annual means
            precip_annual = climate.select('pr').sum()  # Total annual precipitation
            temp_mean = climate.select('tmmx').mean()   # Mean max temperature
            temp_min = climate.select('tmmn').mean()    # Mean min temperature
            vpd_mean = climate.select('vpd').mean()     # Vapor pressure deficit
            
            # Seasonality
            precip_cv = climate.select('pr').reduce(ee.Reducer.stdDev()) \
                .divide(climate.select('pr').mean())  # Precipitation coefficient of variation
            
            # Fire weather
            # Calculate fire season (May-October) averages
            fire_season = climate.filter(ee.Filter.calendarRange(5, 10, 'month'))
            fire_temp = fire_season.select('tmmx').mean()
            fire_vpd = fire_season.select('vpd').mean()
            fire_precip = fire_season.select('pr').sum()
            
            # Extract all stats with proper band names
            climate_img = ee.Image.cat([
                precip_annual.rename('precip_annual'),
                temp_mean.rename('temp_mean'),
                temp_min.rename('temp_min'),
                vpd_mean.rename('vpd_mean'),
                precip_cv.rename('precip_seasonality'),
                fire_temp.rename('fire_temp'),
                fire_vpd.rename('fire_vpd'),
                fire_precip.rename('fire_precip')
            ])
            
            stats = climate_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=feature.geometry(),
                scale=4000,
                maxPixels=1e9,
                bestEffort=True
            )
            
            # Return with default values for missing keys
            return {
                'precip_annual': stats.get('precip_annual', 0),
                'temp_mean': stats.get('temp_mean', 0),
                'temp_min': stats.get('temp_min', 0),
                'vpd_mean': stats.get('vpd_mean', 0),
                'precip_seasonality': stats.get('precip_seasonality', 0),
                'fire_temp': stats.get('fire_temp', 0),
                'fire_vpd': stats.get('fire_vpd', 0),
                'fire_precip': stats.get('fire_precip', 0)
            }
            
        except Exception as e:
            print(f"Error in climate extraction: {e}")
            return {
                'precip_annual': 0,
                'temp_mean': 0,
                'temp_min': 0,
                'vpd_mean': 0,
                'precip_seasonality': 0,
                'fire_temp': 0,
                'fire_vpd': 0,
                'fire_precip': 0
            }
    
    def extract_human_variables(self, feature):
        """
        Extract human influence variables
        """
        try:
            # Population density
            pop_stats = self.datasets['population'].reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=feature.geometry(),
                scale=100,
                maxPixels=1e9,
                bestEffort=True
            )
            
            # Distance to nearest road (simplified approach)
            try:
                # Get nearest road distance
                nearest_road = self.datasets['roads'].distance(1000).clip(feature.geometry())
                road_stats = nearest_road.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=feature.geometry(),
                    scale=100,
                    maxPixels=1e9,
                    bestEffort=True
                )
                road_dist = road_stats.get('distance', 1000)  # Default 1km if no roads
            except:
                road_dist = 1000  # Default distance if roads data fails
            
            return {
                'population_density': pop_stats.get('population', 0),
                'distance_to_road': road_dist
            }
            
        except Exception as e:
            print(f"Error in human variables extraction: {e}")
            return {
                'population_density': 0,
                'distance_to_road': 1000
            }
    
    def extract_forest_variables(self, feature):
        """
        Extract forest-specific variables
        """
        try:
            forest = self.datasets['forest']
            
            # Tree cover in 2000, forest loss, and gain
            forest_composite = ee.Image.cat([
                forest.select('treecover2000').rename('forest_cover_2000'),
                forest.select('loss').rename('forest_loss'),
                forest.select('gain').rename('forest_gain')
            ])
            
            # Calculate statistics
            stats = forest_composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=feature.geometry(),
                scale=30,
                maxPixels=1e9,
                bestEffort=True
            )
            
            return {
                'forest_cover_2000': stats.get('forest_cover_2000', 0),
                'forest_loss_fraction': stats.get('forest_loss', 0),
                'forest_gain_fraction': stats.get('forest_gain', 0)
            }
            
        except Exception as e:
            print(f"Error in forest variables extraction: {e}")
            return {
                'forest_cover_2000': 0,
                'forest_loss_fraction': 0,
                'forest_gain_fraction': 0
            }
    
    def extract_watershed_ecological_data(self, watershed_index):
        """
        Extract all ecological variables for a single watershed
        """
        # Get watershed
        watershed = self.watersheds.iloc[watershed_index]
        
        try:
            # Convert to EE feature with proper geometry handling
            geom = watershed.geometry
            
            # Simplify geometry to avoid complexity issues
            # Use a small tolerance (0.001 degrees, roughly 100m)
            geom = geom.simplify(tolerance=0.005, preserve_topology=True)
            
            # Handle different geometry types
            if geom.geom_type == 'MultiPolygon':
                # For MultiPolygon, use the largest polygon
                largest_poly = max(geom.geoms, key=lambda x: x.area)
                coords = [list(largest_poly.exterior.coords)]
            elif geom.geom_type == 'Polygon':
                # For Polygon, get exterior coordinates
                coords = [list(geom.exterior.coords)]
            else:
                # For other types, try to convert to polygon
                try:
                    if hasattr(geom, 'convex_hull'):
                        geom = geom.convex_hull
                        coords = [list(geom.exterior.coords)]
                    else:
                        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
                except:
                    # Fall back to bounding box
                    bounds = geom.bounds
                    coords = [[(bounds[0], bounds[1]), 
                              (bounds[2], bounds[1]),
                              (bounds[2], bounds[3]),
                              (bounds[0], bounds[3]),
                              (bounds[0], bounds[1])]]
            
            # Create EE geometry
            ee_polygon = ee.Geometry.Polygon(coords)
            ee_feature = ee.Feature(ee_polygon)
            
            # Extract all variable groups
            topo = self.extract_topographic_variables(ee_feature)
            veg = self.extract_vegetation_variables(ee_feature)
            climate = self.extract_climate_variables(ee_feature)
            human = self.extract_human_variables(ee_feature)
            forest = self.extract_forest_variables(ee_feature)
            
            # Combine all variables
            result = {'watershed_id': watershed_index}
            
            # Convert EE objects to Python values
            for var_dict in [topo, veg, climate, human, forest]:
                for key, value in var_dict.items():
                    if isinstance(value, ee.ComputedObject):
                        result[key] = value.getInfo()
                    else:
                        result[key] = value
            
            return result
            
        except Exception as e:
            print(f"Error extracting watershed {watershed_index}: {e}")
            # Return NA values instead of error to keep the dataset consistent
            result = {'watershed_id': watershed_index}
            # Add NA for all expected columns
            for col in ['elevation_mean', 'elevation_std', 'elevation_min', 'elevation_max',
                       'slope_mean', 'slope_std', 'aspect_mean', 'ruggedness',
                       'ndvi_mean', 'evi_mean', 'ndvi_seasonality',
                       'forest_fraction', 'shrub_fraction', 'grass_fraction',
                       'precip_annual', 'temp_mean', 'temp_min', 'vpd_mean',
                       'precip_seasonality', 'fire_temp', 'fire_vpd', 'fire_precip',
                       'population_density', 'distance_to_road',
                       'forest_cover_2000', 'forest_loss_fraction', 'forest_gain_fraction']:
                result[col] = np.nan
            return result
    
    def batch_extract_parallel(self, batch_size=100, n_jobs=10, checkpoint_dir='06_Ecological_Context/checkpoints'):
        """
        Extract ecological data for all watersheds in parallel batches with checkpointing
        """
        print("\n" + "="*60)
        print("BATCH EXTRACTION OF ECOLOGICAL VARIABLES")
        print("="*60)
        
        n_watersheds = len(self.watersheds)
        print(f"Extracting data for {n_watersheds} watersheds")
        print(f"Batch size: {batch_size}, Parallel jobs: {n_jobs}")
        
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_path / 'extraction_progress.parquet'
        
        # Check for existing progress
        results = []
        start_batch = 0
        failed_count = 0
        successful_count = 0
        
        if checkpoint_file.exists():
            print(f"\n📁 Found existing progress file: {checkpoint_file}")
            existing_df = pd.read_parquet(checkpoint_file)
            results = existing_df.reset_index().to_dict('records')
            start_batch = len(results) // batch_size
            successful_count = len([r for r in results if r.get('elevation_mean', 0) != 0])
            failed_count = len(results) - successful_count
            print(f"   Resuming from batch {start_batch + 1}")
            print(f"   Already processed: {len(results)} watersheds")
        
        total_batches = (n_watersheds + batch_size - 1) // batch_size
        
        for batch_num, batch_start in enumerate(range(start_batch * batch_size, n_watersheds, batch_size), start_batch + 1):
            batch_end = min(batch_start + batch_size, n_watersheds)
            batch_indices = range(batch_start, batch_end)
            
            # Skip if already processed
            if batch_num <= start_batch:
                continue
                
            print(f"\nProcessing batch {batch_num}/{total_batches}: "
                  f"watersheds {batch_start}-{batch_end-1}")
            
            # Parallel extraction within batch
            try:
                batch_results = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(self.extract_watershed_ecological_data)(idx)
                    for idx in tqdm(batch_indices, desc=f"Batch {batch_num}")
                )
                
                # Count successes and failures
                for result in batch_results:
                    if 'error' in result or result.get('elevation_mean') == 0:
                        failed_count += 1
                    else:
                        successful_count += 1
                
                results.extend(batch_results)
                
                # Save checkpoint every 10 batches
                if batch_num % 10 == 0:
                    temp_df = pd.DataFrame(results)
                    temp_df.set_index('watershed_id', inplace=True)
                    temp_df.to_parquet(checkpoint_file)
                    print(f"   💾 Checkpoint saved at batch {batch_num}")
                
            except Exception as e:
                print(f"⚠️  Batch {batch_num} failed: {e}")
                # Add NA results for failed batch
                for idx in batch_indices:
                    result = {'watershed_id': idx}
                    for col in ['elevation_mean', 'elevation_std', 'elevation_min', 'elevation_max',
                               'slope_mean', 'slope_std', 'aspect_mean', 'ruggedness',
                               'ndvi_mean', 'evi_mean', 'ndvi_seasonality',
                               'forest_fraction', 'shrub_fraction', 'grass_fraction',
                               'precip_annual', 'temp_mean', 'temp_min', 'vpd_mean',
                               'precip_seasonality', 'fire_temp', 'fire_vpd', 'fire_precip',
                               'population_density', 'distance_to_road',
                               'forest_cover_2000', 'forest_loss_fraction', 'forest_gain_fraction']:
                        result[col] = np.nan
                    results.append(result)
                    failed_count += 1
            
            # Sleep to avoid rate limits (except for last batch)
            if batch_end < n_watersheds:
                print(f"Progress: {successful_count + len(results) - failed_count}/{n_watersheds} successful, {failed_count} failed")
                print("Pausing 3 seconds to avoid GEE rate limits...")
                time.sleep(3)
        
        # Convert to DataFrame
        ecological_df = pd.DataFrame(results)
        
        # Remove error column if it exists
        if 'error' in ecological_df.columns:
            ecological_df = ecological_df.drop('error', axis=1)
        
        # Set watershed_id as index
        ecological_df.set_index('watershed_id', inplace=True)
        
        # Save final checkpoint
        ecological_df.to_parquet(checkpoint_file)
        print(f"\n💾 Final results saved to checkpoint: {checkpoint_file}")
        
        print(f"\n✓ Extraction complete: {len(ecological_df)} watersheds")
        print(f"  Variables extracted: {len(ecological_df.columns)}")
        print(f"  Successful extractions: {successful_count}")
        print(f"  Failed extractions: {failed_count}")
        
        if failed_count > 0:
            print(f"  ⚠️  {failed_count} watersheds have missing data (filled with NA)")
        
        return ecological_df
    
    def merge_with_fire_data(self, ecological_df):
        """
        Merge ecological variables with fire data
        """
        print("\n" + "="*60)
        print("MERGING ECOLOGICAL AND FIRE DATA")
        print("="*60)
        
        # Merge on index
        enhanced_watersheds = self.watersheds.merge(
            ecological_df,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        print(f"Enhanced dataset: {enhanced_watersheds.shape}")
        
        # Check for missing values
        eco_cols = ecological_df.columns
        missing = enhanced_watersheds[eco_cols].isnull().sum()
        
        if missing.sum() > 0:
            print("\nMissing values in ecological variables:")
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]}")
        
        return enhanced_watersheds
    
    def export_enhanced_data(self, enhanced_df, output_dir='06_Ecological_Context'):
        """
        Export enhanced dataset with ecological variables
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet (efficient)
        output_file = output_dir / 'watersheds_with_ecological_context.parquet'
        enhanced_df.to_parquet(output_file)
        
        # Save variable descriptions
        var_descriptions = {
            'elevation_mean': 'Mean elevation (meters)',
            'elevation_std': 'Elevation standard deviation',
            'slope_mean': 'Mean slope (degrees)',
            'aspect_mean': 'Mean aspect (degrees)',
            'ruggedness': 'Terrain ruggedness (elevation range)',
            'ndvi_mean': 'Mean NDVI (vegetation greenness)',
            'evi_mean': 'Mean EVI (enhanced vegetation index)',
            'ndvi_seasonality': 'NDVI seasonality (standard deviation)',
            'forest_fraction': 'Fraction of watershed forested',
            'shrub_fraction': 'Fraction of watershed shrubland',
            'grass_fraction': 'Fraction of watershed grassland',
            'precip_annual': 'Annual precipitation (mm)',
            'temp_mean': 'Mean maximum temperature (°C)',
            'temp_min': 'Mean minimum temperature (°C)',
            'vpd_mean': 'Mean vapor pressure deficit (kPa)',
            'precip_seasonality': 'Precipitation seasonality (CV)',
            'fire_temp': 'Fire season temperature (°C)',
            'fire_vpd': 'Fire season VPD (kPa)',
            'fire_precip': 'Fire season precipitation (mm)',
            'population_density': 'Population density (people/km²)',
            'distance_to_road': 'Distance to nearest road (m)',
            'forest_cover_2000': 'Forest cover in 2000 (%)',
            'forest_loss_fraction': 'Forest loss fraction',
            'forest_gain_fraction': 'Forest gain fraction'
        }
        
        with open(output_dir / 'variable_descriptions.json', 'w') as f:
            json.dump(var_descriptions, f, indent=2)
        
        print(f"\n✓ Enhanced data saved to {output_file}")
        
        return output_file


class SimpleEcologicalExtractor:
    """
    Simplified version for testing without full GEE extraction
    Uses publicly available data sources
    """
    
    def __init__(self, watersheds_gdf):
        self.watersheds = watersheds_gdf
        
    def generate_synthetic_ecological_data(self):
        """
        Generate synthetic ecological variables for testing
        Based on realistic distributions
        """
        print("\n" + "="*60)
        print("GENERATING SYNTHETIC ECOLOGICAL DATA")
        print("="*60)
        
        n = len(self.watersheds)
        np.random.seed(42)
        
        # Generate correlated ecological variables
        # Base variables
        lat = self.watersheds.geometry.centroid.y.values
        lon = self.watersheds.geometry.centroid.x.values
        
        # Elevation decreases with latitude (simplified)
        elevation = 2000 - 20 * lat + np.random.normal(0, 200, n)
        elevation = np.clip(elevation, 0, 4000)
        
        # Temperature increases with lower latitude
        temp_mean = 25 - 0.5 * lat + np.random.normal(0, 2, n)
        
        # Precipitation varies with longitude (simplified)
        precip_annual = 800 + 10 * lon + np.random.normal(0, 200, n)
        precip_annual = np.clip(precip_annual, 100, 3000)
        
        # Vegetation correlates with precipitation
        ndvi_mean = 0.3 + 0.0002 * precip_annual + np.random.normal(0, 0.1, n)
        ndvi_mean = np.clip(ndvi_mean, 0.1, 0.9)
        
        # Forest cover correlates with precipitation and elevation
        forest_fraction = 0.2 + 0.0001 * precip_annual + 0.0001 * elevation + np.random.normal(0, 0.1, n)
        forest_fraction = np.clip(forest_fraction, 0, 1)
        
        # Population inversely correlates with elevation
        population_density = 100 * np.exp(-elevation / 1000) + np.random.exponential(10, n)
        
        # Create DataFrame
        ecological_df = pd.DataFrame({
            'elevation_mean': elevation,
            'elevation_std': np.abs(np.random.normal(100, 50, n)),
            'slope_mean': 10 + np.random.exponential(5, n),
            'aspect_mean': np.random.uniform(0, 360, n),
            'ruggedness': np.abs(np.random.normal(500, 200, n)),
            'ndvi_mean': ndvi_mean,
            'evi_mean': ndvi_mean * 0.8 + np.random.normal(0, 0.05, n),
            'ndvi_seasonality': np.random.beta(2, 5, n) * 0.3,
            'forest_fraction': forest_fraction,
            'shrub_fraction': np.random.beta(2, 5, n) * (1 - forest_fraction),
            'grass_fraction': np.random.beta(2, 5, n) * (1 - forest_fraction),
            'precip_annual': precip_annual,
            'temp_mean': temp_mean,
            'temp_min': temp_mean - np.random.uniform(5, 15, n),
            'vpd_mean': 0.5 + np.random.exponential(0.3, n),
            'precip_seasonality': np.random.beta(2, 5, n),
            'fire_temp': temp_mean + np.random.normal(5, 2, n),
            'fire_vpd': 1.0 + np.random.exponential(0.5, n),
            'fire_precip': precip_annual * np.random.beta(2, 5, n),
            'population_density': population_density,
            'distance_to_road': np.random.exponential(1000, n),
            'forest_cover_2000': forest_fraction * 100,
            'forest_loss_fraction': np.random.beta(2, 10, n) * 0.2,
            'forest_gain_fraction': np.random.beta(2, 10, n) * 0.1
        })
        
        print(f"Generated {len(ecological_df.columns)} ecological variables")
        print(f"for {len(ecological_df)} watersheds")
        
        return ecological_df