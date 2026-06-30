import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from loguru import logger
from tqdm import tqdm
from typing import Dict, Any, Tuple

class GeospatialHarmonizer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Geospatial Harmonizer.
        """
        self.config = config
        self.spatial = config["spatial"]
        self.temporal = config["temporal"]
        self.paths = config["paths"]
        
        # Grid parameters
        self.crs = self.spatial["crs"]
        self.min_x = self.spatial["min_x"]
        self.max_x = self.spatial["max_x"]
        self.min_y = self.spatial["min_y"]
        self.max_y = self.spatial["max_y"]
        self.res = self.spatial["resolution"]
        
        self.width = int((self.max_x - self.min_x) / self.res)
        self.height = int((self.max_y - self.min_y) / self.res)
        self.transform = rasterio.transform.from_origin(self.min_x, self.max_y, self.res, self.res)
        
        # Bounding box polygon for vector clipping
        self.bbox_geom = rasterio.coords.BoundingBox(self.min_x, self.min_y, self.max_x, self.max_y)
        
        # Set up folders
        self.raw_dir = Path(self.paths["raw_dir"])
        self.processed_dir = Path(self.paths["processed_dir"])
        self.harmonized_dir = self.processed_dir / "daily_harmonized"
        self.harmonized_dir.mkdir(parents=True, exist_ok=True)
        
        self.dates = pd.date_range(
            start=self.temporal["start_date"],
            end=self.temporal["end_date"],
            freq=self.temporal["frequency"]
        )

    def align_raster(self, src_path: Path, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
        """
        Reprojects and resamples any input raster to match the target CRS, resolution, and bounds.
        """
        dest_data = np.zeros((self.height, self.width), dtype=np.float32)
        
        with rasterio.open(src_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dest_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.transform,
                dst_crs=self.crs,
                resampling=resampling
            )
        return dest_data

    def align_multiband_raster(self, src_path: Path, num_bands: int, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
        """
        Reprojects and resamples a multi-band raster to match the target grid.
        """
        dest_data = np.zeros((num_bands, self.height, self.width), dtype=np.float32)
        
        with rasterio.open(src_path) as src:
            for band_idx in range(num_bands):
                reproject(
                    source=rasterio.band(src, band_idx + 1),
                    destination=dest_data[band_idx],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=self.transform,
                    dst_crs=self.crs,
                    resampling=resampling
                )
        return dest_data

    def compute_proximity_grid(self, geojson_path: Path) -> np.ndarray:
        """
        Rasterizes a vector layer and calculates Euclidean distance transform in meters.
        Returns a 2D array where each pixel represents distance to the nearest feature.
        """
        if not geojson_path.exists():
            logger.error(f"Vector file {geojson_path} does not exist. Returning zeros.")
            return np.zeros((self.height, self.width), dtype=np.float32)
            
        gdf = gpd.read_file(geojson_path)
        # Ensure correct CRS projection
        if gdf.crs != self.crs:
            gdf = gdf.to_crs(self.crs)
            
        # Clip geometries to bounding box to speed up rasterization
        # Creating a bounding box geometry
        from shapely.geometry import box
        bbox = box(self.min_x, self.min_y, self.max_x, self.max_y)
        gdf = gpd.clip(gdf, bbox)
        
        if len(gdf) == 0:
            logger.warning(f"No geometries found inside bounding box for {geojson_path}. Returning maximum distance.")
            return np.full((self.height, self.width), 100000.0, dtype=np.float32)

        # Rasterize: set 1 on features, 0 on background
        shapes = [(geom, 1) for geom in gdf.geometry]
        mask = rasterize(
            shapes=shapes,
            out_shape=(self.height, self.width),
            transform=self.transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        
        # Calculate Euclidean Distance: distance_transform_edt calculates distance to nearest 0.
        # Since we want distance to nearest 1 (roads/powerlines), we pass (mask == 0).
        # It returns distance in grid indices, so we multiply by resolution to get meters.
        dist_indices = distance_transform_edt(mask == 0)
        dist_meters = dist_indices * self.res
        
        return dist_meters.astype(np.float32)

    def impute_cloud_gaps(self, modis_series: np.ndarray) -> np.ndarray:
        """
        Imputes cloud contamination (NaNs) in satellite datasets.
        Uses temporal forward-fill, then backward-fill, and falls back to spatial median.
        modis_series shape: (num_days, num_bands, height, width)
        """
        logger.info("Imputing cloud gaps in MODIS series...")
        num_days, num_bands, h, w = modis_series.shape
        imputed = modis_series.copy()
        
        for b in range(num_bands):
            # For each pixel, do a temporal fill
            for r in range(h):
                for c in range(w):
                    pixel_series = imputed[:, b, r, c]
                    if np.isnan(pixel_series).any():
                        # Convert to pandas series for easy ffill/bfill
                        s = pd.Series(pixel_series)
                        s = s.ffill().bfill()
                        # If still NaN (meaning the entire summer is cloudy for this pixel, which is rare)
                        if s.isnull().any():
                            s = s.fillna(0.0) # default fallback
                        imputed[:, b, r, c] = s.values
                        
        # Double check if any NaNs remain globally and fill with median
        for b in range(num_bands):
            band_slice = imputed[:, b]
            if np.isnan(band_slice).any():
                median_val = np.nanmedian(band_slice)
                nan_mask = np.isnan(band_slice)
                band_slice[nan_mask] = median_val if not np.isnan(median_val) else 0.0
                imputed[:, b] = band_slice
                
        return imputed

    def run(self) -> None:
        """
        Executes the harmonization pipeline.
        Reads all raw files, reprojects them, rasterizes vectors, fills missing values,
        and saves daily multi-band rasters.
        """
        logger.info("--- STARTING GEOSPATIAL HARMONIZATION ---")
        
        # 1. Align Static Topography Data
        dem_path = self.raw_dir / "dem.tif"
        logger.info("Aligning DEM terrain raster...")
        dem_grid = self.align_raster(dem_path, Resampling.bilinear)
        
        nlcd_path = self.raw_dir / "nlcd.tif"
        logger.info("Aligning NLCD land cover raster (Nearest Neighbor)...")
        nlcd_grid = self.align_raster(nlcd_path, Resampling.nearest)
        
        # 2. Vector distance grids
        logger.info("Calculating road proximity grid...")
        roads_path = self.raw_dir / "roads.geojson"
        dist_roads = self.compute_proximity_grid(roads_path)
        
        logger.info("Calculating powerline proximity grid...")
        powerlines_path = self.raw_dir / "powerlines.geojson"
        dist_powerlines = self.compute_proximity_grid(powerlines_path)
        
        # 3. Read Drought Data timeseries
        logger.info("Parsing drought data timeseries...")
        drought_df = pd.read_csv(self.raw_dir / "drought.csv")
        drought_df["date"] = pd.to_datetime(drought_df["date"])
        drought_df.set_index("date", inplace=True)
        
        # 4. Load all daily MODIS datasets first to do joint cloud imputation
        logger.info("Reading satellite (MODIS) data series for cloud imputation...")
        modis_raw_stack = []
        for date in self.dates:
            date_str = date.strftime("%Y%m%d")
            modis_path = self.raw_dir / "modis_daily" / f"modis_{date_str}.tif"
            modis_grid = self.align_multiband_raster(modis_path, 3, Resampling.bilinear)
            modis_raw_stack.append(modis_grid)
            
        # Shape: (num_days, 3, height, width)
        modis_raw_stack = np.array(modis_raw_stack)
        modis_imputed = self.impute_cloud_gaps(modis_raw_stack)
        
        # 5. Process daily grids and save
        logger.info("Compiling daily harmonized multi-band rasters...")
        
        band_names = [
            "elevation", "land_cover", "dist_to_roads", "dist_to_powerlines",
            "pdsi", "usdm_severity", "temperature", "relative_humidity",
            "wind_speed", "wind_direction", "precip_prob",
            "ndvi", "evi", "lst", "burned_area"
        ]
        
        # Loop through each day and build stack
        for day_idx, date in enumerate(tqdm(self.dates, desc="Harmonizing daily slices")):
            date_str = date.strftime("%Y%m%d")
            
            # Weather variables (5 bands)
            weather_path = self.raw_dir / "weather_daily" / f"weather_{date_str}.tif"
            weather_grid = self.align_multiband_raster(weather_path, 5, Resampling.bilinear)
            
            # Drought values for the day
            row_drought = drought_df.loc[date]
            pdsi_val = row_drought["pdsi"]
            usdm_val = row_drought["usdm_severity"]
            
            pdsi_grid = np.full((self.height, self.width), pdsi_val, dtype=np.float32)
            usdm_grid = np.full((self.height, self.width), usdm_val, dtype=np.float32)
            
            # MODIS imputed values for the day (NDVI, EVI, LST)
            ndvi_grid = modis_imputed[day_idx, 0]
            evi_grid = modis_imputed[day_idx, 1]
            lst_grid = modis_imputed[day_idx, 2]
            
            # Burned Area (yesterday or current day check)
            burned_path = self.raw_dir / "burned_area" / f"burned_area_{date_str}.tif"
            burned_grid = self.align_raster(burned_path, Resampling.nearest)
            
            # Construct multi-band data
            daily_stack = np.stack([
                dem_grid,            # Band 1
                nlcd_grid,           # Band 2
                dist_roads,          # Band 3
                dist_powerlines,     # Band 4
                pdsi_grid,           # Band 5
                usdm_grid,           # Band 6
                weather_grid[0],     # Band 7: Temp
                weather_grid[1],     # Band 8: RH
                weather_grid[2],     # Band 9: Wind Speed
                weather_grid[3],     # Band 10: Wind Dir
                weather_grid[4],     # Band 11: Precip Prob
                ndvi_grid,           # Band 12: NDVI
                evi_grid,            # Band 13: EVI
                lst_grid,            # Band 14: LST
                burned_grid          # Band 15: Burned Area
            ])
            
            # Write to processed/daily_harmonized
            output_path = self.harmonized_dir / f"harmonized_{date_str}.tif"
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=self.height,
                width=self.width,
                count=len(band_names),
                dtype=rasterio.float32,
                crs=self.crs,
                transform=self.transform
            ) as dst:
                for idx in range(len(band_names)):
                    dst.write(daily_stack[idx], idx + 1)
                dst.descriptions = tuple(band_names)
                
        logger.info("--- GEOSPATIAL HARMONIZATION COMPLETED ---")

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    harmonizer = GeospatialHarmonizer(cfg)
    harmonizer.run()
