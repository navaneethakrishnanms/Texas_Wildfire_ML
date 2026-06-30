import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import rasterio
from rasterio.transform import from_origin
from loguru import logger
import datetime
from typing import Dict, Any, List

class GeospatialSimulator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the simulator with configuration bounds.
        """
        self.config = config
        self.spatial = config["spatial"]
        self.temporal = config["temporal"]
        self.paths = config["paths"]
        
        # Spatial grid details
        self.crs = self.spatial["crs"]
        self.min_x = self.spatial["min_x"]
        self.max_x = self.spatial["max_x"]
        self.min_y = self.spatial["min_y"]
        self.max_y = self.spatial["max_y"]
        self.res = self.spatial["resolution"]
        
        self.width = int((self.max_x - self.min_x) / self.res)
        self.height = int((self.max_y - self.min_y) / self.res)
        
        # Geotransform: (x_min, res_x, 0, y_max, 0, -res_y)
        self.transform = from_origin(self.min_x, self.max_y, self.res, self.res)
        
        # Temporal range
        self.dates = pd.date_range(
            start=self.temporal["start_date"],
            end=self.temporal["end_date"],
            freq=self.temporal["frequency"]
        )
        
        # Setup directories
        self.raw_dir = Path(self.paths["raw_dir"])
        self.weather_dir = self.raw_dir / "weather_daily"
        self.modis_dir = self.raw_dir / "modis_daily"
        self.burned_dir = self.raw_dir / "burned_area"
        
        for d in [self.raw_dir, self.weather_dir, self.modis_dir, self.burned_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Set random seed for reproducibility
        np.random.seed(42)

    def write_raster(self, path: Path, data: np.ndarray, num_bands: int, band_names: List[str] = None, dtype: Any = rasterio.float32) -> None:
        """
        Utility to write a multi-band GeoTIFF.
        """
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=num_bands,
            dtype=dtype,
            crs=self.crs,
            transform=self.transform,
        ) as dst:
            if num_bands == 1:
                dst.write(data.astype(dtype), 1)
            else:
                for idx in range(num_bands):
                    dst.write(data[idx].astype(dtype), idx + 1)
            
            if band_names:
                dst.descriptions = tuple(band_names)

    def generate_static_dem(self) -> np.ndarray:
        """
        Generates a 2D elevation grid simulating hills in Texas.
        Elevations range from ~100m to ~600m.
        """
        logger.info("Simulating DEM (Elevation)...")
        # Create coordinates
        x = np.linspace(0, 10, self.width)
        y = np.linspace(0, 10, self.height)
        xx, yy = np.meshgrid(x, y)
        
        # Rolling topography using waves
        elevation = 250.0 + 120.0 * np.sin(xx / 1.5) * np.cos(yy / 2.0) \
                          + 60.0 * np.sin(xx * 2.0) \
                          + 30.0 * np.cos(yy * 3.0) \
                          + np.random.normal(0, 2.0, size=(self.height, self.width))
        
        dem_path = self.raw_dir / "dem.tif"
        self.write_raster(dem_path, elevation, 1, ["elevation"])
        logger.info(f"DEM written to {dem_path}")
        return elevation

    def generate_static_nlcd(self, elevation: np.ndarray) -> None:
        """
        Generates static NLCD classes based on topography.
        Classes: Forest (41), Grassland (71), Shrubland (52), Urban (21), Agriculture (82)
        """
        logger.info("Simulating Land Cover (NLCD)...")
        nlcd = np.zeros_like(elevation, dtype=np.int32)
        
        # Rules:
        # Urban (21) near a simulated city spot in the center-left
        # Agriculture (82) in flat low valleys
        # Forest (41) on higher ridges
        # Grassland (71) / Shrubland (52) everywhere else
        
        nlcd[:] = 71  # default Grassland
        
        # Elevation based forest
        nlcd[elevation > 350.0] = 41  # Forest on high terrain
        nlcd[elevation > 420.0] = 52  # Shrubland on steep ridges
        
        # Valley agriculture
        nlcd[elevation < 200.0] = 82  # Agriculture
        
        # Simulating city center at cell (40, 30)
        for r in range(self.height):
            for c in range(self.width):
                dist = np.sqrt((r - 40)**2 + (c - 30)**2)
                if dist < 12:
                    if np.random.rand() > (dist / 12.0):
                        nlcd[r, c] = 21  # Urban/Developed
                        
        nlcd_path = self.raw_dir / "nlcd.tif"
        self.write_raster(nlcd_path, nlcd, 1, ["land_cover"], dtype=rasterio.int32)
        logger.info(f"NLCD Land Cover written to {nlcd_path}")

    def generate_human_infrastructure(self) -> None:
        """
        Generates mock road networks and powerline shapes as vector lines.
        """
        logger.info("Simulating roads and powerlines...")
        # Let's create lines cutting across our bounding box
        # Min X, Max X, Min Y, Max Y
        
        # Roads: 1 major highway running diagonal, and 2 local roads
        roads_geom = [
            LineString([(self.min_x, self.min_y), (self.max_x, self.max_y)]), # Highway
            LineString([(self.min_x + 10000, self.max_y), (self.max_x - 10000, self.min_y)]),
            LineString([(self.min_x, self.min_y + 15000), (self.max_x, self.min_y + 20000)])
        ]
        roads_df = gpd.GeoDataFrame(
            {"road_id": [1, 2, 3], "type": ["highway", "local", "local"]},
            geometry=roads_geom,
            crs=self.crs
        )
        roads_path = self.raw_dir / "roads.geojson"
        roads_df.to_file(roads_path, driver="GeoJSON")
        logger.info(f"Roads written to {roads_path}")
        
        # Powerlines: 2 main corridors
        powerlines_geom = [
            LineString([(self.min_x, self.max_y - 10000), (self.max_x, self.min_y + 10000)]),
            LineString([(self.min_x + 25000, self.min_y), (self.min_x + 25000, self.max_y)])
        ]
        powerlines_df = gpd.GeoDataFrame(
            {"line_id": [1, 2], "voltage_kv": [138, 345]},
            geometry=powerlines_geom,
            crs=self.crs
        )
        powerlines_path = self.raw_dir / "powerlines.geojson"
        powerlines_df.to_file(powerlines_path, driver="GeoJSON")
        logger.info(f"Powerlines written to {powerlines_path}")

    def generate_weather_and_modis(self, elevation: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates daily weather rasters and daily MODIS rasters.
        Also tracks potential fire ignition zones based on physics.
        Returns a list of daily active fire points for later export.
        """
        logger.info("Simulating daily weather and satellite time series...")
        
        # We model a summer season (June to August)
        # Weather variables: Temp (deg C), RH (%), Wind Speed (m/s), Wind Dir (deg), Precip Prob (0-1)
        # Satellite: NDVI (0-1), EVI (0-1), LST (deg C)
        
        fire_detections = []
        
        for day_idx, date in enumerate(self.dates):
            date_str = date.strftime("%Y%m%d")
            
            # 1. Base regional values for the day (temporal trend with weather fronts)
            # High temperatures in July/August
            base_temp = 32.0 + 6.0 * np.sin(2 * np.pi * (day_idx + 15) / 365) + np.random.normal(0, 1.5)
            # Inverse relationship between temperature and humidity
            base_rh = 55.0 - 20.0 * np.sin(2 * np.pi * (day_idx + 15) / 365) + np.random.normal(0, 5.0)
            base_rh = np.clip(base_rh, 15.0, 95.0)
            
            # Wind speed fluctuates (ranges from 1.5 to 8 m/s)
            base_wind_speed = 3.5 + 2.0 * np.sin(2 * np.pi * day_idx / 12) + np.random.normal(0, 1.0)
            base_wind_speed = np.clip(base_wind_speed, 0.5, 12.0)
            
            base_wind_dir = (180.0 + 90.0 * np.cos(2 * np.pi * day_idx / 30) + np.random.normal(0, 20.0)) % 360.0
            
            # Rain events are rare in summer Texas
            if np.random.rand() > 0.88:
                base_precip_prob = np.random.uniform(0.4, 0.9)
                base_rh += 20.0
                base_temp -= 5.0
            else:
                base_precip_prob = np.random.uniform(0.0, 0.15)
                
            # Apply spatial variation across the grid based on elevation/topography
            # Temp decreases with elevation (-6.5 C per 1000m)
            temp_grid = base_temp - 0.0065 * (elevation - 250.0) + np.random.normal(0, 0.2, size=elevation.shape)
            # RH increases with elevation slightly
            rh_grid = base_rh + 0.02 * (elevation - 250.0) + np.random.normal(0, 1.0, size=elevation.shape)
            rh_grid = np.clip(rh_grid, 10.0, 100.0)
            
            # Wind speed increases on higher ridges
            wind_speed_grid = base_wind_speed * (1.0 + 0.001 * (elevation - 250.0)) + np.random.normal(0, 0.2, size=elevation.shape)
            wind_speed_grid = np.clip(wind_speed_grid, 0.0, 25.0)
            
            wind_dir_grid = np.full(elevation.shape, base_wind_dir) + np.random.normal(0, 5.0, size=elevation.shape)
            wind_dir_grid = wind_dir_grid % 360.0
            
            precip_grid = np.full(elevation.shape, base_precip_prob) + np.random.normal(0, 0.02, size=elevation.shape)
            precip_grid = np.clip(precip_grid, 0.0, 1.0)
            
            weather_data = np.stack([temp_grid, rh_grid, wind_speed_grid, wind_dir_grid, precip_grid])
            weather_path = self.weather_dir / f"weather_{date_str}.tif"
            self.write_raster(
                weather_path, 
                weather_data, 
                5, 
                ["temperature", "relative_humidity", "wind_speed", "wind_direction", "precip_prob"]
            )
            
            # 2. MODIS data (NDVI, EVI, LST)
            # NDVI decays over the summer as it dries out
            base_ndvi = 0.55 - 0.15 * (day_idx / len(self.dates)) + np.random.normal(0, 0.02)
            # Forests stay greener (NDVI 0.65), urban is low (NDVI 0.2), grass is sensitive
            ndvi_grid = np.full(elevation.shape, base_ndvi)
            ndvi_grid[elevation > 350.0] += 0.15  # Forest zones
            ndvi_grid[elevation < 200.0] += 0.05  # Agriculture zones
            # Urban low greenness
            # We can mock this simply:
            ndvi_grid = np.clip(ndvi_grid + np.random.normal(0, 0.03, size=elevation.shape), 0.05, 0.9)
            
            evi_grid = ndvi_grid * 0.7 + np.random.normal(0, 0.01, size=elevation.shape)
            evi_grid = np.clip(evi_grid, 0.0, 0.7)
            
            # LST is highly correlated with air temperature and inversely with NDVI (dry areas are hotter)
            lst_grid = temp_grid * 1.25 - 10.0 * (ndvi_grid - 0.4) + np.random.normal(0, 0.5, size=elevation.shape)
            
            # Introduce simulated cloud cover (NaNs/Missing values) on 5% of pixels
            # In a real scenario satellite data has cloud contamination
            cloud_mask = np.random.rand(*elevation.shape) < 0.05
            ndvi_grid[cloud_mask] = np.nan
            evi_grid[cloud_mask] = np.nan
            lst_grid[cloud_mask] = np.nan
            
            modis_data = np.stack([ndvi_grid, evi_grid, lst_grid])
            modis_path = self.modis_dir / f"modis_{date_str}.tif"
            self.write_raster(modis_path, modis_data, 3, ["ndvi", "evi", "lst"])
            
            # 3. Fire Occurrence Simulation (MODIS / VIIRS Active Fire points)
            # Fire ignition probability increases with:
            # - High LST / temperature (> 34 deg C)
            # - Low Relative Humidity (< 25%)
            # - Low greenness (NDVI < 0.4)
            # - High winds (> 5 m/s)
            # - Low precip probability
            # Let's compute a fire risk index to decide where fires ignite
            
            # Clean nan values for risk index calculation
            ndvi_clean = np.nan_to_num(ndvi_grid, nan=0.3)
            rh_clean = np.nan_to_num(rh_grid, nan=40.0)
            
            risk = (
                (temp_grid - 25.0) / 15.0 * 2.0 +
                (70.0 - rh_clean) / 50.0 * 2.5 +
                (0.6 - ndvi_clean) / 0.5 * 2.0 +
                (wind_speed_grid / 6.0) * 1.5
            )
            risk = np.clip(risk - precip_grid * 4.0, 0.0, None)
            
            # Highly conditional ignition: Only if risk exceeds threshold
            ignition_prob = 1.0 / (1.0 + np.exp(-(risk - 7.5)))  # Sigmoid risk
            
            # Burned area raster (starts as all zeros)
            burned_raster = np.zeros(elevation.shape, dtype=np.int32)
            
            # Fire event extraction
            rows, cols = np.where(ignition_prob > 0.70)
            for r, c in zip(rows, cols):
                # Sample fire points based on probability
                if np.random.rand() < (ignition_prob[r, c] * 0.008):  # Calibrated for scarce fires
                    # Convert grid coordinates (r, c) to projected spatial coordinates (x, y)
                    # Note: row r corresponds to y = max_y - r * res
                    # col c corresponds to x = min_x + c * res
                    x_coord = self.min_x + c * self.res + self.res/2.0
                    y_coord = self.max_y - r * self.res - self.res/2.0
                    
                    # Convert to Lat/Lon for NASA Active Fire csv format
                    # Since this is UTM 14N (EPSG:26914), we can approximate WGS84 coordinates.
                    # In pyproj:
                    # Let's project UTM to Lat/Lon
                    # We will create a local projection transformer to convert EPSG:26914 to EPSG:4326
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(x_coord, y_coord)
                    
                    frp = float(10.0 + 40.0 * risk[r, c] + np.random.normal(0, 5.0))
                    confidence = int(np.clip(50 + int(risk[r, c]*5), 30, 100))
                    
                    fire_detections.append({
                        "latitude": lat,
                        "longitude": lon,
                        "acq_date": date.strftime("%Y-%m-%d"),
                        "acq_time": "1830",  # simulated daily pass time in UTC
                        "confidence": confidence,
                        "frp": frp,
                        "satellite": "VIIRS",
                        "x_projected": x_coord,
                        "y_projected": y_coord
                    })
                    
                    # Mark burned area pixel
                    burned_raster[r, c] = 1
                    
            # Write daily burned area raster
            burned_path = self.burned_dir / f"burned_area_{date_str}.tif"
            self.write_raster(burned_path, burned_raster, 1, ["burned_area"], dtype=rasterio.int32)
            
        logger.info(f"Daily variables simulated. Generated {len(fire_detections)} active fire points.")
        return fire_detections

    def generate_drought_data(self) -> None:
        """
        Generates daily drought timeseries (USDM and PDSI).
        """
        logger.info("Simulating drought index timeseries...")
        # USDM category ranges from 0 (D0-Abnormally Dry) to 4 (D4-Exceptional Drought)
        # PDSI ranges from -10 (extreme drought) to +10 (extreme wet), with -3 to -4 being severe drought.
        
        # Let's generate a daily sequence where drought increases over the summer (common in Texas)
        data = []
        current_usdm = 1.0
        current_pdsi = -1.5
        
        for idx, date in enumerate(self.dates):
            # Slow random walk towards worse drought
            current_usdm = np.clip(current_usdm + np.random.choice([-1, 0, 1], p=[0.05, 0.85, 0.10]), 0, 4)
            current_pdsi = np.clip(current_pdsi - 0.02 + np.random.normal(0, 0.05), -6.0, 2.0)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "usdm_severity": int(current_usdm),
                "pdsi": float(current_pdsi)
            })
            
        df = pd.DataFrame(data)
        drought_path = self.raw_dir / "drought.csv"
        df.to_csv(drought_path, index=False)
        logger.info(f"Drought index data saved to {drought_path}")

    def generate_historical_fires(self, dem: np.ndarray) -> None:
        """
        Generates a point list of historical fire ignitions from the last 5 years.
        Used as historical fire records.
        """
        logger.info("Simulating historical wildfire records...")
        # Historical fires over the last 5 years (2020 - 2024)
        hist_records = []
        
        # Select 50 random historical ignition points, biased towards high risk areas
        for i in range(50):
            r = np.random.randint(0, self.height)
            c = np.random.randint(0, self.width)
            
            x_coord = self.min_x + c * self.res + self.res/2.0
            y_coord = self.max_y - r * self.res - self.res/2.0
            
            # Random year, size, cause
            year = np.random.choice([2020, 2021, 2022, 2023, 2024])
            month = np.random.choice([6, 7, 8, 9])
            day = np.random.randint(1, 29)
            date_str = f"{year}-{month:02d}-{day:02d}"
            
            acres = float(np.exp(np.random.normal(3, 1)))  # lognormal fire size
            cause = np.random.choice(["Lightning", "Campfire", "Debris Burning", "Powerline", "Equipment Use"])
            
            hist_records.append({
                "fire_id": f"TX-HIST-{2020 + i}",
                "ignition_date": date_str,
                "acres_burned": acres,
                "cause": cause,
                "geometry": Point(x_coord, y_coord)
            })
            
        df = gpd.GeoDataFrame(hist_records, crs=self.crs)
        hist_path = self.raw_dir / "historical_fires.geojson"
        df.to_file(hist_path, driver="GeoJSON")
        logger.info(f"Historical fires written to {hist_path}")

    def run(self) -> None:
        """
        Executes all simulation steps.
        """
        logger.info("--- STARTING SPATIOTEMPORAL DATA SIMULATION ---")
        dem = self.generate_static_dem()
        self.generate_static_nlcd(dem)
        self.generate_human_infrastructure()
        self.generate_drought_data()
        self.generate_historical_fires(dem)
        
        # Daily simulation and active fires extraction
        fire_points = self.generate_weather_and_modis(dem)
        
        # Save active fire points
        active_fire_df = pd.DataFrame(fire_points)
        active_fire_path = self.raw_dir / "nasa_active_fire.csv"
        active_fire_df.to_csv(active_fire_path, index=False)
        logger.info(f"NASA MODIS/VIIRS Active Fire points saved to {active_fire_path}")
        logger.info("--- DATA SIMULATION COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    import yaml
    # Test execution
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    sim = GeospatialSimulator(cfg)
    sim.run()
