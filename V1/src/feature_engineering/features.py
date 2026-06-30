import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from tqdm import tqdm
from typing import Dict, Any, Tuple, List

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spatial = config["spatial"]
        self.temporal = config["temporal"]
        self.paths = config["paths"]

        # Spatial grid parameters needed for coordinate generation
        self.res = self.spatial["resolution"]
        self.min_x = self.spatial["min_x"]
        self.max_x = self.spatial["max_x"]
        self.min_y = self.spatial["min_y"]
        self.max_y = self.spatial["max_y"]
        self.width = int((self.max_x - self.min_x) / self.res)
        self.height = int((self.max_y - self.min_y) / self.res)

        self.raw_dir = Path(self.paths["raw_dir"])
        self.processed_dir = Path(self.paths["processed_dir"])
        self.harmonized_dir = self.processed_dir / "daily_harmonized"

        # Directory for intermediate daily features
        self.features_daily_dir = self.processed_dir / "daily_features"
        self.features_daily_dir.mkdir(parents=True, exist_ok=True)

        self.dates = pd.date_range(
            start=self.temporal["start_date"],
            end=self.temporal["end_date"],
            freq=self.temporal["frequency"]
        )

        # We start extracting features from day 3 to accommodate 3-day weather lags
        self.prediction_dates = self.dates[2:]

    def compute_terrain_derivatives(self, elevation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes slope (degrees), aspect_sin, aspect_cos, and Terrain Ruggedness Index (TRI).
        Uses a vectorized 8-neighbor elevation shift.
        """
        h, w = elevation.shape
        diffs_sq_sum = np.zeros_like(elevation, dtype=np.float32)
        valid_neighbors = np.zeros_like(elevation, dtype=np.float32)
        
        # Shift in 8 directions to calculate TRI
        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            shifted = np.roll(np.roll(elevation, dr, axis=0), dc, axis=1)
            
            # Mask out invalid boundary cells
            mask = np.ones((h, w), dtype=bool)
            if dr > 0: mask[:dr, :] = False
            elif dr < 0: mask[dr:, :] = False
            if dc > 0: mask[:, :dc] = False
            elif dc < 0: mask[:, dc:] = False
            
            diff = np.zeros_like(elevation)
            diff[mask] = shifted[mask] - elevation[mask]
            diffs_sq_sum += np.where(mask, diff**2, 0.0)
            valid_neighbors += mask.astype(float)
            
        tri = np.sqrt(diffs_sq_sum / np.clip(valid_neighbors, 1, None))
        
        # Gradients for slope/aspect
        dy, dx = np.gradient(elevation, self.res)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        aspect_rad = np.arctan2(dy, -dx)
        aspect_sin = np.sin(aspect_rad)
        aspect_cos = np.cos(aspect_rad)
        
        return slope_deg.astype(np.float32), aspect_sin.astype(np.float32), aspect_cos.astype(np.float32), tri.astype(np.float32)

    def compute_vpd(self, temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """
        Computes Vapor Pressure Deficit (VPD) in kPa.
        """
        svp = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
        avp = svp * (rh / 100.0)
        vpd = svp - avp
        return np.clip(vpd, 0.0, None)

    def compute_ffwi(self, temp_c: np.ndarray, rh: np.ndarray, wind_m_s: np.ndarray) -> np.ndarray:
        """
        Computes Fosberg Fire Weather Index (FFWI).
        """
        temp_f = temp_c * 1.8 + 32.0
        wind_mph = wind_m_s * 2.23694
        
        emc = np.zeros_like(temp_c)
        
        mask1 = rh < 10.0
        mask2 = (rh >= 10.0) & (rh <= 50.0)
        mask3 = rh > 50.0
        
        if mask1.any():
            emc[mask1] = 0.03229 + 0.281073 * rh[mask1] - 0.000578 * rh[mask1] * temp_f[mask1]
        if mask2.any():
            emc[mask2] = 2.22749 + 0.160107 * rh[mask2] - 0.014784 * temp_f[mask2]
        if mask3.any():
            emc[mask3] = 21.0606 + 0.005531 * (rh[mask3]**2) - 0.100276 * rh[mask3] - 0.001436 * rh[mask3] * temp_f[mask3]
            
        m = emc
        
        m_30 = m / 30.0
        n = 1.0 - 2.0 * m_30 + 1.5 * (m_30 ** 2) - 0.5 * (m_30 ** 3)
        n = np.where(m > 30.0, 0.0, n)
        
        ffwi = n * np.sqrt(1.0 + wind_mph**2) / 0.3002
        return np.clip(ffwi, 0.0, 100.0)

    def read_harmonized_bands(self, date_str: str) -> Dict[str, np.ndarray]:
        """
        Reads bands of the daily harmonized geotiff.
        """
        path = self.harmonized_dir / f"harmonized_{date_str}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Harmonized raster not found for: {date_str}")
            
        with rasterio.open(path) as src:
            data = src.read()
            descriptions = src.descriptions
            
        bands = {}
        for idx, desc in enumerate(descriptions):
            bands[desc] = data[idx]
            
        return bands

    def build_daily_features(self, date_idx: int) -> pd.DataFrame:
        """
        Calculates all features for a specific date (date_idx corresponds to self.dates).
        """
        date = self.dates[date_idx]
        date_str = date.strftime("%Y%m%d")
        
        # Load current day harmonized data
        curr_bands = self.read_harmonized_bands(date_str)
        
        # 1. Terrain derivatives (Slope, Aspect, TRI) from elevation
        slope, aspect_sin, aspect_cos, tri = self.compute_terrain_derivatives(curr_bands["elevation"])
        
        # 2. Fire Indices (VPD, HDW, FFWI)
        vpd = self.compute_vpd(curr_bands["temperature"], curr_bands["relative_humidity"])
        hdw = curr_bands["wind_speed"] * vpd
        ffwi = self.compute_ffwi(curr_bands["temperature"], curr_bands["relative_humidity"], curr_bands["wind_speed"])
        
        # 3. Weather Antecedent Lags (3-day history)
        # Load days t-1 and t-2
        prev1_str = self.dates[date_idx - 1].strftime("%Y%m%d")
        prev2_str = self.dates[date_idx - 2].strftime("%Y%m%d")
        
        prev1_bands = self.read_harmonized_bands(prev1_str)
        prev2_bands = self.read_harmonized_bands(prev2_str)
        
        # Temperature 3-day max
        temp_3d = np.stack([curr_bands["temperature"], prev1_bands["temperature"], prev2_bands["temperature"]])
        temp_max_3d = np.max(temp_3d, axis=0)
        
        # Humidity 3-day min
        rh_3d = np.stack([curr_bands["relative_humidity"], prev1_bands["relative_humidity"], prev2_bands["relative_humidity"]])
        rh_min_3d = np.min(rh_3d, axis=0)
        
        # Wind speed 3-day mean
        wind_3d = np.stack([curr_bands["wind_speed"], prev1_bands["wind_speed"], prev2_bands["wind_speed"]])
        wind_mean_3d = np.mean(wind_3d, axis=0)
        
        # 4. Land cover one-hot encoding
        lc = curr_bands["land_cover"].astype(np.int32)
        lc_urban = (lc == 21).astype(np.float32)
        lc_forest = (lc == 41).astype(np.float32)
        lc_shrubland = (lc == 52).astype(np.float32)
        lc_grassland = (lc == 71).astype(np.float32)
        lc_agriculture = (lc == 82).astype(np.float32)
        
        # 5. Extract dimensions and coordinates
        h, w = curr_bands["elevation"].shape
        
        # Get coordinates for the grid cells
        # Rows and cols meshgrid
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        
        # Spatial coordinate transformation
        x_proj = self.min_x + cols * self.res + self.res/2.0
        y_proj = self.max_y - rows * self.res - self.res/2.0
        
        # Construct flat features dictionary
        df_dict = {
            "date": date.strftime("%Y-%m-%d"),
            "row": rows.flatten(),
            "col": cols.flatten(),
            "x_projected": x_proj.flatten(),
            "y_projected": y_proj.flatten(),
            "cell_id": [f"cell_{r}_{c}" for r, c in zip(rows.flatten(), cols.flatten())],
            
            # Terrain
            "elevation": curr_bands["elevation"].flatten(),
            "slope": slope.flatten(),
            "aspect_sin": aspect_sin.flatten(),
            "aspect_cos": aspect_cos.flatten(),
            "tri": tri.flatten(),
            
            # Vegetation
            "ndvi": curr_bands["ndvi"].flatten(),
            "evi": curr_bands["evi"].flatten(),
            "lst": curr_bands["lst"].flatten(),
            
            # Raw Weather
            "temperature": curr_bands["temperature"].flatten(),
            "relative_humidity": curr_bands["relative_humidity"].flatten(),
            "wind_speed": curr_bands["wind_speed"].flatten(),
            "wind_direction": curr_bands["wind_direction"].flatten(),
            "precip_prob": curr_bands["precip_prob"].flatten(),
            
            # Weather Lags
            "temp_max_3d": temp_max_3d.flatten(),
            "rh_min_3d": rh_min_3d.flatten(),
            "wind_mean_3d": wind_mean_3d.flatten(),
            
            # Derived Fire Weather Indices
            "vpd": vpd.flatten(),
            "hdw": hdw.flatten(),
            "ffwi": ffwi.flatten(),
            
            # Land Cover
            "lc_urban": lc_urban.flatten(),
            "lc_forest": lc_forest.flatten(),
            "lc_shrubland": lc_shrubland.flatten(),
            "lc_grassland": lc_grassland.flatten(),
            "lc_agriculture": lc_agriculture.flatten(),
            
            # Drought
            "pdsi": curr_bands["pdsi"].flatten(),
            "usdm_severity": curr_bands["usdm_severity"].flatten(),
            
            # Human proximity
            "dist_to_roads": curr_bands["dist_to_roads"].flatten(),
            "dist_to_powerlines": curr_bands["dist_to_powerlines"].flatten(),
            
            # Previous day burned state (useful lag feature for fire spread check)
            "burned_yesterday": prev1_bands["burned_area"].flatten()
        }
        
        return pd.DataFrame(df_dict)

    def run(self) -> None:
        """
        Loops through all forecast days, computes daily tabular features,
        and aggregates into the final master feature table.
        """
        logger.info("--- STARTING FEATURE ENGINEERING ---")
        
        all_days_df = []
        
        # We start from day 3 (index 2) to accommodate the 3-day lag features
        for date_idx in tqdm(range(2, len(self.dates)), desc="Engineering features daily"):
            day_df = self.build_daily_features(date_idx)
            
            # Save daily features to disk
            date_str = self.dates[date_idx].strftime("%Y%m%d")
            day_df.to_csv(self.features_daily_dir / f"features_{date_str}.csv", index=False)
            
            all_days_df.append(day_df)
            
        # Combine all days into one master matrix
        logger.info("Merging daily features into unified master feature matrix...")
        master_df = pd.concat(all_days_df, ignore_index=True)
        
        output_path = self.processed_dir / "final_feature_matrix.csv"
        master_df.to_csv(output_path, index=False)
        logger.info(f"Master feature matrix generated with shape {master_df.shape}")
        logger.info(f"Saved to {output_path}")
        logger.info("--- FEATURE ENGINEERING COMPLETED ---")

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    fe = FeatureEngineer(cfg)
    fe.run()
