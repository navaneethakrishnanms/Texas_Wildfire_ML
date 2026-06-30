from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from tqdm import tqdm
from typing import Dict, Any


class LabelGenerator:
    """
    Generates binary fire ignition labels aligned to the feature matrix.

    Label definition
    ----------------
    y = 1  if a *new* ignition is detected in the same grid cell on day t+1
           OR day t+2 (24–48 h lead time), AND the cell shows no prior fire
           activity during t, t-1, t-2  (prevents labelling ongoing spread).
    y = 0  otherwise.

    Rows where the cell is *already* burning on day t are excluded from
    the training set to keep the model focused on ignition, not propagation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temporal = config["temporal"]
        self.paths = config["paths"]

        self.processed_dir = Path(self.paths["processed_dir"])
        self.harmonized_dir = self.processed_dir / "daily_harmonized"

        self.dates = pd.date_range(
            start=self.temporal["start_date"],
            end=self.temporal["end_date"],
            freq=self.temporal["frequency"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_burned(self, date_str: str) -> np.ndarray:
        """Read burned-area band (Band 15) from a harmonised GeoTIFF."""
        path = self.harmonized_dir / f"harmonized_{date_str}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Harmonised raster not found: {path}")
        with rasterio.open(path) as src:
            return src.read(15)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_labels(self) -> None:
        """
        Vectorised label generation pipeline.

        Steps
        -----
        1. Load all daily burned-area rasters into a 3-D array
           (n_days × height × width).
        2. For each day t, compute new-ignition grids for t+1 and t+2
           using element-wise boolean logic (fully vectorised).
        3. Flatten grids and join to the feature CSV by (date, row, col).
        4. Filter out active-fire rows, drop trailing no-future-data rows,
           and save the final training dataset.
        """
        logger.info("--- STARTING LABEL GENERATION (vectorised) ---")

        # ── 1. Load feature matrix ───────────────────────────────────
        feat_path = self.processed_dir / "final_feature_matrix.csv"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature matrix not found: {feat_path}")

        logger.info("Loading master feature matrix …")
        features_df = pd.read_csv(feat_path)
        logger.info(f"  Shape: {features_df.shape}")

        # ── 2. Load all burned-area rasters ─────────────────────────
        logger.info("Loading all daily burned-area rasters …")
        n_days = len(self.dates)
        burned_stack: list[np.ndarray] = []
        for date in tqdm(self.dates, desc="Reading burned rasters"):
            burned_stack.append(self._read_burned(date.strftime("%Y%m%d")))

        # Shape: (n_days, height, width)
        burned = np.array(burned_stack, dtype=np.int8)
        logger.info(f"Burned stack shape: {burned.shape}")

        # ── 3. Compute per-day label grids ───────────────────────────
        # date_label_map: date_str → flat 1-D label array (one value per cell)
        # date_excl_map : date_str → flat 1-D bool exclusion array
        date_label_map: dict[str, np.ndarray] = {}
        date_excl_map: dict[str, np.ndarray] = {}

        for t in tqdm(range(n_days), desc="Generating ignition grids"):
            date_str = self.dates[t].strftime("%Y-%m-%d")

            # Cannot label if we lack t+1 or t+2 future data
            if t >= n_days - 2:
                h, w = burned.shape[1], burned.shape[2]
                date_label_map[date_str] = np.full(h * w, np.nan)
                date_excl_map[date_str] = np.ones(h * w, dtype=bool)
                continue

            burn_t   = burned[t]                  # today
            burn_t1  = burned[t + 1]              # tomorrow
            burn_t2  = burned[t + 2]              # day after tomorrow
            burn_pm1 = burned[max(t - 1, 0)]      # yesterday
            burn_pm2 = burned[max(t - 2, 0)]      # two days ago

            # Exclusion mask: already burning today
            excl = (burn_t == 1)

            # New ignition at t+1:
            #   fire at t+1, no fire at t, t-1, t-2
            ign_t1 = (
                (burn_t1 == 1) &
                (burn_t   == 0) &
                (burn_pm1 == 0) &
                (burn_pm2 == 0)
            )

            # New ignition at t+2:
            #   fire at t+2, no fire at t+1, t, t-1
            ign_t2 = (
                (burn_t2  == 1) &
                (burn_t1  == 0) &
                (burn_t   == 0) &
                (burn_pm1 == 0)
            )

            label_grid = (ign_t1 | ign_t2).astype(np.float32)

            date_label_map[date_str] = label_grid.flatten()
            date_excl_map[date_str]  = excl.flatten()

        # ── 4. Join labels onto the feature DataFrame ─────────────────
        logger.info("Joining labels onto feature matrix …")

        # Build flat label & excl arrays in the same row-order as features_df
        all_labels: list[np.ndarray] = []
        all_excl:   list[np.ndarray] = []

        for date_str in features_df["date"].unique():
            mask = features_df["date"] == date_str
            all_labels.append(date_label_map[date_str])
            all_excl.append(date_excl_map[date_str])

        # Concatenate in feature_df order (dates are already sorted in the CSV)
        features_df["target_ignition"]    = np.concatenate(all_labels)
        features_df["exclude_active_fire"] = np.concatenate(all_excl)

        # ── 5. Save unfiltered dataset ───────────────────────────────
        unfiltered_path = self.processed_dir / "master_dataset_unfiltered.csv"
        features_df.to_csv(unfiltered_path, index=False)
        logger.info(f"Unfiltered dataset → {unfiltered_path}")

        # ── 6. Filter and save final training dataset ─────────────────
        valid_mask = (
            features_df["target_ignition"].notna() &
            (~features_df["exclude_active_fire"].astype(bool))
        )
        final_df = features_df[valid_mask].copy()
        final_df.drop(columns=["exclude_active_fire"], inplace=True)

        final_path = self.processed_dir / "final_training_dataset.csv"
        final_df.to_csv(final_path, index=False)

        # ── 7. Report class distribution ─────────────────────────────
        num_pos = int((final_df["target_ignition"] == 1).sum())
        num_neg = int((final_df["target_ignition"] == 0).sum())
        pct_pos = num_pos / max(1, num_pos + num_neg) * 100

        logger.info(f"Final training dataset → {final_path} | shape: {final_df.shape}")
        logger.info(
            f"Class distribution — Negatives: {num_neg:,}  "
            f"Positives: {num_pos:,}  ({pct_pos:.4f}% fire)"
        )
        logger.info("--- LABEL GENERATION COMPLETED ---")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    LabelGenerator(cfg).generate_labels()
