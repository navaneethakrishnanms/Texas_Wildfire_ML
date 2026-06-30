import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any
from src.harmonization.harmonizer import GeospatialHarmonizer

def run_harmonization_pipeline(config: Dict[str, Any]) -> None:
    """
    Runs the geospatial harmonization pipeline.
    Reprojects all daily weather/satellite maps, computes distance vectors,
    imputes cloud gaps, and exports standard daily multi-band geotiffs.
    """
    logger.info("========================================")
    logger.info("RUNNING PHASE 3: GEOSPATIAL HARMONIZATION")
    logger.info("========================================")
    
    harmonizer = GeospatialHarmonizer(config)
    harmonizer.run()
    
    logger.info("Phase 3 pipeline completed successfully.")

if __name__ == "__main__":
    from src.utils.config_utils import load_config
    from src.utils.logging_utils import setup_logging
    
    cfg = load_config("configs/config.yaml")
    setup_logging(cfg["paths"]["log_file"])
    run_harmonization_pipeline(cfg)
