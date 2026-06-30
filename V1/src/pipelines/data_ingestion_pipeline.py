import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any

from src.ingestion.simulator import GeospatialSimulator
from src.ingestion.noaa_ndfd import NOAAWeatherIngestor
from src.ingestion.nasa_earth_observation import NASAEarthObservationIngestor
from src.ingestion.terrain_dem import TerrainDEMIngestor
from src.ingestion.land_cover import LandCoverIngestor
from src.ingestion.drought import DroughtIngestor
from src.ingestion.human_activity import HumanActivityIngestor
from src.ingestion.historical_fires import HistoricalFiresIngestor

def run_ingestion_pipeline(config: Dict[str, Any], force_simulate: bool = True) -> None:
    """
    Orchestrates the raw data ingestion pipeline.
    If force_simulate is True (default for POC), it generates high-fidelity simulated 
    geospatial/weather data to enable instant local training.
    Then, it triggers ingestors to inspect and verify files.
    """
    logger.info("========================================")
    logger.info("RUNNING PHASE 2: DATA ACQUISITION PIPELINE")
    logger.info("========================================")
    
    if force_simulate:
        logger.info("Simulation mode enabled: Running Geospatial & Weather Simulation Engine...")
        simulator = GeospatialSimulator(config)
        simulator.run()
    else:
        logger.info("Direct ingestion mode: Attempting to verify existing raw files...")

    # Instantiate ingestor validation steps
    weather_ingestor = NOAAWeatherIngestor(config)
    weather_ingestor.download_data(
        start_date=config["temporal"]["start_date"],
        end_date=config["temporal"]["end_date"],
        bbox=config["spatial"]
    )
    
    nasa_ingestor = NASAEarthObservationIngestor(config)
    nasa_ingestor.download_data(
        start_date=config["temporal"]["start_date"],
        end_date=config["temporal"]["end_date"]
    )
    
    dem_ingestor = TerrainDEMIngestor(config)
    dem_ingestor.download_data()
    
    lc_ingestor = LandCoverIngestor(config)
    lc_ingestor.download_data()
    
    drought_ingestor = DroughtIngestor(config)
    drought_ingestor.download_data()
    
    infra_ingestor = HumanActivityIngestor(config)
    infra_ingestor.download_data()
    
    history_ingestor = HistoricalFiresIngestor(config)
    history_ingestor.download_data()
    
    logger.info("Phase 2 pipeline completed successfully. All raw datasets initialized.")

if __name__ == "__main__":
    from src.utils.config_utils import load_config
    from src.utils.logging_utils import setup_logging
    
    cfg = load_config("configs/config.yaml")
    setup_logging(cfg["paths"]["log_file"])
    run_ingestion_pipeline(cfg)
