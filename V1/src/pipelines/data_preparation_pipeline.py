import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any
from src.preprocessing.data_prep import DataPreparer

def run_preparation_pipeline(config: Dict[str, Any]) -> None:
    """
    Orchestrates the data preparation and split preprocessing pipeline.
    """
    logger.info("========================================")
    logger.info("RUNNING PHASE 6: DATA PREPARATION & SPLITTING")
    logger.info("========================================")
    
    preparer = DataPreparer(config)
    preparer.process_and_save()
    
    logger.info("Phase 6 pipeline completed successfully. Data matrices ready for training.")

if __name__ == "__main__":
    from src.utils.config_utils import load_config
    from src.utils.logging_utils import setup_logging
    
    cfg = load_config("configs/config.yaml")
    setup_logging(cfg["paths"]["log_file"])
    run_preparation_pipeline(cfg)
