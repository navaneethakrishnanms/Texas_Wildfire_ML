import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any

from src.feature_engineering.features import FeatureEngineer
from src.feature_engineering.labels import LabelGenerator

def run_feature_engineering_pipeline(config: Dict[str, Any]) -> None:
    """
    Orchestrates the feature engineering and label generation pipeline.
    Calculates spatial-temporal features and assigns fire ignition labels.
    """
    logger.info("========================================")
    logger.info("RUNNING PHASE 4 & 5: FEATURE ENGINEERING & LABEL GENERATION")
    logger.info("========================================")
    
    # 1. Feature Engineering
    fe = FeatureEngineer(config)
    fe.run()
    
    # 2. Label Generation
    lg = LabelGenerator(config)
    lg.generate_labels()
    
    logger.info("Phase 4 & 5 pipeline completed successfully. Final training data compiled.")

if __name__ == "__main__":
    from src.utils.config_utils import load_config
    from src.utils.logging_utils import setup_logging
    
    cfg = load_config("configs/config.yaml")
    setup_logging(cfg["paths"]["log_file"])
    run_feature_engineering_pipeline(cfg)
