import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any
from src.training.trainer import WildfireRiskTrainer

def run_training_pipeline(config: Dict[str, Any]) -> Any:
    """
    Orchestrates the XGBoost model training pipeline.
    """
    logger.info("========================================")
    logger.info("RUNNING PHASE 8: XGBOOST MODEL TRAINING")
    logger.info("========================================")
    
    trainer = WildfireRiskTrainer(config)
    model = trainer.train_final_model()
    
    logger.info("Phase 8 pipeline completed successfully. Model weights saved.")
    return model

if __name__ == "__main__":
    from src.utils.config_utils import load_config
    from src.utils.logging_utils import setup_logging
    
    cfg = load_config("configs/config.yaml")
    setup_logging(cfg["paths"]["log_file"])
    run_training_pipeline(cfg)
