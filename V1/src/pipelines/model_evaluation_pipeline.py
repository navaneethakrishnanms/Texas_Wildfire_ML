import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any
from src.evaluation.evaluator import ModelEvaluator

def run_evaluation_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the model evaluation pipeline.
    """
    logger.info("========================================")
    logger.info("RUNNING PHASE 9: MODEL EVALUATION")
    logger.info("========================================")
    
    evaluator = ModelEvaluator(config)
    report = evaluator.evaluate()
    
    logger.info("Phase 9 pipeline completed successfully. Evaluation reports generated.")
    return report

if __name__ == "__main__":
    from src.utils.config_utils import load_config
    from src.utils.logging_utils import setup_logging
    
    cfg = load_config("configs/config.yaml")
    setup_logging(cfg["paths"]["log_file"])
    run_evaluation_pipeline(cfg)
