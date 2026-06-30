from pathlib import Path
from loguru import logger
from typing import Dict, Any
from src.evaluation.explainability import ModelExplainer


def run_explainability_pipeline(config: Dict[str, Any]) -> None:
    """
    Orchestrates the SHAP explainability pipeline (Phase 10).
    Generates global feature importance and local waterfall plots.
    """
    logger.info("========================================")
    logger.info("RUNNING PHASE 10: MODEL EXPLAINABILITY (SHAP)")
    logger.info("========================================")

    explainer = ModelExplainer(config)
    explainer.explain()

    logger.info("Phase 10 pipeline completed. SHAP outputs saved to outputs/.")


if __name__ == "__main__":
    from src.utils.config_utils import load_config
    from src.utils.logging_utils import setup_logging

    cfg = load_config("configs/config.yaml")
    setup_logging(cfg["paths"]["log_file"])
    run_explainability_pipeline(cfg)
