from pathlib import Path
from typing import Any, Dict
import yaml
from loguru import logger

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads and parses the YAML configuration file.
    
    Args:
        config_path: Path to the yaml config file.
        
    Returns:
        A dictionary containing configurations.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise e
            
    # Ensure standard directories exist
    if "paths" in config:
        for name, dir_path in config["paths"].items():
            # If the path specifies a directory (doesn't have an extension)
            if "dir" in name:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                
    return config
