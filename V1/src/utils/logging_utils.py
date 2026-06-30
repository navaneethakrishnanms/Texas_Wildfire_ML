import sys
from pathlib import Path
from loguru import logger

def setup_logging(log_file_path: str = "logs/wildfire_pipeline.log", level: str = "INFO") -> None:
    """
    Sets up application-wide logging using loguru.
    Logs are written to both standard output and a file.
    
    Args:
        log_file_path: Path to the log file to create.
        level: Minimum logging level (e.g. DEBUG, INFO, WARNING, ERROR).
    """
    # Create the directory for logs if it does not exist
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add console logger with clean formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        enqueue=True
    )

    # Add file logger with rotation and retention
    logger.add(
        log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True
    )
    
    logger.info("Logging initialized successfully.")
