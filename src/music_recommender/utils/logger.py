import sys
from pathlib import Path

from loguru import logger

from music_recommender.config import Config

cfg = Config()


def setup_logger(
    level="INFO", log_file: Path | None = None, context: str = "music_recommender"
):
    """
    Configure loguru logger with consistent formatting

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Custom log file path (optional)
        context: Context name for default log file (e.g., 'training', 'inference')
    """
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    if log_file is None:
        # Default: logs/training_2025-11-02.log or logs/inference_2025-11-02.log
        log_file = cfg.paths.logs / f"{context}_{{time:YYYY-MM-DD}}.log"

    cfg.paths.logs.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",  # Compress rotated logs
    )

    return logger


_logger = setup_logger()


def get_logger(context: str | None = None):
    """
    Get the configured logger instance

    Args:
        context: Optional context to reconfigure logger (e.g., 'training', 'inference')
    """
    global _logger
    if context:
        _logger = setup_logger(context=context)
    return _logger