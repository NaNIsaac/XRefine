import sys
from pathlib import Path

import loguru

from xrefine.config import CONFIG


def setup_logger(level="INFO") -> None:
    """
    Configures and sets up the logger with a custom format and log file settings.

    Returns:
        Logger: The configured logger instance.
    """

    formatter: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}"
    )
    formatter = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    log_dir: Path = CONFIG.log_dir
    log_file: Path = log_dir / "{time:YYYY-MM-DD}/{time:YYYYMMDD_HHmmss}.log"

    loguru.logger.remove()
    loguru.logger.add(log_file, format=formatter, rotation="500 MB", compression="zip", level=level)
    loguru.logger.add(sys.stdout, format=formatter, level=level)


setup_logger("DEBUG")
logger = loguru.logger

if __name__ == "__main__":
    setup_logger()
    logger.debug("This is a debug message.")

    logger.info("This is an info message.")

    logger.warning("This is a warning message.")

    logger.error("This is an error message.")

    try:
        raise ValueError("This is a test exception.")
    except ValueError as e:
        logger.exception(f"An exception occurred: {e}")

    metadata = {"key1": "value1", "key2": "value2"}
    logger.info(metadata)

    logger.info("All tests passed.")
