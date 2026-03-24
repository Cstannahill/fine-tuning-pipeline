"""
Logger Utilities
Setup and configuration for logging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""

    # Create logger
    logger = logging.getLogger("unsloth_pipeline")
    logger.setLevel(getattr(logging, config.get("level", "INFO")))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = config.get("log_file", "training.log")
    if log_file:
        # Create logs directory
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)

        # Add timestamp to log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_path / f"{timestamp}_{log_file}"

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class TrainingLogger:
    """Custom logger for training metrics"""

    def __init__(self, logger: logging.Logger, log_dir: Path):
        self.logger = logger
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create metrics log file
        self.metrics_file = self.log_dir / "metrics.jsonl"

    def log_step(self, step: int, metrics: Dict[str, Any]):
        """Log training step metrics"""
        import json

        log_entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}

        # Append to metrics file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch-level metrics"""
        self.logger.info(
            f"Epoch {epoch} completed - "
            + " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

    def log_eval(self, metrics: Dict[str, Any]):
        """Log evaluation metrics"""
        self.logger.info(
            "Evaluation results - "
            + " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )
