from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "omega_quant/logs") -> None:
    """Auto-docstring."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    import os
    log_name = "serverless_logs.txt" if os.getenv("GITHUB_ACTIONS") else "omega_quant.log"
    file_handler = logging.FileHandler(Path(log_dir) / log_name, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
