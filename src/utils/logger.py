import logging
import os
import sys

def get_logger(name: str = "gpt", log_dir: str | None = None, rank: int = 0) -> logging.Logger:
    """
    Returns a logger with both console and optional file logging.

    Args:
        name (str): Logger name.
        log_dir (str, optional): If provided, logs will also be saved to a file in this directory.
        rank (int): For distributed training; only rank 0 should log to avoid spam.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger