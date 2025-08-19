import logging
import os
import sys

def get_logger(name: str = "gpt", log_dir: str | None = "logs",
    exp_name: str | None = None, rank: int = 0, verbose: bool = False) -> logging.Logger:
    """
    Logger with console + file output.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = logging.DEBUG if verbose and rank == 0 else (logging.INFO if rank == 0 else logging.ERROR)
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            if exp_name is None:
                exp_name = name
            exp_dir = os.path.join(log_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(exp_dir, f"{exp_name}_{timestamp}.log")

            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"Logging to {log_path}")

    return logger
