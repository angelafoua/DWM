"""
Centralised logging configuration for the ANN-ER pipeline.

Every module obtains its logger via get_logger(__name__) so that the root
handler can be configured once (here) and all child loggers inherit it.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger attached to a stream handler.

    Calling this function multiple times with the same *name* is safe — Python's
    logging registry returns the same Logger instance each time, and we only add
    a StreamHandler when none is already present.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
