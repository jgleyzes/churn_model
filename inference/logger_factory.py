# -*- coding: utf-8 -*-

# Internal imports
import logging
import time

logging.Formatter.converter = time.gmtime
LOG_FORMAT = "%(asctime)s::%(name)s::%(levelname)s::%(message)s"
FORMATTER = logging.Formatter(LOG_FORMAT)
console_output = logging.StreamHandler()


def create_logger(name, lvl=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    logger.propagate = False

    console_output.setLevel(logging.DEBUG)
    console_output.setFormatter(FORMATTER)
    logger.addHandler(console_output)
    return logger
