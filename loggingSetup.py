import logging

logger = logging.getLogger("robot")
logger.setLevel(logging.DEBUG)
handle = logging.FileHandler("robot.log")
handle.setLevel(logging.DEBUG)
logger.addHandler(handle)
handle.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
