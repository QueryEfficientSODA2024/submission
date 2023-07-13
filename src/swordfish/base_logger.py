import logging

log = logging
log.basicConfig(
    format=f"%(asctime)s[%(filename)s.%(levelname)s]: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = log.getLogger(__name__)
