import logging
import coloredlogs


logger = logging.getLogger(__name__)
logger.propagate = False

coloredlogs.DEFAULT_FIELD_STYLES = {
    "asctime": {"color": "green"},
    "hostname": {"color": "magenta"},
    "name": {"color": "blue"},
    "programname": {"color": "cyan"},
    "funcName": {"color": "blue"},
}
coloredlogs.install(
    level="DEBUG",
    logger=logger,
    fmt="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
    datefmt="%H:%M:%S",
)