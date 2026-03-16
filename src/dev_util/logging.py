import logging
import logging.handlers

from dev_util.dir import LOGS

DEFAULT_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGING_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logger() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    level = logging.INFO

    sh = logging.StreamHandler()
    sh.setFormatter(DEFAULT_FORMATTER)
    sh.setLevel(level)

    root_logger.addHandler(sh)
    root_logger.setLevel(logging.INFO)


def get_logger(name: str, file_handler: bool, logging_level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    dir = LOGS / name
    dir.mkdir(parents=True, exist_ok=True)
    filename = dir / "logs.log"
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=filename,
        when="midnight",
        backupCount=10,
    )
    handler.setFormatter(DEFAULT_FORMATTER)
    handler.setLevel(LOGGING_LEVELS[logging_level])

    logger.addHandler(handler)
    return logger
