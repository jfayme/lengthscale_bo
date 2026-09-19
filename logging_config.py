"""Central logging configuration for the whole project.

Modules do not configure logging themselves; they ask for a named logger with
`get_logger(__name__)` and emit records through it. Handlers, levels and the
record format are attached once, by `setup_logging`, from the `[logger]` table
of the settings file. Keeping the names per-module while the configuration is
shared means a single call decides where the records go, and every record
still says which module wrote it.

Entry points (scripts, notebooks, the test runner) call `setup_logging` once
at start-up. Library code never does: importing a module must not decide where
someone else's log records end up.

"""

import logging
from pathlib import Path

from settings import SETTINGS

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s (%(name)s)"
DATE_FORMAT = "%d-%b-%y %H:%M:%S"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return the logger a module should write to.

    Parameters
    ----------
    name
        Name of the logger, conventionally the module's `__name__`.

    Returns
    -------
    logger
        The named logger, which inherits the handlers and the level set by
        `setup_logging`.

    """
    return logging.getLogger(name)


def setup_logging(
    level: int | str | None = None,
    path: str | Path | None = None,
    console: bool | None = None,
    force: bool = False,
) -> logging.Logger:
    """Attach the project handlers to the root logger.

    Calling this more than once is harmless: the configuration is applied on
    the first call and subsequent ones are ignored unless `force` is True.
    Warnings issued through the `warnings` module are captured so that they
    reach the same handlers.

    Parameters
    ----------
    level
        Lowest severity to record. Defaults to the settings value.
    path
        File the records are appended to. Defaults to the settings value;
        an empty value disables the file handler.
    console
        If True, records are also written to stderr. Defaults to the
        settings value.
    force
        If True, existing handlers on the root logger are replaced rather
        than left alone.

    Returns
    -------
    logger
        The configured root logger.

    """
    global _configured

    config = SETTINGS["logger"]
    root = logging.getLogger()

    if _configured and not force:
        return root

    if level is None:
        level = config["level"]
    if path is None:
        path = config["path"]
    if console is None:
        console = config["console"]

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers: list[logging.Handler] = []

    if path:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))

    if console:
        handlers.append(logging.StreamHandler())

    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(level)
    logging.captureWarnings(True)

    _configured = True
    return root
