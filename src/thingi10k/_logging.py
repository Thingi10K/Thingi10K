"""Logging utilities for thingi10k package.

This module provides colored logging functionality using colorama for cross-platform
color support. It sets up a logger with colored output based on log levels.

The module automatically handles Windows console compatibility and provides
different colors for different log levels (DEBUG/INFO: grey, WARNING: yellow,
ERROR: red, CRITICAL: bright red).
"""
import logging
import colorama # type: ignore[import]
import platform

if platform.system() == "Windows":
    colorama.just_fix_windows_console()

logger = logging.getLogger("thingi10k")
handler = logging.StreamHandler()


class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds colors to log messages based on log level.

    This formatter uses colorama to provide cross-platform colored output in the console.
    Different log levels are displayed in different colors:
    - DEBUG/INFO: Light grey
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bright red

    The format includes timestamp, logger name, colored log level, and the message.
    """
    grey = colorama.Fore.LIGHTBLACK_EX
    yellow = colorama.Fore.YELLOW
    red = colorama.Fore.RED
    bold_red = colorama.Style.BRIGHT + colorama.Fore.RED
    reset = colorama.Style.RESET_ALL
    format_template = (
        "[%(asctime)s] [%(name)s] {color}[%(levelname)s]{reset} %(message)s"
    )

    FORMATS = {
        logging.DEBUG: format_template.format(color=grey, reset=reset),
        logging.INFO: format_template.format(color=grey, reset=reset),
        logging.WARNING: format_template.format(color=yellow, reset=reset),
        logging.ERROR: format_template.format(color=red, reset=reset),
        logging.CRITICAL: format_template.format(color=bold_red, reset=reset),
    }

    def format(self, record):
        """Format a log record with appropriate colors based on log level.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with color codes.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler.setFormatter(ColorFormatter())
logger.addHandler(handler)
