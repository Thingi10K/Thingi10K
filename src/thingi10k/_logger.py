import logging
import colorama

colorama.just_fix_windows_console()

logger = logging.getLogger("thingi10k")

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")


class ColorFormatter(logging.Formatter):
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
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler.setFormatter(ColorFormatter())
logger.addHandler(handler)
