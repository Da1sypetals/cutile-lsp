import logging
import sys
from datetime import datetime
from typing import Optional

from colorama import Fore, Style, init

init()


class ColoredFormatter(logging.Formatter):
    # Colorama color codes
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA,
        "TIME": Fore.LIGHTBLACK_EX,
        "MODULE": Fore.BLUE,
    }

    def format(self, record):
        # Create colored log level
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_level = f"{Style.BRIGHT}{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        else:
            colored_level = levelname

        # Create timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        colored_time = f"{self.COLORS['TIME']}{timestamp}{Style.RESET_ALL}"

        # Create module name
        module_name = record.module
        colored_module = f"{self.COLORS['MODULE']}{module_name}{Style.RESET_ALL}"

        # Format the message
        message = record.getMessage()

        # Combine all parts
        formatted = f"{colored_time} [{colored_level} {colored_module}] {message}"

        return formatted


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger with colored output.

    Args:
        name: Logger name (defaults to calling module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the name of the calling module
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")
        else:
            name = "cutile-lsp"

    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper()))

        # Create console handler (use stderr to avoid interfering with LSP stdout communication)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Create and set formatter
        formatter = ColoredFormatter()
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger
