"""Logging singleton."""

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console(width=100)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)

logger = logging.getLogger("march_madness")
