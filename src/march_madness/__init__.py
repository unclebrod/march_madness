"""Initialize the March Madness package."""

import logging

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.logging import RichHandler

load_dotenv(find_dotenv())

console = Console(width=100)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger("dc_music")
