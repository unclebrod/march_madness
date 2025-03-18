"""Initialize the March Madness package."""

import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.logging import RichHandler

load_dotenv(find_dotenv())

ROOT_DIR = Path(__file__).resolve().parents[2]  # go up two levels to march_madness
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

console = Console(width=100)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger("dc_music")
