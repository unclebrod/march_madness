"""Path management for the march_madness project."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # go up two levels to march_madness
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
