"""Base trainer class."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import joblib
import polars as pl

from march_madness.log import logger
from march_madness.path import OUTPUT_DIR
from march_madness.settings import LEAGUE


class BaseTrainer(ABC):
    """Base trainer class."""

    model_cls: ClassVar[Any]

    def __init__(
        self,
        league: LEAGUE = "M",
        preprocessors: dict[str, Any] | None = None,
    ) -> None:
        self.model = self.model_cls()
        self.league = league
        self.preprocessors = preprocessors or {}

    @abstractmethod
    def train(self, df: pl.DataFrame, **kwargs) -> None:
        """Method responsible for training the model."""
        pass

    @abstractmethod
    def predict(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Method responsible for generating predictions."""
        pass

    @abstractmethod
    def generate_data(self, df: pl.DataFrame, *, predict: bool = False, **kwargs) -> Any:
        """Method responsible for generating data for training or prediction."""
        pass

    def save(self, path: str = "M") -> None:
        """Saves the trainer instance to disk."""
        save_to = OUTPUT_DIR / f"{path}/{self.model.name}trainer.joblib"
        joblib.dump(self.preprocessors, save_to)
        logger.info(f"Saved trainer to {save_to}")

    @classmethod
    def load(cls, path: str = "M") -> Self:
        """Loads the trainer instance from disk."""
        load_from = OUTPUT_DIR / f"{path}/{cls.model.name}trainer.joblib"
        preprocessors = joblib.load(load_from)
        logger.info(f"Loaded trainer from {load_from}")
        return cls(preprocessors=preprocessors, league=path)
