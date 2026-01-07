"""Base trainer class."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import joblib
import polars as pl

from march_madness.loader import DataLoader
from march_madness.log import logger
from march_madness.settings import LEAGUE, OUTPUT_DIR


class BaseTrainer(ABC):
    """Base trainer class."""

    model_cls: ClassVar[Any]

    def __init__(
        self,
        model: Any | None = None,
        league: LEAGUE = "M",
        preprocessors: dict[str, Any] | None = None,
    ) -> None:
        self.model = model if model is not None else self.model_cls()
        self.league = league
        self.preprocessors = preprocessors or {}
        self.data_loader = DataLoader(league=league)

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
        save_to = OUTPUT_DIR / f"{path}/{self.model.name}/trainer.joblib"
        joblib.dump(self.preprocessors, save_to)
        self.model.save(path=path)
        logger.info(f"Saved trainer to {save_to}")

    @classmethod
    def load(cls, path: str = "M") -> Self:
        """Loads the trainer instance from disk."""
        model_name = getattr(cls.model_cls, "name", None)
        if model_name is None:
            model_name = cls.model_cls().name
        load_from = OUTPUT_DIR / f"{path}/{model_name}/trainer.joblib"
        preprocessors = joblib.load(load_from)
        model = cls.model_cls.load(path=path)
        logger.info(f"Loaded trainer from {load_from}")
        return cls(preprocessors=preprocessors, league=path, model=model)
