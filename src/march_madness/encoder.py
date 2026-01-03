"""Encoder classes."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

import dill as pickle
import numpy as np
import polars as pl

from march_madness.log import logger
from march_madness.path import OUTPUT_DIR

ARRAY = pl.DataFrame | pl.Series | np.ndarray


class Encoder(ABC):
    """Base encoder class."""

    name: ClassVar[str]

    def __init__(self, **kwargs) -> None:
        self.encoding = {}
        self.is_fit = False

    @property
    def reverse_encoding(self) -> dict:
        return {val: i for i, val in self.encoding.items()}

    @property
    def classes_(self) -> list:
        return list(self.encoding.keys())

    @abstractmethod
    def fit(self, x: ARRAY) -> Self: ...

    @abstractmethod
    def transform(self, x: ARRAY) -> np.ndarray: ...

    def fit_transform(self, x: ARRAY) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.vectorize(self.reverse_encoding.get)(x)

    def _convert_to_numpy(self, x: ARRAY) -> np.ndarray:
        if isinstance(x, pl.DataFrame | pl.Series):
            x_arr = x.to_numpy()
        elif isinstance(x, np.ndarray):
            x_arr = x
        else:
            raise TypeError("Input must be a Polars DataFrame, Series, or NumPy array.")
        return x_arr

    def _check_if_fit(self) -> None:
        if not self.is_fit:
            raise ValueError(f"{self.__class__.__name__} instance must first be set.")

    def save(self, prefix: str, path: str = "M") -> None:
        save_to = OUTPUT_DIR / f"{path}/{prefix}{self.name}encoder.pkl"
        with Path(save_to).open("wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved encoder to {save_to}")

    @classmethod
    def load(cls, prefix: str, path: str = "M"):
        load_from = OUTPUT_DIR / f"{path}/{prefix}{cls.name}encoder.pkl"
        with Path(load_from).open("rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded encoder from {load_from}")
        return model


class LabelEncoder(Encoder):
    """Label encoder for categorical variables."""

    name: ClassVar[str] = "label"

    def fit(self, x: ARRAY) -> Self:
        x_arr = self._convert_to_numpy(x)
        arr = np.sort(np.unique(x_arr)).tolist()
        self.encoding = dict(zip([None] + arr, range(0, len(arr) + 1), strict=False))
        self.is_fit = True
        return self

    def transform(self, x: ARRAY) -> np.ndarray:
        self._check_if_fit()
        if isinstance(x, pl.DataFrame):
            x_transform = x.with_columns(
                *[pl.col(col).replace_strict(self.encoding, default=0) for col in x.columns]
            ).to_numpy()
        elif isinstance(x, pl.Series):
            x_transform = x.replace_strict(self.encoding, default=0).to_numpy()
        elif isinstance(x, np.ndarray):
            x_transform = np.vectorize(self.encoding.get)(x)
        else:
            raise TypeError("Input must be a Polars DataFrame, Series, or NumPy array.")
        return x_transform


class SequentialEncoder(Encoder):
    """Sequential encoder for ordinal, numerical variables."""

    name: ClassVar[str] = "sequential"

    def fit(self, x: ARRAY, padding: int = 1000) -> Self:
        x_arr = self._convert_to_numpy(x)
        arr = np.sort(np.unique(x_arr)).tolist()
        arr += list(range(arr[-1] + 1, arr[-1] + padding + 1))
        self.encoding = dict(zip(arr, range(len(arr)), strict=False))
        self.is_fit = True
        return self

    def transform(self, x: ARRAY) -> np.ndarray:
        self._check_if_fit()
        if isinstance(x, pl.DataFrame):
            x_transform = x.with_columns(
                *[pl.col(col).replace_strict(self.encoding, default=-1) for col in x.columns]
            ).to_numpy()
        elif isinstance(x, pl.Series):
            x_transform = x.replace_strict(self.encoding, default=-1).to_numpy()
        elif isinstance(x, np.ndarray):
            x_transform = np.vectorize(self.encoding.get)(x)
        else:
            raise TypeError("Input must be a Polars DataFrame, Series, or NumPy array.")
        return x_transform
