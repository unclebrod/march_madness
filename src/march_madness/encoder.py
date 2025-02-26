from typing import Self

import jax.numpy as jnp
import numpy as np
import polars as pl


class LabelEncoder:
    """Encode categorical features as integers."""

    def __init__(self):
        self.encoding = {}
        self.is_fit = False

    def fit(self, x: pl.DataFrame | pl.Series | np.ndarray) -> Self:
        if isinstance(x, pl.DataFrame | pl.Series):
            x_arr = x.to_numpy()
        elif isinstance(x, np.ndarray):
            x_arr = x
        else:
            raise TypeError("Input must be a Polars DataFrame, Series, or NumPy array.")
        arr = np.sort(np.unique(x_arr)).tolist()
        self.encoding = dict(zip(arr, range(1, len(arr) + 1), strict=False))
        self.is_fit = True
        return self

    def transform(self, x: pl.DataFrame | pl.Series) -> np.ndarray:
        if not self.is_fit:
            raise ValueError("LabelEncoder instance must first be set.")
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

    def fit_transform(self, x: pl.DataFrame | pl.Series | np.ndarray) -> jnp.array:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.vectorize(self.reverse_encoding.get)(x)

    @property
    def reverse_encoding(self) -> dict:
        return {val: i for i, val in self.encoding.items()}

    @property
    def classes_(self) -> list:
        return list(self.encoding.keys())
