"""Tuning class for model hyperparameters."""

from collections.abc import Generator

import numpy as np
import polars as pl

from march_madness.log import logger
from march_madness.trainer import Trainer


class Tuner:
    """Class responsible for tuning model hyperparameters."""

    def __init__(self, df: pl.DataFrame, trainer_cls: type[Trainer], league: str = "M") -> None:
        self.df = df
        self.trainer_cls = trainer_cls
        self.league = league

    def tune(self, **kwargs) -> None:
        """Method responsible for tuning the model."""
        for train_df, test_df in self.generate_train_val_data(**kwargs):
            print("ok")

    def generate_train_val_data(
        self,
        start_season: int = 2012,
        num_regular_season_cutpoints: int = 3,
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
        """Generate training and validation data for tuning."""
        max_season = self.df["season"].max()
        for season in range(start_season, max_season + 1):
            season_df = self.df.filter(pl.col("season").eq(season))
            max_days_into_regular_season = season_df.filter(
                pl.col("is_ncaa_tourney").eq(0),
                pl.col("is_secondary_tourney").eq(0),
            )["days_into_season"].max()
            max_days_into_season = season_df["days_into_season"].max() + 1
            regular_season_cutpoints = (
                np.rint(np.linspace(0, max_days_into_regular_season + 1, num=num_regular_season_cutpoints + 1))
                .astype(int)
                .tolist()
            )
            cutpoints = regular_season_cutpoints + [max_days_into_season]
            for i in range(len(cutpoints) - 1):
                start_day = cutpoints[i]
                end_day = cutpoints[i + 1]
                logger.info(f"Tuning for season {season} from day {start_day} to day {end_day}.")
                train_df = self.df.filter(
                    pl.col("season")
                    .lt(season)
                    .or_(pl.col("season").eq(season).and_(pl.col("days_into_season").lt(start_day))),
                )
                test_df = self.df.filter(
                    pl.col("season").eq(season),
                    pl.col("days_into_season").ge(start_day),
                    pl.col("days_into_season").lt(end_day),
                )
                yield train_df, test_df
