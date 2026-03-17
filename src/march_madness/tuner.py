"""Tuning class for model hyperparameters."""

from collections.abc import Generator

import numpy as np
import optuna
import polars as pl
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import root_mean_squared_error

from march_madness.log import logger
from march_madness.models.base import McmcParams, SviParams
from march_madness.trainer import Trainer


class Tuner:
    """Class responsible for tuning model hyperparameters."""

    def __init__(self, df: pl.DataFrame, trainer_cls: type[Trainer], league: str = "M") -> None:
        self.df = df
        self.trainer_cls = trainer_cls
        self.league = league

    def tune(
        self,
        inference: str = "svi",
        num_samples: int = 1_000,
        n_trials: int = 100,
        mcmc_params: McmcParams | None = None,
        svi_params: SviParams | None = None,
        **kwargs,
    ) -> None:
        """Method responsible for tuning the model."""

        def objective(
            trial: optuna.Trial,
        ) -> float:
            svi = svi_params or SviParams(num_steps=15_000)

            # TODO: move params elsewhere to make this more flexible and not specific to PPP model
            params = {
                "alpha": trial.suggest_float("alpha", 0.0, 1.5),
                "beta_ppp_std": trial.suggest_float("beta_ppp_std", 0.01, 0.5),
                "beta_pace_std": trial.suggest_float("beta_pace_std", 0.01, 0.5),
                "offense_global_mean_std": trial.suggest_float("offense_global_mean_std", 0.1, 1),
                "defense_global_mean_std": trial.suggest_float("defense_global_mean_std", 0.1, 1),
                "pace_global_mean_std": trial.suggest_float("pace_global_mean_std", 0.1, 1),
                "sigma_offense_season_rate": trial.suggest_float("sigma_offense_season_rate", 1.0, 15.0),
                "sigma_defense_season_rate": trial.suggest_float("sigma_defense_season_rate", 1.0, 15.0),
                "sigma_pace_season_rate": trial.suggest_float("sigma_pace_season_rate", 1.0, 15.0),
                "sigma_offense_team_rate": trial.suggest_float("sigma_offense_team_rate", 0.1, 2.0),
                "sigma_defense_team_rate": trial.suggest_float("sigma_defense_team_rate", 0.1, 2.0),
                "sigma_pace_team_rate": trial.suggest_float("sigma_pace_team_rate", 0.1, 2.0),
                "hca_team_mu_mean": trial.suggest_float("hca_team_mu_mean", 0.0, 0.1),
                "hca_team_mu_std": trial.suggest_float("hca_team_mu_std", 0.01, 0.05),
                "hca_team_std": trial.suggest_float("hca_team_std", 0.01, 0.1),
                "phi_score_rate": trial.suggest_float("phi_score_rate", 0.01, 2.0),
            }

            team1_score_true: list[float] = []
            team2_score_true: list[float] = []

            team1_score_pred: list[float] = []
            team2_score_pred: list[float] = []

            for train_df, test_df in self.generate_train_val_data(**kwargs):
                if train_df.is_empty() or test_df.is_empty():
                    logger.info("Empty train or test dataframe, skipping this fold.")
                    continue
                trainer = self.trainer_cls(league=self.league)
                trainer.train(
                    df=train_df,
                    inference=inference,
                    num_samples=num_samples,
                    mcmc_params=mcmc_params,
                    svi_params=svi,
                    **params,
                )
                preds = trainer.predict(df=test_df)

                # Accumulate actual scores and predicted scores for evaluation
                team1_score_true.extend(trainer.predict_df["team1_score"].to_list())
                team2_score_true.extend(trainer.predict_df["team2_score"].to_list())

                team1_score_pred.extend(preds["team1_score"].to_list())
                team2_score_pred.extend(preds["team2_score"].to_list())

            # Compute evaluation metric (e.g., mean squared error)
            team1_rmse = root_mean_squared_error(y_true=team1_score_true, y_pred=team1_score_pred)
            team2_rmse = root_mean_squared_error(y_true=team2_score_true, y_pred=team2_score_pred)

            return (team1_rmse + team2_rmse) / 2

        study = optuna.create_study(
            study_name=f"{self.trainer_cls.model_cls.name}_{self.league}_tuning",
            storage="sqlite:///optuna.sqlite3",
            load_if_exists=True,
            direction="minimize",
            sampler=TPESampler(),
            pruner=MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials)
        logger.info("Study complete.")

    def generate_train_val_data(
        self,
        start_season: int = 2018,
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
