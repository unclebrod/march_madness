"""Trainer implementation for the PPP model."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from march_madness.encoder import LabelEncoder, SequentialEncoder
from march_madness.loader import DataConfig
from march_madness.log import logger
from march_madness.models.ppp_model import PointsPerPossessionData, PointsPerPossessionModel
from march_madness.settings import OUTPUT_DIR
from march_madness.trainers.base_trainer import BaseTrainer
from march_madness.utils import summarize_samples

CONTEXT_INDICATOR_COLUMNS = [
    "is_ncaa_tourney",
    "is_conf_tourney",
]

CONTEXT_NUMERIC_COLUMNS = [
    "days_into_season_norm",
    "days_into_season_norm_sq",
    "log_travel_distance_diff",
    "log_elevation_diff",
    "fatigue_effect_1",
    "fatigue_effect_2",
    "fatigue_effect_3",
    "fatigue_effect_4",
]


@dataclass
class PointsPerPossessionInference:
    coef_df: pl.DataFrame
    season_coef_df: pl.DataFrame
    team_coef_df: pl.DataFrame
    season_team_coef_df: pl.DataFrame

    def save(self, path: str = "M") -> None:
        self.coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/coefs.csv")
        self.season_coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/season_coefs.csv")
        self.team_coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/team_coefs.csv")
        self.season_team_coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/season_team_coefs.csv")


class PointsPerPossessionTrainer(BaseTrainer):
    """Trainer for the Points Per Possession model."""

    model_cls = PointsPerPossessionModel

    def __init__(self, preprocessors: dict[str, Any] | None = None, **kwargs) -> None:
        if preprocessors is None:
            preprocessors = {
                "season_encoder": SequentialEncoder(),
                "team_encoder": LabelEncoder(),
                "coach_encoder": LabelEncoder(),
                "imputer": SimpleImputer(strategy="mean"),
                "scaler": StandardScaler(),
            }
        super().__init__(preprocessors=preprocessors, **kwargs)

    def train(self, df: pl.DataFrame, **kwargs) -> None:
        train_df = df.filter(
            pl.col("team1_score").is_not_null(),
            pl.col("team2_score").is_not_null(),
            pl.col("avg_poss").is_not_null(),
            pl.col("season").is_not_null(),
            pl.col("team1_id").is_not_null(),
            pl.col("team2_id").is_not_null(),
        )
        logger.info(
            f"Removed {len(df) - len(train_df)} rows with missing values ({len(train_df) / len(df):.2%})."
        )

        context_num = train_df.select(CONTEXT_NUMERIC_COLUMNS).to_numpy()
        context_num_imputed = self.preprocessors["imputer"].fit_transform(context_num)

        self.preprocessors["scaler"].fit(context_num_imputed)
        self.preprocessors["season_encoder"].fit(train_df["season"])
        self.preprocessors["team_encoder"].fit(train_df.select(["team1_id", "team2_id"]))
        self.preprocessors["coach_encoder"].fit(train_df.select(["team1_coach", "team2_coach"]))

        data = self.generate_data(train_df, predict=False)
        self.model.fit(data=data, **kwargs)

    def predict(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        data = self.generate_data(df, predict=True)
        return self.model.predict(data=data, **kwargs)

    def generate_data(
        self,
        df: pl.DataFrame,
        *,
        predict: bool = False,
        **kwargs,
    ) -> PointsPerPossessionData:
        context_num = self.preprocessors["imputer"].transform(
            df.select(CONTEXT_NUMERIC_COLUMNS).to_numpy()
        )
        context_num = self.preprocessors["scaler"].transform(context_num)

        return PointsPerPossessionData(
            context_ind=jnp.asarray(df.select(CONTEXT_INDICATOR_COLUMNS).to_numpy()),
            context_num=jnp.asarray(context_num),
            n_teams=len(self.preprocessors["team_encoder"].classes_),
            n_seasons=len(self.preprocessors["season_encoder"].classes_),
            season=self.preprocessors["season_encoder"].transform(df["season"]),
            team1=self.preprocessors["team_encoder"].transform(df["team1_id"]),
            team2=self.preprocessors["team_encoder"].transform(df["team2_id"]),
            minutes=df["minutes"].to_jax(),
            is_neutral=df["is_neutral"].to_jax(),
            team1_home=df["team1_home"].to_jax(),
            team2_home=df["team2_home"].to_jax(),
            avg_poss=df["avg_poss"].to_jax() if not predict else None,
            team1_score=df["team1_score"].to_jax() if not predict else None,
            team2_score=df["team2_score"].to_jax() if not predict else None,
        )

    def infer(self) -> PointsPerPossessionInference:
        """Run inference on the trained model."""
        coef_list: list[pl.DataFrame] = []
        season_list: list[pl.DataFrame] = []
        team_list: list[pl.DataFrame] = []
        season_team_list: list[pl.DataFrame] = []

        season_classes = self.preprocessors["season_encoder"].classes_
        team_classes = self.preprocessors["team_encoder"].classes_

        teams = self.data_loader.load_data(DataConfig.teams).select(
            pl.col("TeamID").alias("team_id"),
            pl.col("TeamName").alias("team_name"),
            pl.col("FirstD1Season").alias("first_d1_season"),
            pl.col("LastD1Season").alias("last_d1_season"),
        )

        for k, v in self.model.samples.items():
            summary = summarize_samples(v)
            if k in [
                "defense_team_rw",
                "offense_team_rw",
                "pace_team_rw",
            ]:
                for team_idx, team in enumerate(team_classes):
                    df = pl.DataFrame(summary.get_index(team_idx).model_dump()).with_columns(
                        team_id=pl.lit(team),
                        season=pl.Series(season_classes),
                        name=pl.lit(k),
                    )
                    season_team_list.append(df)
                continue

            summary_dict = summary.model_dump()

            if k in [
                "defense_season_rw",
                "offense_season_rw",
                "pace_season_rw",
                "hca_season",
            ]:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.lit(k),
                    season=pl.Series(season_classes),
                )
                season_list.append(df)
                continue

            if k in ["hca_team"]:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.lit(k),
                    team_id=pl.Series(team_classes),
                )
                team_list.append(df)
                continue

            if (
                k
                in [
                    "defense_global_mean",
                    "offense_global_mean",
                    "pace_global_mean",
                    "phi_score",
                ]
                or "sigma" in k
            ):
                df = pl.DataFrame(summary_dict).with_columns(name=pl.lit(k))
            elif "beta_ind" in k:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.Series([f"{k}_{x}" for x in CONTEXT_INDICATOR_COLUMNS]),
                )
            elif "beta_num" in k:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.Series([f"{k}_{x}" for x in CONTEXT_NUMERIC_COLUMNS]),
                )
            else:
                logger.warning(f"Unhandled posterior sample key: {k}")
                continue

            coef_list.append(df)

        inference = PointsPerPossessionInference(
            coef_df=pl.concat(coef_list),
            season_coef_df=pl.concat(season_list),
            team_coef_df=pl.concat(team_list).join(teams, on="team_id", how="left"),
            season_team_coef_df=pl.concat(season_team_list, how="vertical_relaxed").join(
                teams, on="team_id", how="left"
            ),
        )
        inference.save(path=self.league)
        return inference
