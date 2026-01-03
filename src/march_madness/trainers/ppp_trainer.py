"""Trainer implementation for the PPP model."""

import jax.numpy as jnp
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from march_madness.encoder import LabelEncoder, SequentialEncoder
from march_madness.log import logger
from march_madness.models.ppp_model import PointsPerPossessionData, PointsPerPossessionModel
from march_madness.trainers.base_trainer import BaseTrainer


class PointsPerPossessionTrainer(BaseTrainer):
    """Trainer for the Points Per Possession model."""

    model_cls = PointsPerPossessionModel

    def __init__(self, **kwargs) -> None:
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

        self.preprocessors["season_encoder"].fit(train_df["season"])
        self.preprocessors["team_encoder"].fit(train_df.select(["team1_id", "team2_id"]))
        self.preprocessors["coach_encoder"].fit(train_df.select(["team1_coach", "team2_coach"]))

        context_ind = train_df.select(
            pl.lit(1).alias("intercept"),
            pl.col("is_ncaa_tourney").cast(pl.Int8),
            pl.col("is_conf_tourney").cast(pl.Int8),
        ).to_numpy()

        context_num = train_df.select(
            pl.col("days_into_season"),
            pl.col("days_into_season_sq"),
            pl.col("log_travel_distance_diff"),
            pl.col("log_elevation_diff"),
            pl.col("fatigue_effect_1"),
            pl.col("fatigue_effect_2"),
            pl.col("fatigue_effect_3"),
            pl.col("fatigue_effect_4"),
        ).to_numpy()

        context_num = self.preprocessors["imputer"].fit_transform(context_num)
        context_num = self.preprocessors["scaler"].fit_transform(context_num)

        data = self.generate_data(
            train_df,
            context_ind=jnp.asarray(context_ind),
            context_num=jnp.asarray(context_num),
            predict=False,
        )
        self.model.fit(data=data, **kwargs)
        print("ok")

    def predict(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        pass

    def generate_data(
        self,
        df: pl.DataFrame,
        context_ind: jnp.ndarray,
        context_num: jnp.ndarray,
        *,
        predict: bool = False,
        **kwargs,
    ) -> PointsPerPossessionData:
        return PointsPerPossessionData(
            context_ind=context_ind,
            context_num=context_num,
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
