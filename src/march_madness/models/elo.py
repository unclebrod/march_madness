"""Bradley-Terry/Elo probabilistic model."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from march_madness.encoder import LabelEncoder, SequentialEncoder
from march_madness.loader import DataConfig
from march_madness.log import logger
from march_madness.models.base import BaseNumpyroModel
from march_madness.settings import OUTPUT_DIR
from march_madness.trainers.base_trainer import BaseTrainer
from march_madness.utils import summarize_samples

CONTEXT_INDICATOR_COLUMNS = [
    "b2b_diff",
    "2in3_diff",
    "3in4_diff",
    "4in5_diff",
]

CONTEXT_NUMERIC_COLUMNS = [
    "team1_log_travel_distance_diff",
    "team1_log_elevation_diff",
]


@dataclass
class EloData:
    context: jnp.array  # fixed effects (tournament, travel, rest, elevation, etc.)

    n_teams: int | None  # number of unique teams
    n_seasons: int | None  # number of unique seasons

    season: jnp.ndarray  # season encoding
    team1: jnp.ndarray  # team1 encoding
    team2: jnp.ndarray  # team2 encoding

    team1_home: jnp.ndarray  # is team1 the home team
    team2_home: jnp.ndarray  # is team2 the home team

    spread: jnp.ndarray  # point spread (team2_score - team1_score)

    team1_win: jnp.ndarray  # whether team1 won the game
    team2_win: jnp.ndarray  # whether team2 won the game


@dataclass
class EloInference:
    coef_df: pl.DataFrame
    season_team_df: pl.DataFrame

    def save(self, path: str = "M") -> None:
        self.coef_df.write_csv(OUTPUT_DIR / f"{path}/elo/coefs.csv")
        self.season_team_df.write_csv(OUTPUT_DIR / f"{path}/elo/season_team_coefs.csv")


class EloModel(BaseNumpyroModel):
    name = "elo"

    def model(
        self,
        data: EloData,
        beta_std: float = 0.05,
        sigma_rating_rate: float = 1.0,
        predict: bool = False,
        **kwargs,
    ) -> None:
        # ---- Fixed effects matrices ----
        n_context = data.context.shape[1]
        beta = numpyro.sample("beta", dist.Normal(0, beta_std).expand((n_context,)))
        fixed_effects = jnp.dot(data.context, beta)

        # ---- Team-specific ratings as deviations from the mean ----
        sigma_rating = numpyro.sample("sigma_rating", dist.HalfNormal(sigma_rating_rate))
        with numpyro.plate("teams", data.n_teams):
            rating = numpyro.sample(
                "rating",
                dist.GaussianRandomWalk(sigma_rating, num_steps=data.n_seasons),
            )
        rating = rating - jnp.mean(rating)

        # ---- Game spread as a function of team latent ratings ----
        # alpha = numpyro.sample("alpha", dist.LogNormal(0, 1))
        # sigma_spread = numpyro.sample("sigma_spread", dist.HalfNormal(5.0))
        hca = numpyro.sample("hca", dist.Normal(0.5, 1.0))
        # intercept = numpyro.sample("intercept", dist.Normal(0, 1.0))
        home_diff = data.team1_home - data.team2_home
        mu = (
            # alpha * (rating[data.team1, data.season] - rating[data.team2, data.season])
            (rating[data.team1, data.season] - rating[data.team2, data.season]) + fixed_effects + (hca * home_diff)
            # + (intercept if not predict else 0.0)
        )
        # mu_centered = mu - mu.mean()

        # ---- Likelihood for spread and win probability ----
        numpyro.sample("team1_win_prob", dist.Bernoulli(logits=mu), obs=data.team1_win)
        numpyro.sample("team2_win_prob", dist.Bernoulli(logits=-mu), obs=data.team2_win)


class EloTrainer(BaseTrainer):
    """Trainer implementation for the Elo model."""

    model_cls = EloModel

    def __init__(self, preprocessors: dict[str, Any] | None = None, **kwargs) -> None:
        if preprocessors is None:
            preprocessors = {
                "season_encoder": SequentialEncoder(),
                "team_encoder": LabelEncoder(),
                "imputer": SimpleImputer(strategy="mean"),
                "scaler": StandardScaler(),
            }
        super().__init__(preprocessors=preprocessors, **kwargs)

    def _filter_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        filtered_df = df.filter(
            pl.col("spread").is_not_null(),
            pl.col("team1_win").is_not_null(),
            pl.col("team2_win").is_not_null(),
            pl.col("season").is_not_null(),
            pl.col("team1_id").is_not_null(),
            pl.col("team2_id").is_not_null(),
        )
        logger.info(
            f"Removed {len(df) - len(filtered_df)} rows with missing values ({len(filtered_df) / len(df):.2%})."
        )
        return filtered_df

    def train(self, df: pl.DataFrame, **kwargs) -> None:
        """Method responsible for training the model."""
        train_df = self._filter_dataframe(df)

        context_num = train_df.select(CONTEXT_NUMERIC_COLUMNS).to_numpy()
        context_num_imputed = self.preprocessors["imputer"].fit_transform(context_num)

        self.preprocessors["scaler"].fit(context_num_imputed)
        self.preprocessors["season_encoder"].fit(train_df["season"])
        self.preprocessors["team_encoder"].fit(train_df.select(["team1_id", "team2_id"]))

        data = self.generate_data(train_df, predict=False)
        self.model.fit(data=data, **kwargs)

    def predict(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Method responsible for generating predictions."""
        predict_df = self._filter_dataframe(df)
        data = self.generate_data(predict_df, predict=True)
        samples = self.model.predict(data=data, predict=True, **kwargs)
        # TODO: delete later
        # spread = samples["spread"].mean(axis=0)
        team1_win_prob = samples["team1_win_prob"].mean(axis=0)
        team2_win_prob = samples["team2_win_prob"].mean(axis=0)
        # team1_win_prob = samples["team1_win_prob"].mean(axis=0)
        predict_df = predict_df.with_columns(
            # spread=pl.Series(spread.tolist()),
            team1_win_prob=pl.Series(team1_win_prob.tolist()),
            team2_win_prob=pl.Series(team2_win_prob.tolist()),
        )
        x = predict_df.select(
            "ID",
            "date",
            "season",
            "team1_name",
            "team2_name",
            "team1_win_prob",
            "team2_win_prob",
            # "team1_score",
            # "team2_score",
            "team1_home",
            # pl.col("team2_score").sub("team1_score").alias("actual_spread"),
            "team1_win",
            "team2_win",
            "spread",
        )
        xx = x.with_columns(count=pl.cum_count("ID").over("ID")).pivot(
            "count",
            index="ID",
            # values=["spread", "team1_win_prob"],
            values=["team1_win_prob"],
        )
        xxx = df.with_columns(count=pl.cum_count("ID").over("ID")).pivot("count", index="ID", values=["spread"])
        print("ok")

    def generate_data(self, df: pl.DataFrame, *, predict: bool = False, **kwargs) -> Any:
        """Method responsible for generating data for training or prediction."""
        context_num = self.preprocessors["imputer"].transform(df.select(CONTEXT_NUMERIC_COLUMNS).to_numpy())
        context_num = self.preprocessors["scaler"].transform(context_num)
        context_ind = jnp.asarray(df.select(CONTEXT_INDICATOR_COLUMNS).to_numpy())
        context = jnp.concatenate([context_ind, jnp.asarray(context_num)], axis=1)

        return EloData(
            context=context,
            n_teams=len(self.preprocessors["team_encoder"].classes_),
            n_seasons=len(self.preprocessors["season_encoder"].classes_),
            season=jnp.asarray(self.preprocessors["season_encoder"].transform(df["season"])),
            team1=jnp.asarray(self.preprocessors["team_encoder"].transform(df["team1_id"])),
            team2=jnp.asarray(self.preprocessors["team_encoder"].transform(df["team2_id"])),
            team1_home=df["team1_home"].to_jax(),
            team2_home=df["team2_home"].to_jax(),
            spread=df["spread"].to_jax() if not predict else None,
            team1_win=df["team1_win"].to_jax() if not predict else None,
            team2_win=df["team2_win"].to_jax() if not predict else None,
        )

    def infer(self, *, save: bool = True, **kwargs) -> Any:
        """Method responsible for running inference."""
        coef_list: list[pl.DataFrame] = []
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
            if k in ["rating"]:
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
                "intercept",
                "alpha",
                "hca",
                "sigma_rating",
                "sigma_spread",
            ]:
                df = pl.DataFrame(summary_dict).with_columns(name=pl.lit(k))

            elif k in ["beta"]:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.Series([f"{k}_{x}" for x in CONTEXT_INDICATOR_COLUMNS + CONTEXT_NUMERIC_COLUMNS]),
                )
            else:
                logger.warning(f"Unhandled posterior sample key: {k}")
                continue

            coef_list.append(df)

        inference = EloInference(
            coef_df=pl.concat(coef_list),
            season_team_df=pl.concat(season_team_list, how="vertical_relaxed").join(teams, on="team_id", how="left"),
        )
        if save:
            inference.save(path=self.league)
        return inference
        # z = mu / sigma_spread
        # numpyro.sample("spread", dist.Normal(mu, sigma_spread), obs=data.spread)
        # numpyro.sample("team1_win_prob", dist.Bernoulli(probs=dist.Normal(0, 1).cdf(z)), obs=data.team1_win)
        # numpyro.sample("team2_win_prob", dist.Bernoulli(probs=dist.Normal(0, 1).cdf(-z)), obs=data.team2_win)
