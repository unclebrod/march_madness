"""Points per possession probabilistic model."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.nn import softplus
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from march_madness.encoder import LabelEncoder, SequentialEncoder
from march_madness.loader import DataConfig
from march_madness.log import logger
from march_madness.models.base import BaseNumpyroModel
from march_madness.settings import OUTPUT_DIR
from march_madness.trainers.base_trainer import BaseTrainer
from march_madness.utils import summarize_samples

# recency = game_number / max_game_number_per_season_team
# weights = jnp.exp(k * recency)

# linear
# weights = 1.0 + alpha * recency

# numpyro.factor(name, log_weight) adds an arbitrary log-density term to the joint log-probability:
# with numpyro.plate("obs", N):
#     mu = (
#         offense[season_id, team_id]
#         - defense[season_id, opp_id]
#     )

#     logp = dist.Normal(mu, sigma).log_prob(y)
#     numpyro.factor("recency_weighted_ll", weights * logp)
CONTEXT_INDICATOR_COLUMNS = [
    "is_ncaa_tourney",
    "is_conf_tourney",
    "team1_b2b",
    "team1_2in3",
    "team1_3in4",
    "team1_4in5",
    "team2_b2b",
    "team2_2in3",
    "team2_3in4",
    "team2_4in5",
]

CONTEXT_NUMERIC_COLUMNS = [
    "days_into_season_norm",
    "days_into_season_norm_sq",
    "team1_log_travel_distance_diff",
    "team1_log_elevation_diff",
    "team2_log_travel_distance_diff",
    "team2_log_elevation_diff",
]


@dataclass
class PointsPerPossessionData:
    context: jnp.array  # fixed effects (tournament, travel, rest, elevation, etc.)

    n_teams: int | None  # number of unique teams
    n_seasons: int | None  # number of unique seasons
    n_coaches: int | None  # number of unique coaches

    season: jnp.ndarray  # season encoding
    team1: jnp.ndarray  # team1 encoding
    team2: jnp.ndarray  # team2 encoding
    team1_coach: jnp.ndarray  # team1 coach encoding
    team2_coach: jnp.ndarray  # team2 coach encoding

    minutes: jnp.ndarray  # game minutes
    team1_home: jnp.ndarray  # is team1 the home team
    team2_home: jnp.ndarray  # is team2 the home team

    avg_poss: jnp.ndarray | None  # average possessions of the game
    team1_score: jnp.ndarray | None  # team1's score
    team2_score: jnp.ndarray | None  # team2's score


@dataclass
class PointsPerPossessionInference:
    coef_df: pl.DataFrame
    season_coef_df: pl.DataFrame
    team_coef_df: pl.DataFrame
    season_team_coef_df: pl.DataFrame
    coach_coef_df: pl.DataFrame
    game_outputs_df: pl.DataFrame

    def save(self, path: str = "M") -> None:
        self.coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/coefs.csv")
        self.season_coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/season_coefs.csv")
        self.team_coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/team_coefs.csv")
        self.season_team_coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/season_team_coefs.csv")
        self.coach_coef_df.write_csv(OUTPUT_DIR / f"{path}/ppp/coach_coefs.csv")
        self.game_outputs_df.write_csv(OUTPUT_DIR / f"{path}/ppp/game_outputs.csv")


class PointsPerPossessionModel(BaseNumpyroModel):
    name = "ppp"

    def model(
        self,
        data: PointsPerPossessionData,
        beta_ppp_std: float = 0.05,
        beta_pace_std: float = 0.05,
        offense_global_mean_std: float = 0.3,
        defense_global_mean_std: float = 0.1,
        pace_global_mean_std: float = 0.3,
        sigma_offense_season_rate: float = 10.0,
        sigma_defense_season_rate: float = 10.0,
        sigma_pace_season_rate: float = 10.0,
        sigma_offense_team_rate: float = 1.0,
        sigma_defense_team_rate: float = 1.0,
        sigma_pace_team_rate: float = 1.0,
        coach_std: float = 0.01,
        hca_team_std: float = 0.02,
        phi_score_rate: float = 0.05,
        **kwargs,
    ) -> None:
        # TODO: consider adding weighting so end of season games are more important
        # TODO: latent factor analysis?
        # ---- Fixed effects matrices ----
        n_context = data.context.shape[1]

        beta_ppp = numpyro.sample("beta_ppp", dist.Normal(0, beta_ppp_std).expand((n_context,)))
        beta_pace = numpyro.sample("beta_pace", dist.Normal(0, beta_pace_std).expand((n_context,)))

        fixed_effects_ppp = jnp.dot(data.context, beta_ppp)
        fixed_effects_pace = jnp.dot(data.context, beta_pace)

        # ---- Global means & standard deviations for each offense, defense, and pace ----
        offense_global_mean = numpyro.sample("offense_global_mean", dist.Normal(0, offense_global_mean_std))
        defense_global_mean = numpyro.sample("defense_global_mean", dist.Normal(0, defense_global_mean_std))
        pace_global_mean = numpyro.sample("pace_global_mean", dist.Normal(1.5, pace_global_mean_std))

        sigma_offense_season = numpyro.sample("sigma_offense_season", dist.Exponential(sigma_offense_season_rate))
        sigma_defense_season = numpyro.sample("sigma_defense_season", dist.Exponential(sigma_defense_season_rate))
        sigma_pace_season = numpyro.sample("sigma_pace_season", dist.Exponential(sigma_pace_season_rate))

        sigma_offense_team = numpyro.sample("sigma_offense_team", dist.HalfNormal(sigma_offense_team_rate))
        sigma_defense_team = numpyro.sample("sigma_defense_team", dist.HalfNormal(sigma_defense_team_rate))
        sigma_pace_team = numpyro.sample("sigma_pace_team", dist.HalfNormal(sigma_pace_team_rate))

        # ---- Season-specific means for each offense, defense, and pace ----
        offense_season_rw = numpyro.sample(
            "offense_season_rw",
            dist.GaussianRandomWalk(sigma_offense_season, num_steps=data.n_seasons),
        )
        defense_season_rw = numpyro.sample(
            "defense_season_rw",
            dist.GaussianRandomWalk(sigma_defense_season, num_steps=data.n_seasons),
        )
        pace_season_rw = numpyro.sample(
            "pace_season_rw",
            dist.GaussianRandomWalk(sigma_pace_season, num_steps=data.n_seasons),
        )

        # Center season means to assist with identifiability
        offense_season_mean = offense_global_mean + offense_season_rw - offense_season_rw.mean()
        defense_season_mean = defense_global_mean + defense_season_rw - defense_season_rw.mean()
        pace_season_mean = pace_global_mean + pace_season_rw - pace_season_rw.mean()

        # ---- Team-specific offense, defense, and pace ratings as deviations from the mean ----
        with numpyro.plate("teams", data.n_teams):
            offense_team_rw = numpyro.sample(
                "offense_team_rw",
                dist.GaussianRandomWalk(sigma_offense_team, num_steps=data.n_seasons),
            )
            defense_team_rw = numpyro.sample(
                "defense_team_rw",
                dist.GaussianRandomWalk(sigma_defense_team, num_steps=data.n_seasons),
            )
            pace_team_rw = numpyro.sample(
                "pace_team_rw",
                dist.GaussianRandomWalk(sigma_pace_team, num_steps=data.n_seasons),
            )

        # Center team deltas to assist with identifiability
        offense_team_delta = offense_team_rw - offense_team_rw.mean(axis=0, keepdims=True)
        defense_team_delta = defense_team_rw - defense_team_rw.mean(axis=0, keepdims=True)
        pace_team_delta = pace_team_rw - pace_team_rw.mean(axis=0, keepdims=True)

        # ---- Coach effects ----
        with numpyro.plate("coaches", data.n_coaches):
            coach = numpyro.sample("coach", dist.Normal(0, coach_std))

        # ---- Home court advantage, per team ----
        hca_team_mu = numpyro.sample("hca_team_mu", dist.Normal(0.02, 0.01))
        with numpyro.plate("teams", data.n_teams):
            hca_team = numpyro.sample("hca_team", dist.Normal(hca_team_mu, hca_team_std))

        # ---- Combine offense, defense, and pace components ----
        offense = offense_season_mean + offense_team_delta
        defense = defense_season_mean + defense_team_delta
        pace = pace_season_mean + pace_team_delta

        # ---- Team-level ratings for offense, defense, and pace ----
        team1_offense = offense[data.team1, data.season]
        team2_offense = offense[data.team2, data.season]
        team1_defense = defense[data.team1, data.season]
        team2_defense = defense[data.team2, data.season]
        team1_pace = pace[data.team1, data.season]
        team2_pace = pace[data.team2, data.season]

        # ---- Coach effects ----
        team1_coach = coach[data.team1_coach]
        team2_coach = coach[data.team2_coach]

        # ---- Expected points per possession for each team ----
        team1_ppp = softplus(
            team1_offense
            - team2_defense
            + team1_coach
            - team2_coach
            + fixed_effects_ppp
            + (hca_team[data.team1] * data.team1_home)
        )
        team2_ppp = softplus(
            team2_offense
            - team1_defense
            + team2_coach
            - team1_coach
            + fixed_effects_ppp
            + (hca_team[data.team2] * data.team2_home)
        )

        # ---- Expected pace, possessions, and scores ----
        pace = softplus(((team1_pace + team2_pace) / 2) + fixed_effects_pace)
        possessions = softplus(pace * data.minutes)
        team1_score = softplus(team1_ppp * possessions)
        team2_score = softplus(team2_ppp * possessions)

        # ---- Likelihood functions for possessions and scores ----
        phi_score = numpyro.sample("phi_score", dist.Exponential(phi_score_rate))

        numpyro.sample("possessions", dist.Poisson(possessions), obs=data.avg_poss)
        numpyro.sample("team1_score", dist.NegativeBinomial2(team1_score, phi_score), obs=data.team1_score)
        numpyro.sample("team2_score", dist.NegativeBinomial2(team2_score, phi_score), obs=data.team2_score)

        # ---- Deterministic outputs for inference ----
        numpyro.deterministic("team1_ppp", team1_ppp)
        numpyro.deterministic("team2_ppp", team2_ppp)
        numpyro.deterministic("pace", pace)
        numpyro.deterministic("spread", team2_score - team1_score)
        numpyro.deterministic("total", team1_score + team2_score)
        numpyro.deterministic("team1_win_prob", (team1_score / (team1_score + team2_score)).astype(jnp.float32))
        numpyro.deterministic("team2_win_prob", (team2_score / (team1_score + team2_score)).astype(jnp.float32))


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

    def _filter_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        filtered_df = df.filter(
            pl.col("team1_score").is_not_null(),
            pl.col("team2_score").is_not_null(),
            pl.col("avg_poss").is_not_null(),
            pl.col("season").is_not_null(),
            pl.col("team1_id").is_not_null(),
            pl.col("team2_id").is_not_null(),
        )
        logger.info(
            f"Removed {len(df) - len(filtered_df)} rows with missing values ({len(filtered_df) / len(df):.2%})."
        )
        return filtered_df

    def train(self, df: pl.DataFrame, **kwargs) -> None:
        train_df = self._filter_dataframe(df)

        context_num = train_df.select(CONTEXT_NUMERIC_COLUMNS).to_numpy()
        context_num_imputed = self.preprocessors["imputer"].fit_transform(context_num)

        self.preprocessors["scaler"].fit(context_num_imputed)
        self.preprocessors["season_encoder"].fit(train_df["season"])
        self.preprocessors["team_encoder"].fit(train_df.select(["team1_id", "team2_id"]))
        self.preprocessors["coach_encoder"].fit(train_df.select(["team1_coach", "team2_coach"]))

        data = self.generate_data(train_df, predict=False)
        self.model.fit(data=data, **kwargs)

    def predict(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        predict_df = self._filter_dataframe(df)
        data = self.generate_data(predict_df, predict=True)
        samples = self.model.predict(data=data, **kwargs)
        print("ok")

    def generate_data(
        self,
        df: pl.DataFrame,
        *,
        predict: bool = False,
        **kwargs,
    ) -> PointsPerPossessionData:
        context_num = self.preprocessors["imputer"].transform(df.select(CONTEXT_NUMERIC_COLUMNS).to_numpy())
        context_num = self.preprocessors["scaler"].transform(context_num)
        context_ind = jnp.asarray(df.select(CONTEXT_INDICATOR_COLUMNS).to_numpy())
        context = jnp.concatenate([context_ind, jnp.asarray(context_num)], axis=1)

        return PointsPerPossessionData(
            context=context,
            n_teams=len(self.preprocessors["team_encoder"].classes_),
            n_seasons=len(self.preprocessors["season_encoder"].classes_),
            n_coaches=len(self.preprocessors["coach_encoder"].classes_),
            season=self.preprocessors["season_encoder"].transform(df["season"]),
            team1=self.preprocessors["team_encoder"].transform(df["team1_id"]),
            team2=self.preprocessors["team_encoder"].transform(df["team2_id"]),
            team1_coach=self.preprocessors["coach_encoder"].transform(df["team1_coach"]),
            team2_coach=self.preprocessors["coach_encoder"].transform(df["team2_coach"]),
            minutes=df["minutes"].to_jax(),
            team1_home=df["team1_home"].to_jax(),
            team2_home=df["team2_home"].to_jax(),
            avg_poss=df["avg_poss"].to_jax() if not predict else None,
            team1_score=df["team1_score"].to_jax() if not predict else None,
            team2_score=df["team2_score"].to_jax() if not predict else None,
        )

    def infer(self, *, save: bool = True, **kwargs) -> PointsPerPossessionInference:
        """Run inference on the trained model."""
        coef_list: list[pl.DataFrame] = []
        season_list: list[pl.DataFrame] = []
        team_list: list[pl.DataFrame] = []
        season_team_list: list[pl.DataFrame] = []
        coach_list: list[pl.DataFrame] = []
        game_list: list[pl.DataFrame] = []

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

            if k in ["coach"]:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.lit(k),
                    coach_name=pl.Series(self.preprocessors["coach_encoder"].classes_),
                )
                coach_list.append(df)
                continue

            if k in ["hca_team"]:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.lit(k),
                    team_id=pl.Series(team_classes),
                )
                team_list.append(df)
                continue

            if k in [
                "team1_ppp",
                "team2_ppp",
                "pace",
                "spread",
                "total",
                "team1_win_prob",
                "team2_win_prob",
            ]:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.lit(k),
                )
                game_list.append(df)
                continue

            if any(x in k for x in ["sigma", "phi", "global_mean", "hca_team_mu"]):
                df = pl.DataFrame(summary_dict).with_columns(name=pl.lit(k))
            elif "beta" in k:
                df = pl.DataFrame(summary_dict).with_columns(
                    name=pl.Series([f"{k}_{x}" for x in CONTEXT_INDICATOR_COLUMNS + CONTEXT_NUMERIC_COLUMNS]),
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
            coach_coef_df=pl.concat(coach_list),
            game_outputs_df=pl.concat(game_list),
        )
        if save:
            inference.save(path=self.league)
        return inference
