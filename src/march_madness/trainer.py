from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import polars as pl

from march_madness import OUTPUT_DIR
from march_madness.encoder import LabelEncoder
from march_madness.loader import DataConfig, DataConstructor, DataLoader
from march_madness.model import MarchMadnessModel, ModelData


def get_quantiles(arr: jnp.ndarray, col_name: str = "value") -> pl.DataFrame:
    if np.ndim(arr) == 1:
        arr = arr.reshape(-1, 1)  # Ensure arr is 2D for quantile calculation
    return pl.DataFrame(
        data=np.quantile(arr, q=[0.025, 0.5, 0.975], axis=0),
        schema=[f"{col_name}_025", f"{col_name}_50", f"{col_name}_975"],
    )


def get_win_probs(predictive: dict[str, Any], team_str: str, opp_team_str: str) -> pl.DataFrame:
    return pl.DataFrame(
        data=np.array((predictive[f"{team_str}_score"] > predictive[f"{opp_team_str}_score"]).mean(axis=0)),
        schema=[f"{team_str}_win_prob"],
    )


class Trainer:
    def __init__(
        self,
        league: Literal["M", "W"] = "M",
        season_encoder: LabelEncoder | None = None,
        team_encoder: LabelEncoder | None = None,
        loc_encoder: LabelEncoder | None = None,
        model: MarchMadnessModel | None = None,
        data_loader: DataLoader | None = None,
        data_config: DataConfig | None = None,
        data_constructor: DataConstructor | None = None,
    ):
        self.league = league
        self.season_encoder = LabelEncoder() if not season_encoder else season_encoder
        self.team_encoder = LabelEncoder() if not team_encoder else team_encoder
        self.loc_encoder = LabelEncoder() if not loc_encoder else loc_encoder
        self.model = MarchMadnessModel() if not model else model
        self.data_loader = DataLoader(league=league) if not data_loader else data_loader
        self.data_config = DataConfig() if not data_config else data_config
        self.data_constructor = DataConstructor(league=league) if not data_constructor else data_constructor

    def train(self, df: pl.DataFrame, **kwargs):
        self.season_encoder.fit(df["season"])
        self.team_encoder.fit(df.select(["team1_id", "team2_id"]))
        self.loc_encoder.fit(df.select(["team1_loc"]))
        data = self.generate_data(df=df)
        self.model.fit(data=data, **kwargs)

    def predict(self, df: pl.DataFrame):
        data = self.generate_data(df=df, predict=True)
        samples = self.model.predict(data=data)

        # get lower and upper bounds for our posterior predictions
        df_list = []
        for key in ["possessions", "team1_score"]:
            df_list.append(get_quantiles(samples, key))

        # for t1, t2 in zip(["team1", "team2"], ["team2", "team1"], strict=False):
        #     df_list.append(get_win_probs(samples, t1, t2))

        # combine into results
        return pl.concat(df_list, how="horizontal")

    def submit(self, *, save: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = self.data_constructor.generate_test_data()
        data = self.generate_data(df=df, predict=True)
        samples = self.model.predict(data=data)

        team_meta = self.data_loader.load_data(self.data_config.teams).select(
            pl.col("TeamID").alias("team_id"),
            pl.col("TeamName").alias("team_name"),
        )

        n_games = int(df.shape[0] / 2)
        team1_arr = samples["team1_score"][:, :n_games]
        team2_arr = samples["team1_score"][:, n_games:]
        team1_wins = (team1_arr > team2_arr).mean(axis=0)
        team2_wins = (team2_arr > team1_arr).mean(axis=0)
        overtime = (team1_arr == team2_arr).mean(axis=0)
        win_probs = jnp.concat([team1_wins, team2_wins])
        score_df = get_quantiles(samples["team1_score"], col_name="team1_score")
        possession_df = get_quantiles(samples["possessions"], col_name="team1_possessions")
        results_df = (
            df.with_columns(
                team1_pred=pl.Series(name="Pred", values=win_probs.tolist()),
                overtime=pl.Series(name="overtime", values=jnp.concat([overtime, overtime]).tolist()),
            )
            .join(
                team_meta.rename({"team_id": "team1_id", "team_name": "team1_name"}),
                how="left",
                on="team1_id",
            )
            .join(
                team_meta.rename({"team_id": "team2_id", "team_name": "team2_name"}),
                how="left",
                on="team2_id",
            )
        )
        all_results = pl.concat([results_df, score_df, possession_df], how="horizontal")
        all_results_wide = (
            pl.concat(
                [
                    all_results[:n_games],
                    all_results[n_games:].select(
                        team2_score_025=pl.col("team1_score_025"),
                        team2_score_50=pl.col("team1_score_50"),
                        team2_score_975=pl.col("team1_score_975"),
                        team2_possessions_025=pl.col("team1_possessions_025"),
                        team2_possessions_50=pl.col("team1_possessions_50"),
                        team2_possessions_975=pl.col("team1_possessions_975"),
                        team2_pred=pl.col("team1_pred"),
                    ),
                ],
                how="horizontal",
            )
            .with_columns(
                Pred=pl.col("team1_pred").truediv(pl.col("team1_pred").add(pl.col("team2_pred"))).alias("Pred"),
            )
            .with_columns(
                OppPred=pl.lit(1).sub("Pred"),
                spread=pl.col("team2_score_50").sub(pl.col("team1_score_50")),  # negative means team 1 is favored
            )
            .select(
                [
                    "ID",
                    "season",
                    "team1_id",
                    "team1_name",
                    "team2_id",
                    "team2_name",
                    "Pred",
                    "OppPred",
                    "team1_score_025",
                    "team2_score_025",
                    "team1_score_50",
                    "team2_score_50",
                    "team1_score_975",
                    "team2_score_975",
                    "team1_possessions_025",
                    "team2_possessions_025",
                    "team1_possessions_50",
                    "team2_possessions_50",
                    "team1_possessions_975",
                    "team2_possessions_975",
                    "team1_pred",
                    "team2_pred",
                    "overtime",
                ]
            )
        )
        submission_df = all_results_wide.select(
            pl.col("ID"),
            pl.col("Pred"),
        )

        if save:
            all_results_wide.write_csv(OUTPUT_DIR / f"{self.league}/results.csv")
            submission_df.write_csv(OUTPUT_DIR / f"{self.league}/submission.csv")

        return all_results_wide, submission_df

    def generate_data(self, df: pl.DataFrame, *, predict: bool = False) -> ModelData:
        context = df.select(
            pl.col("is_conf_tourney"),
            pl.col("is_ncaa_tourney"),
        ).to_jax()

        return ModelData(
            context=context,
            n=df.shape[0],
            n_teams=len(self.team_encoder.classes_),
            n_seasons=len(self.season_encoder.classes_),
            season=jnp.array(self.season_encoder.transform(df.get_column("season"))),
            team1=jnp.array(self.team_encoder.transform(df.get_column("team1_id"))),
            team2=jnp.array(self.team_encoder.transform(df.get_column("team2_id"))),
            minutes=df["minutes"].to_jax(),
            avg_poss=None if predict else df["avg_poss"].to_jax(),
            team1_score=None if predict else df["team1_score"].to_jax(),
            loc=jnp.array(self.loc_encoder.transform(df.get_column("team1_loc"))),
            is_team1_home=df["is_team1_home"].to_jax(),
            is_team2_home=df["is_team2_home"].to_jax(),
            is_neutral=df["is_neutral"].to_jax(),
        )

    def infer(self, *, save: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
        samples = self.model.samples

        seasons = self.season_encoder.classes_
        teams = self.team_encoder.classes_
        # locs = self.loc_encoder.classes_

        coef_cols = ["coefficient", "value_025", "value_50", "value_975"]

        coef_list = []
        for beta in ["beta_ppp"]:
            for idx, context in enumerate(["is_conf_tourney", "is_ncaa_tourney"]):
                beta_df = get_quantiles(samples[beta][:, idx]).with_columns(
                    coefficient=pl.lit(f"{beta}_{context}"),
                )
                coef_list.append(beta_df)

        for coef in [
            "hca",
            "mu_defense_ppp",
            "mu_offense_ppp",
            "mu_pace",
            "phi_score",
            "sigma_defense_ppp",
            "sigma_offense_ppp",
            "sigma_pace",
            "sigma_poss",
        ]:
            beta_df = get_quantiles(samples[coef]).with_columns(
                coefficient=pl.lit(coef),
            )
            coef_list.append(beta_df)

        coef_df = pl.concat(coef_list, how="vertical_relaxed").select(coef_cols)

        team_meta = self.data_loader.load_data(self.data_config.teams).select(
            pl.col("TeamID").alias("team_id"),
            pl.col("TeamName").alias("team_name"),
        )
        ratings_list = []
        for idx, season in enumerate(seasons):
            for coef in ["defense_ppp", "offense_ppp", "pace"]:
                quantiles_df = get_quantiles(samples[f"{coef}_{idx}"]).with_columns(
                    coefficient=pl.lit(coef),
                    season=pl.lit(season),
                    team_id=pl.Series(name="teams", values=teams),
                )
                ratings_list.append(quantiles_df)

        ratings_df = (
            pl.concat(ratings_list, how="vertical_relaxed")
            .join(
                team_meta,
                on="team_id",
                how="left",
            )
            .select(["team_id", "team_name", "season"] + coef_cols)
        )

        if save:
            coef_df.write_csv(OUTPUT_DIR / f"{self.league}/coefs.csv")
            ratings_df.write_csv(OUTPUT_DIR / f"{self.league}/ratings.csv")

        return coef_df, ratings_df

    def save(self, path: str | None = None) -> None:
        path = path or self.league
        self.season_encoder.save(prefix="season_", path=path)
        self.team_encoder.save(prefix="team_", path=path)
        self.loc_encoder.save(prefix="loc_", path=path)
        self.model.save(path=path)

    @classmethod
    def load(cls, league: Literal["M", "W"] = "M"):
        return cls(
            league=league,
            season_encoder=LabelEncoder.load(prefix="season_", path=league),
            team_encoder=LabelEncoder.load(prefix="team_", path=league),
            loc_encoder=LabelEncoder.load(prefix="loc_", path=league),
            model=MarchMadnessModel.load(path=league),
        )
