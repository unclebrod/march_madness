"""Trainer class for March Madness modeling."""

from pathlib import Path
from typing import Any, Literal

import dill as pickle
import jax.numpy as jnp
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from march_madness.encoder import LabelEncoder, SequentialEncoder
from march_madness.loader import DataConfig, DataConstructor, DataLoader
from march_madness.model import MarchMadnessModel, ModelData
from march_madness.settings import OUTPUT_DIR


def get_quantiles(arr: jnp.ndarray, col_name: str = "value") -> pl.DataFrame:
    if np.ndim(arr) == 1:
        arr = arr.reshape(-1, 1)  # Ensure arr is 2D for quantile calculation
    return pl.DataFrame(
        data=np.quantile(arr, q=[0.025, 0.5, 0.975], axis=0),
        schema=[f"{col_name}_025", f"{col_name}_50", f"{col_name}_975"],
    )


def get_win_probs(predictive: dict[str, Any], team_str: str, opp_team_str: str) -> pl.DataFrame:
    return pl.DataFrame(
        data=np.array(
            (predictive[f"{team_str}_score"] > predictive[f"{opp_team_str}_score"]).mean(axis=0)
        ),
        schema=[f"{team_str}_win_prob"],
    )


class Trainer:
    def __init__(
        self,
        league: Literal["M", "W"] = "M",
        season_encoder: SequentialEncoder | None = None,
        team_encoder: LabelEncoder | None = None,
        coach_encoder: LabelEncoder | None = None,
        loc_encoder: LabelEncoder | None = None,
        imputer: SimpleImputer | None = None,
        scaler: StandardScaler | None = None,
        model: MarchMadnessModel | None = None,
        data_loader: DataLoader | None = None,
        data_config: DataConfig | None = None,
        data_constructor: DataConstructor | None = None,
    ):
        self.league = league
        self.season_encoder = SequentialEncoder() if not season_encoder else season_encoder
        self.team_encoder = LabelEncoder() if not team_encoder else team_encoder
        self.coach_encoder = LabelEncoder() if not coach_encoder else coach_encoder
        self.loc_encoder = LabelEncoder() if not loc_encoder else loc_encoder
        self.imputer = SimpleImputer(strategy="mean") if not imputer else imputer
        self.scaler = StandardScaler() if not scaler else scaler
        self.model = MarchMadnessModel() if not model else model
        self.data_loader = DataLoader(league=league) if not data_loader else data_loader
        self.data_config = DataConfig() if not data_config else data_config
        self.data_constructor = (
            DataConstructor(league=league) if not data_constructor else data_constructor
        )

    def train(self, df: pl.DataFrame, **kwargs):
        self.season_encoder.fit(df["season"])
        self.team_encoder.fit(df.select(["team1_id", "team2_id"]))
        self.coach_encoder.fit(df.select(["team1_coach", "team2_coach"]))
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
            for coef in ["defense_ppp", "offense_ppp", "pace", "net_rating"]:
                if coef == "net_rating":
                    samples[f"{coef}_{idx}"] = (
                        samples[f"offense_ppp_{idx}"] + samples[f"defense_ppp_{idx}"]
                    )
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

    def submit(
        self, suffix: str | None = None, *, save: bool = True
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
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
        score_df = get_quantiles(samples["team1_score"], col_name="team1_score").with_columns(
            team1_score_mean=pl.Series(
                "team1_score_mean", samples["team1_score"].mean(axis=0).tolist()
            )
        )
        possession_df = get_quantiles(samples["possessions"], col_name="team1_possessions")
        results_df = (
            df.with_columns(
                team1_pred=pl.Series(name="Pred", values=win_probs.tolist()),
                overtime=pl.Series(
                    name="overtime", values=jnp.concat([overtime, overtime]).tolist()
                ),
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
                        team2_score_mean=pl.col("team1_score_mean"),
                        team2_possessions_025=pl.col("team1_possessions_025"),
                        team2_possessions_50=pl.col("team1_possessions_50"),
                        team2_possessions_975=pl.col("team1_possessions_975"),
                        team2_pred=pl.col("team1_pred"),
                    ),
                ],
                how="horizontal",
            )
            .with_columns(
                Pred=pl.col("team1_pred")
                .truediv(pl.col("team1_pred").add(pl.col("team2_pred")))
                .alias("Pred"),
            )
            .with_columns(
                OppPred=pl.lit(1).sub("Pred"),
                spread=pl.col("team2_score_50").sub(
                    pl.col("team1_score_50")
                ),  # negative means team 1 is favored
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
                    "team1_score_mean",
                    "team2_score_mean",
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
        if suffix == "manual":
            # TODO: could make better logic for this?
            if self.league == "M":
                # Make Duke (1181) the winner of all games
                all_results_wide = all_results_wide.with_columns(
                    Pred=pl.when(pl.col("team1_id").eq(1181)).then(1).otherwise(pl.col("Pred")),
                    OppPred=pl.when(pl.col("team1_id").eq(1181))
                    .then(0)
                    .otherwise(pl.col("OppPred")),
                ).with_columns(
                    Pred=pl.when(pl.col("team2_id").eq(1181)).then(0).otherwise(pl.col("Pred")),
                    OppPred=pl.when(pl.col("team2_id").eq(1181))
                    .then(1)
                    .otherwise(pl.col("OppPred")),
                )
            else:
                # Let all higher seeds advance in round 1
                tourney_seeds = (
                    self.data_loader.load_data(self.data_config.ncaa_tourney_seeds)
                    .filter(pl.col("Season").eq(2025))
                    .with_columns(pl.col("Seed").str.slice(0, 3))
                    .drop("Season")
                )
                tourney_slots = (
                    self.data_loader.load_data(self.data_config.ncaa_tourney_slots)
                    .filter(pl.col("Season").eq(2025))
                    .drop("Season")
                )
                tourney_matchups = (
                    tourney_slots.filter(pl.col("Slot").str.starts_with("R1"))
                    .join(
                        tourney_seeds.rename({"Seed": "StrongSeed", "TeamID": "StrongTeamID"}),
                        on="StrongSeed",
                        how="left",
                    )
                    .join(
                        tourney_seeds.rename({"Seed": "WeakSeed", "TeamID": "WeakTeamID"}),
                        on="WeakSeed",
                        how="left",
                    )
                    .with_columns(
                        Pred=pl.lit(1),
                        OppPred=pl.lit(0),
                    )
                )
                matchups = pl.concat(
                    [
                        tourney_matchups,
                        tourney_matchups.with_columns(
                            WeakTeamID=pl.col("StrongTeamID"),
                            StrongTeamID=pl.col("WeakTeamID"),
                            WeakSeed=pl.col("StrongSeed"),
                            StrongSeed=pl.col("WeakSeed"),
                            Pred=pl.lit(0),
                            OppPred=pl.lit(1),
                        ),
                    ]
                ).select(
                    team1_id=pl.col("StrongTeamID"),
                    team2_id=pl.col("WeakTeamID"),
                    Pred=pl.col("Pred"),
                    OppPred=pl.col("OppPred"),
                )
                # Make UConn (3163) and South Carolina (3376) the winner of all games, except when they play each other
                all_results_wide = (
                    all_results_wide.with_columns(
                        Pred=pl.when(pl.col("team1_id").eq(3163) & pl.col("team2_id").ne(3376))
                        .then(1)
                        .otherwise(pl.col("Pred")),
                        OppPred=pl.when(pl.col("team1_id").eq(3163) & pl.col("team2_id").ne(3376))
                        .then(0)
                        .otherwise(pl.col("OppPred")),
                    )
                    .with_columns(
                        Pred=pl.when(pl.col("team1_id").eq(3376) & pl.col("team2_id").ne(3163))
                        .then(1)
                        .otherwise(pl.col("Pred")),
                        OppPred=pl.when(pl.col("team1_id").eq(3376) & pl.col("team2_id").ne(3163))
                        .then(0)
                        .otherwise(pl.col("OppPred")),
                    )
                    .with_columns(
                        Pred=pl.when(pl.col("team2_id").eq(3163) & pl.col("team1_id").ne(3376))
                        .then(0)
                        .otherwise(pl.col("Pred")),
                        OppPred=pl.when(pl.col("team2_id").eq(3163) & pl.col("team1_id").ne(3376))
                        .then(1)
                        .otherwise(pl.col("OppPred")),
                    )
                    .with_columns(
                        Pred=pl.when(pl.col("team2_id").eq(3376) & pl.col("team1_id").ne(3163))
                        .then(0)
                        .otherwise(pl.col("Pred")),
                        OppPred=pl.when(pl.col("team2_id").eq(3376) & pl.col("team1_id").ne(3163))
                        .then(1)
                        .otherwise(pl.col("OppPred")),
                    )
                    .update(
                        matchups,
                        on=["team1_id", "team2_id"],
                        how="left",
                    )
                )

        submission_df = all_results_wide.select(
            pl.col("ID"),
            pl.col("Pred"),
        )

        if save:
            sfx = f"_{suffix}" if suffix else ""
            all_results_wide.write_csv(OUTPUT_DIR / f"{self.league}/results{sfx}.csv")
            submission_df.write_csv(OUTPUT_DIR / f"{self.league}/submission{sfx}.csv")
            with Path(OUTPUT_DIR / f"{self.league}/samples{sfx}.pkl").open("wb") as f:
                pickle.dump(samples, f)

        return all_results_wide, submission_df

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
