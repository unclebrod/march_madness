import jax.numpy as jnp
import numpyro
import polars as pl
from sklearn.preprocessing import StandardScaler

from march_madness.encoder import LabelEncoder
from march_madness.loading import load_box_scores
from march_madness.model import MarchMadnessModel, ModelData

if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    numpyro.set_platform("cpu")

    # Load data
    box_scores = load_box_scores(league="M")

    # Run model
    season_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    scaler = StandardScaler()

    # preprocess
    season_encoder.fit(box_scores["season"])
    team_encoder.fit(box_scores.select(["team1_id", "team2_id"]))

    # scale_columns = [
    #     "team1_ppp",
    #     "team2_ppp",
    #     "team1_poss",
    #     "team2_poss",
    #     "team1_pace",
    #     "team2_pace",
    # ]
    # scaled_arr = jnp.array(scaler.fit_transform(box_scores.select(scale_columns)))

    context = box_scores.select(
        pl.col("is_conf_tourney"),
        pl.col("is_ncaa_tourney"),
        pl.lit(1).alias("intercept"),
    ).to_jax()

    data = ModelData(
        context=context,
        n=box_scores.shape[0],
        n_teams=len(team_encoder.classes_) + 1,
        n_seasons=len(season_encoder.classes_) + 1,
        season=jnp.array(season_encoder.transform(box_scores.get_column("season"))),
        team1=jnp.array(team_encoder.transform(box_scores.get_column("team1_id"))),
        team2=jnp.array(team_encoder.transform(box_scores.get_column("team2_id"))),
        # team1_ppp=scaled_arr[:, scale_columns.index("team1_ppp")],
        # team2_ppp=scaled_arr[:, scale_columns.index("team2_ppp")],
        # team1_poss=scaled_arr[:, scale_columns.index("team1_poss")],
        # team2_poss=scaled_arr[:, scale_columns.index("team2_poss")],
        # team1_pace=scaled_arr[:, scale_columns.index("team1_pace")],
        # team2_pace=scaled_arr[:, scale_columns.index("team2_pace")],
        # team1_ppp=box_scores["team1_ppp"].to_jax(),
        # team2_ppp=box_scores["team2_ppp"].to_jax(),
        # team1_poss=box_scores["team1_poss"].to_jax(),
        # team2_poss=box_scores["team2_poss"].to_jax(),
        avg_poss=box_scores["avg_poss"].to_jax(),
        # team1_pace=box_scores["team1_pace"].to_jax(),
        # team2_pace=box_scores["team2_pace"].to_jax(),
        # avg_pace=box_scores["avg_pace"].to_jax(),
        is_neutral=box_scores["is_neutral"].to_jax(),
        is_team1_home=box_scores["is_team1_home"].to_jax(),
        is_team2_home=box_scores["is_team2_home"].to_jax(),
        minutes=box_scores["minutes"].to_jax(),
        team1_score=box_scores["team1_score"].to_jax(),
        team2_score=box_scores["team2_score"].to_jax(),
    )

    model = MarchMadnessModel()
    model.fit(
        data,
        inference="svi",
        # num_warmup=10,
        # num_samples=10,
        # num_chains=4,
        num_steps=10_000,
    )

    print("okay")
