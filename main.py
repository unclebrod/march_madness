"""Main entry point for March Madness modeling and analysis."""

import numpyro
import polars as pl
from cyclopts import App
from dotenv import find_dotenv, load_dotenv

from march_madness import dashboard, geocoding
from march_madness.loader import DataConstructor
from march_madness.log import logger
from march_madness.models.base import McmcParams, SviParams
from march_madness.models.elo import EloTrainer
from march_madness.models.ppp import PointsPerPossessionTrainer
from march_madness.settings import OUTPUT_DIR
from march_madness.trainer import Trainer
from march_madness.tuner import Tuner

TRAINER_MAP: dict[str, type[Trainer]] = {
    "ppp": PointsPerPossessionTrainer,
    "elo": EloTrainer,
}


app = App()


@app.command
def train(
    league: str = "M",
    model: str = "ppp",
    inference: str = "svi",
    num_samples: int = 2_000,
    mcmc_params: McmcParams | None = None,
    svi_params: SviParams | None = None,
    *,
    save: bool = True,
) -> None:
    logger.info(f"Training {model} model for {league} league with {inference} inference.")
    data_constructor = DataConstructor(league=league)
    trainer = TRAINER_MAP[model](league=league)
    df = data_constructor.load_game_team_box_scores()
    trainer.train(
        df=df,
        inference=inference,
        num_samples=num_samples,
        mcmc_params=mcmc_params,
        svi_params=svi_params,
    )
    # preds = trainer.predict(df=df)  # Generate predictions on training data for evaluation
    if save:
        trainer.save(path=league)
    logger.info("Training complete.")


@app.command
def infer(
    league: str = "M",
    model: str = "ppp",
    *,
    save: bool = True,
) -> None:
    logger.info(f"Running inference for {model} model in {league} league.")
    trainer = TRAINER_MAP[model].load(path=league)
    trainer.infer(save=save)
    logger.info("Inference complete.")


@app.command
def tune(
    league: str = "M",
    model: str = "ppp",
    inference: str = "svi",
    num_samples: int = 1_000,
    mcmc_params: McmcParams | None = None,
    svi_params: SviParams | None = None,
    n_trials: int = 50,
) -> None:
    logger.info(f"Tuning {model} model for {league} league.")
    data_constructor = DataConstructor(league=league)
    df = data_constructor.load_game_team_box_scores()
    trainer_cls = TRAINER_MAP[model]
    tuner = Tuner(
        df=df,
        trainer_cls=trainer_cls,
        league=league,
    )
    tuner.tune(
        inference=inference,
        num_samples=num_samples,
        n_trials=n_trials,
        mcmc_params=mcmc_params,
        svi_params=svi_params,
    )
    logger.info("Tuning complete.")


@app.command
def submit(
    season: int,
    model: str = "ppp",
) -> None:
    df_list: list[pl.DataFrame] = []
    for league in ["M", "W"]:
        data_constructor = DataConstructor(league=league)
        df = data_constructor.load_test_data(season=season)
        trainer = TRAINER_MAP[model].load(path=league)
        preds = trainer.predict(df=df)
        df_list.append(preds)
    pl.concat(df_list, how="vertical_relaxed").select(
        pl.col("ID"),
        pl.col("team1_win_prob").alias("Pred"),
    ).write_csv(OUTPUT_DIR / "final_submission.csv")


@app.command
def bracket(
    season: int | None = None,
    league: str = "M",
    *,
    save: bool = True,
) -> None:
    data_constructor = DataConstructor(league=league)
    data_constructor.generate_bracket(season=season, save=save)


@app.command
def streamlit() -> None:
    dashboard.main()


@app.command
def geocode() -> None:
    geocoding.geocode_cities()


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)

    app()
