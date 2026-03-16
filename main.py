"""Main entry point for March Madness modeling and analysis."""

import numpyro
from cyclopts import App
from dotenv import find_dotenv, load_dotenv

from march_madness import dashboard, geocoding
from march_madness.loader import DataConstructor
from march_madness.models.base import McmcParams, SviParams
from march_madness.models.elo import EloTrainer
from march_madness.models.ppp import PointsPerPossessionTrainer
from march_madness.trainer import Trainer

TRAINER_MAP = {
    "ppp": PointsPerPossessionTrainer,
    "elo": EloTrainer,
}


app = App()


@app.command
def train(
    league: str = "M",
    inference: str = "svi",
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 4,
    learning_rate: float = 0.01,
    num_steps: int = 25_000,
    *,
    save: bool = True,
) -> None:
    data_constructor = DataConstructor(league=league)
    trainer = Trainer(league=league)
    box_scores = data_constructor.load_game_team_box_scores()
    trainer.train(
        df=box_scores,
        inference=inference,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )
    if save:
        trainer.save(path=league)


@app.command
def predict(
    league: str = "M",
) -> None:
    data_constructor = DataConstructor(league=league)
    trainer = Trainer.load(league=league)
    box_scores = data_constructor.load_game_team_box_scores()
    trainer.predict(
        df=box_scores,
    )


@app.command
def infer(
    league: str = "M",
    *,
    save: bool = True,
) -> None:
    trainer = Trainer.load(league=league)
    trainer.infer(
        save=save,
    )


@app.command
def submit(
    league: str = "M",
    suffix: str | None = None,
    *,
    save: bool = True,
) -> None:
    trainer = Trainer.load(league=league)
    trainer.submit(
        save=save,
        suffix=suffix,
    )


@app.command
def final(
    suffix: str | None = None,
) -> None:
    data_constructor = DataConstructor(league="M")
    data_constructor.get_final_submission(suffix=suffix)


@app.command
def bracket(
    league: str = "M",
    suffix: str | None = None,
    *,
    save: bool = True,
) -> None:
    data_constructor = DataConstructor(league=league)
    data_constructor.generate_bracket(
        suffix=suffix,
        save=save,
    )


@app.command
def streamlit() -> None:
    dashboard.main()


@app.command
def geocode() -> None:
    geocoding.geocode_cities()


# TODO: eventually make this the main method after testing
@app.command
def ppp(
    league: str = "M",
    model: str = "ppp",
    inference: str = "svi",
    num_samples: int = 1_000,
    mcmc_params: McmcParams | None = None,
    svi_params: SviParams | None = None,
    *,
    save: bool = True,
) -> None:
    data_constructor = DataConstructor(league=league)
    trainer = TRAINER_MAP[model](league=league)
    box_scores = data_constructor.load_game_team_box_scores()
    trainer.train(
        df=box_scores,
        inference=inference,
        num_samples=num_samples,
        mcmc_params=mcmc_params,
        svi_params=svi_params,
    )
    inference = trainer.infer()
    trainer.predict(df=box_scores)
    print("ok")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)

    app()
