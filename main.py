import numpyro
import typer

from march_madness.loader import DataConstructor
from march_madness.trainer import Trainer

app = typer.Typer()


@app.command()
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
):
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


@app.command()
def predict(
    league: str = "M",
):
    data_constructor = DataConstructor(league=league)
    trainer = Trainer.load(league=league)
    box_scores = data_constructor.load_game_team_box_scores()
    trainer.predict(
        df=box_scores,
    )


@app.command()
def infer(
    league: str = "M",
    *,
    save: bool = True,
):
    trainer = Trainer.load(league=league)
    trainer.infer(
        save=save,
    )


@app.command()
def submit(
    league: str = "M",
    *,
    save: bool = True,
):
    trainer = Trainer.load(league=league)
    trainer.submit(
        save=save,
    )


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    numpyro.set_platform("cpu")

    app()
