from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import dill as pickle
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from jax import random
from jax.nn import softplus
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, autoguide, initialization

from march_madness.log import logger
from march_madness.path import OUTPUT_DIR


@dataclass
class ModelData:
    context: jnp.array  # fixed effects matrix (is_conf_tourney, is_ncaa_tourney)

    n: int | None  # number of rows
    n_teams: int | None  # number of unique teams
    n_seasons: int | None  # number of unique seasons

    season: jnp.ndarray  # season encoding
    team1: jnp.ndarray  # team1 encoding
    team2: jnp.ndarray  # team2 encoding
    loc: jnp.ndarray  # game location encoding (home, away, neutral)

    minutes: jnp.ndarray  # game minutes
    is_team1_home: jnp.ndarray  # is team1 the home team
    is_team2_home: jnp.ndarray  # is team2 the home team
    is_neutral: jnp.ndarray  # is the game at a neutral location

    avg_poss: jnp.ndarray | None  # average possessions of the game
    team1_score: jnp.ndarray | None  # team1's score


def model(data: ModelData) -> None:
    # TODO: hardcoding many of the values that I should later try and tune for, time permitting
    # TODO: ideas for fixed effects: days rest, back to back, etc.
    # TODO: consider adding weighting so end of season games are more important
    # TODO: latent factor analysis?
    # TODO: covid indicator?

    # fixed effects matrices
    n_context = data.context.shape[1]

    beta_ppp = numpyro.sample("beta_ppp", dist.Normal(0, 0.1).expand((n_context,)))
    # beta_pace = numpyro.sample("beta_pace", dist.Normal(0, 1).expand((n_context,)))

    context_ppp = jnp.dot(data.context, beta_ppp)
    # context_pace = jnp.dot(data.context, beta_pace)

    # hyperpriors for the first season
    mu_offense_ppp = numpyro.sample("mu_offense_ppp", dist.Normal(0, 5))
    mu_defense_ppp = numpyro.sample("mu_defense_ppp", dist.Normal(0, 5))
    mu_pace = numpyro.sample("mu_pace", dist.HalfNormal(5))

    sigma_offense_ppp = numpyro.sample("sigma_offense_ppp", dist.Exponential(1))
    sigma_defense_ppp = numpyro.sample("sigma_defense_ppp", dist.Exponential(1))
    sigma_pace = numpyro.sample("sigma_pace", dist.Exponential(1))

    # team-season offense and defense vectors for ppp, pace, and possessions
    offense_ppp = jnp.zeros((data.n_teams, data.n_seasons))
    defense_ppp = jnp.zeros((data.n_teams, data.n_seasons))
    pace = jnp.zeros((data.n_teams, data.n_seasons))

    # set the first season's ratings to the hyperpriors
    offense_ppp = offense_ppp.at[:, 0].set(
        numpyro.sample(
            "offense_ppp_0",
            dist.Normal(mu_offense_ppp, sigma_offense_ppp).expand((data.n_teams,)),
        )
    )
    defense_ppp = defense_ppp.at[:, 0].set(
        numpyro.sample(
            "defense_ppp_0",
            dist.Normal(mu_defense_ppp, sigma_defense_ppp).expand((data.n_teams,)),
        )
    )
    pace = pace.at[:, 0].set(
        numpyro.sample(
            "pace_0",
            dist.Normal(mu_pace, sigma_pace).expand((data.n_teams,)),
        )
    )

    # use the previous season's ratings to inform the current season's ratings
    for s in range(1, data.n_seasons):
        offense_ppp = offense_ppp.at[:, s].set(
            numpyro.sample(
                f"offense_ppp_{s}",
                dist.Normal(offense_ppp[:, s - 1], sigma_offense_ppp),
            )
        )
        defense_ppp = defense_ppp.at[:, s].set(
            numpyro.sample(
                f"defense_ppp_{s}",
                dist.Normal(defense_ppp[:, s - 1], sigma_defense_ppp),
            )
        )
        pace = pace.at[:, s].set(
            numpyro.sample(
                f"pace_{s}",
                dist.Normal(pace[:, s - 1], sigma_pace),
            )
        )

    # get offense and defense ratings for each team
    team1_offense_ppp = offense_ppp[data.team1, data.season]
    team2_defense_ppp = defense_ppp[data.team2, data.season]
    team1_pace = pace[data.team1, data.season]
    team2_pace = pace[data.team2, data.season]

    # expected points per possession
    hca_ppp = numpyro.sample("hca", dist.HalfNormal(0.01))
    team1_ppp = softplus(
        team1_offense_ppp
        - team2_defense_ppp
        + context_ppp
        + (1 - data.is_neutral) * hca_ppp * data.is_team1_home
    )

    # expected pace
    pace_rating = softplus((team1_pace + team2_pace) / 2)

    # expected possessions
    possessions = softplus(pace_rating * data.minutes)

    # expected scores
    team1_score = softplus(team1_ppp * possessions)

    # deterministic sites
    # TODO: replace with offense_ppp and defense_ppp
    net_rating = numpyro.deterministic("net_rating", team1_offense_ppp - team2_defense_ppp)

    # likelihood function
    sigma_poss = numpyro.sample("sigma_poss", dist.Exponential(1))
    phi_score = numpyro.sample("phi_score", dist.Exponential(0.01))

    numpyro.sample("possessions", dist.Normal(possessions, sigma_poss), obs=data.avg_poss)
    numpyro.sample(
        "team1_score", dist.NegativeBinomial2(team1_score, phi_score), obs=data.team1_score
    )


class MarchMadnessModel:
    def __init__(self):
        self.infer = None
        self.samples = None
        self.results = None
        self.model = model

    def fit(
        self,
        data: ModelData,
        inference: Literal["mcmc", "svi"] = "svi",
        num_warmup: int = 1_000,
        num_samples: int = 1_000,
        num_chains: int = 4,
        learning_rate: float = 0.01,
        num_steps: int = 25_000,
        seed: int = 0,
        **kwargs,
    ):
        # TODO: consider a learning rate schedule
        rng_key = random.PRNGKey(seed)
        init_strategy = initialization.init_to_median(num_samples=100)
        match inference:
            case "mcmc":
                kernel = NUTS(self.model, init_strategy=init_strategy)
                self.infer = MCMC(
                    sampler=kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
                self.infer.run(rng_key, data)
                self.infer.print_summary()
                self.samples = self.infer.get_samples()

            case "svi":
                schedule = optax.linear_onecycle_schedule(
                    peak_value=learning_rate,
                    transition_steps=num_steps,
                )
                optimizer = optax.adamw(learning_rate=schedule)
                guide = autoguide.AutoNormal(self.model)
                self.infer = SVI(
                    model=self.model,
                    guide=guide,
                    optim=optimizer,
                    loss=Trace_ELBO(),
                    **kwargs,
                )
                self.results = self.infer.run(rng_key, num_steps, data)
                rng_key, rng_key_ = random.split(rng_key)
                self.samples = guide.sample_posterior(
                    rng_key=rng_key_,
                    params=self.results.params,
                    sample_shape=(num_samples,),
                )

            case _:
                raise NotImplementedError("The provided inference method is not implemented.")

    def predict(self, data: Any, seed: int = 0):
        if self.samples is None:
            raise ValueError("Model must be fitted before predictions can be made.")
        rng_key = random.PRNGKey(seed)
        predictive = Predictive(self.model, self.samples)
        return predictive(rng_key, data=data)

    def save(self, path: str = "M"):
        save_to = OUTPUT_DIR / f"{path}/model.pkl"
        with Path(save_to).open("wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved model to {save_to}")

    @classmethod
    def load(cls, path: str = "M"):
        load_from = OUTPUT_DIR / f"{path}/model.pkl"
        with Path(load_from).open("rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {load_from}")
        return model
