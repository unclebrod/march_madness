from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import optim
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, autoguide, initialization

INFERENCE = Literal["mcmc", "svi"]


@dataclass
class ModelData:
    context: jnp.array  # fixed effects matrix (home, neutral, is_conf_tourney, is_ncaa_tourney)

    n: int | None  # number of rows
    n_teams: int | None  # number of unique teams
    n_seasons: int | None  # number of unique seasons

    season: jnp.ndarray  # season encoding
    team1: jnp.ndarray  # team1 encoding
    team2: jnp.ndarray  # team2 encoding
    minutes: jnp.ndarray  # game minutes

    is_neutral: jnp.ndarray  # is the game played on a neutral court?
    is_team1_home: jnp.ndarray  # is team1 the home team?
    is_team2_home: jnp.ndarray  # is team2 the home team?

    avg_poss: jnp.ndarray | None  # average possessions of the game
    team1_score: jnp.ndarray | None  # team1's score
    team2_score: jnp.ndarray | None  # team2's score


class MarchMadnessModel:
    def __init__(self):
        self.infer = None
        self.samples = None
        self.results = None

    @staticmethod
    def model(data: ModelData) -> None:
        # TODO: hardcoding many of the values that I should later try and tune for, time permitting
        # TODO: consider poisson/negative binomial for points/pace/possessions
        # TODO: ppp is pretty close to normally distributed, as is pace
        # TODO: could consider poisson for points

        # fixed effects matrices
        n_context = data.context.shape[1]

        beta_ppp = numpyro.sample("beta_ppp", dist.Normal(0, 1).expand((n_context,)))
        beta_pace = numpyro.sample("beta_pace", dist.Normal(0, 1).expand((n_context,)))

        context_ppp = jnp.dot(data.context, beta_ppp)
        context_pace = jnp.dot(data.context, beta_pace)

        home_advantage_ppp = numpyro.sample("home_advantage_ppp", dist.Normal(0, 1))

        # hyperpriors for the first season
        mu_offense_ppp = numpyro.sample("mu_offense_ppp", dist.Normal(0, 10))
        mu_defense_ppp = numpyro.sample("mu_defense_ppp", dist.Normal(0, 10))

        sigma_offense_ppp = numpyro.sample("sigma_offense_ppp", dist.Exponential(1))
        sigma_defense_ppp = numpyro.sample("sigma_defense_ppp", dist.Exponential(1))

        mu_offense_pace = numpyro.sample("mu_offense_pace", dist.Normal(0, 10))
        mu_defense_pace = numpyro.sample("mu_defense_pace", dist.Normal(0, 10))

        sigma_offense_pace = numpyro.sample("sigma_offense_pace", dist.Exponential(1))
        sigma_defense_pace = numpyro.sample("sigma_defense_pace", dist.Exponential(1))

        # team-season offense and defense vectors for ppp, pace, and possessions
        offense_ppp = jnp.zeros((data.n_teams, data.n_seasons))
        defense_ppp = jnp.zeros((data.n_teams, data.n_seasons))

        offense_pace = jnp.zeros((data.n_teams, data.n_seasons))
        defense_pace = jnp.zeros((data.n_teams, data.n_seasons))

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
        offense_pace = offense_pace.at[:, 0].set(
            numpyro.sample(
                "offense_pace_0",
                dist.Normal(mu_offense_pace, sigma_offense_pace).expand((data.n_teams,)),
            )
        )
        defense_pace = defense_pace.at[:, 0].set(
            numpyro.sample(
                "defense_pace_0",
                dist.Normal(mu_defense_pace, sigma_defense_pace).expand((data.n_teams,)),
            )
        )

        # temporal shrinkage factor, biased towards 1
        phi_ppp = numpyro.sample("phi_ppp", dist.Beta(5, 1))
        phi_pace = numpyro.sample("phi_pace", dist.Beta(5, 1))

        # use the previous season's ratings to inform the current season's ratings
        for s in range(1, data.n_seasons):
            offense_ppp = offense_ppp.at[:, s].set(
                numpyro.sample(
                    f"offense_ppp_{s}",
                    dist.Normal(phi_ppp * offense_ppp[:, s - 1], sigma_offense_ppp),
                )
            )
            defense_ppp = defense_ppp.at[:, s].set(
                numpyro.sample(
                    f"defense_ppp_{s}",
                    dist.Normal(phi_ppp * defense_ppp[:, s - 1], sigma_defense_ppp),
                )
            )
            offense_pace = offense_pace.at[:, s].set(
                numpyro.sample(
                    f"offense_pace_{s}",
                    dist.Normal(phi_pace * offense_pace[:, s - 1], sigma_offense_pace),
                )
            )
            defense_pace = defense_pace.at[:, s].set(
                numpyro.sample(
                    f"defense_pace_{s}",
                    dist.Normal(phi_pace * defense_pace[:, s - 1], sigma_defense_pace),
                )
            )

        # get offense and defense ratings for each team
        team1_offense_ppp = offense_ppp[data.team1, data.season]
        team1_defense_ppp = defense_ppp[data.team1, data.season]

        team1_offense_pace = offense_pace[data.team1, data.season]
        team1_defense_pace = defense_pace[data.team1, data.season]

        team2_offense_ppp = offense_ppp[data.team2, data.season]
        team2_defense_ppp = defense_ppp[data.team2, data.season]

        team2_offense_pace = offense_pace[data.team2, data.season]
        team2_defense_pace = defense_pace[data.team2, data.season]

        # expected points per possession
        team1_ppp = (
            team1_offense_ppp
            - team2_defense_ppp
            + (1 - data.is_neutral) * home_advantage_ppp * data.is_team1_home
            + context_ppp
        )
        team2_ppp = (
            team2_offense_ppp
            - team1_defense_ppp
            + (1 - data.is_neutral) * home_advantage_ppp * data.is_team2_home
            + context_ppp
        )

        # expected pace
        pace = ((team1_offense_pace - team2_defense_pace + team2_offense_pace - team1_defense_pace) / 2) + context_pace

        # expected possessions
        poss = pace * data.minutes

        # expected scores
        team1_score = team1_ppp * poss
        team2_score = team2_ppp * poss

        # likelihood function
        sigma_poss = numpyro.sample("sigma_poss", dist.Exponential(1))
        sigma_team1_score = numpyro.sample("sigma_team1_score", dist.Exponential(1))
        sigma_team2_score = numpyro.sample("sigma_team2_score", dist.Exponential(1))
        # TODO: consider other distributions
        numpyro.sample("poss", dist.Normal(poss, sigma_poss), obs=data.avg_poss)
        numpyro.sample("team1_score", dist.Normal(team1_score, sigma_team1_score), obs=data.team1_score)
        numpyro.sample("team2_score", dist.Normal(team2_score, sigma_team2_score), obs=data.team2_score)

    def fit(
        self,
        data: ModelData,
        inference: INFERENCE = "svi",
        num_warmup: int = 1_000,
        num_samples: int = 1_000,
        num_chains: int = 4,
        learning_rate: float = 0.01,
        num_steps: int = 25_000,
        seed: int = 0,
        **kwargs,
    ):
        # TODO: consider a learning rate schedule
        init_strategy = initialization.init_to_median(num_samples=100)
        rng_key = random.PRNGKey(seed)
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
                guide = autoguide.AutoNormal(self.model)
                self.infer = SVI(
                    model=self.model,
                    guide=guide,
                    loss=Trace_ELBO(),
                    optim=optim.Adam(learning_rate),
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

    def predict(self, data: ModelData, seed: int = 0):
        if self.samples is None:
            raise ValueError("Model must be fitted before predictions can be made.")
        rng_key = random.PRNGKey(seed)
        predictive = Predictive(self.model, self.samples)
        return predictive(rng_key, data)
