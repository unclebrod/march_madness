"""Bradley-Terry/Elo probabilistic model."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from march_madness.models.base_model import BaseNumpyroModel


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
            (rating[data.team1, data.season] - rating[data.team2, data.season])
            + fixed_effects
            + (hca * home_diff)
            # + (intercept if not predict else 0.0)
        )
        # mu_centered = mu - mu.mean()

        # ---- Likelihood for spread and win probability ----
        numpyro.sample("team1_win_prob", dist.Bernoulli(logits=mu), obs=data.team1_win)
        numpyro.sample("team2_win_prob", dist.Bernoulli(logits=-mu), obs=data.team2_win)
        # z = mu / sigma_spread
        # numpyro.sample("spread", dist.Normal(mu, sigma_spread), obs=data.spread)
        # numpyro.sample("team1_win_prob", dist.Bernoulli(probs=dist.Normal(0, 1).cdf(z)), obs=data.team1_win)
        # numpyro.sample("team2_win_prob", dist.Bernoulli(probs=dist.Normal(0, 1).cdf(-z)), obs=data.team2_win)
