"""Points per possession probabilistic model."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.nn import softplus

from march_madness.models.base_model import BaseNumpyroModel

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
