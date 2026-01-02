"""Points per possession probabilistic model."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from march_madness.models.base import BaseNumpyroModel

# # For each (season, team), normalize time to [0, 1]
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
    context: jnp.array  # fixed effects matrix (is_conf_tourney, is_ncaa_tourney)

    n_games: int | None  # number of rows
    n_teams: int | None  # number of unique teams
    n_seasons: int | None  # number of unique seasons

    season: jnp.ndarray  # season encoding
    team1: jnp.ndarray  # team1 encoding
    team2: jnp.ndarray  # team2 encoding

    minutes: jnp.ndarray  # game minutes
    is_neutral: jnp.ndarray  # is the game at a neutral location
    is_team1_home: jnp.ndarray  # is team1 the home team
    is_team2_home: jnp.ndarray  # is team2 the home team

    avg_poss: jnp.ndarray | None  # average possessions of the game
    team1_score: jnp.ndarray | None  # team1's score
    team2_score: jnp.ndarray | None  # team2's score


class PointsPerPossessionModel(BaseNumpyroModel):
    name = "ppp"

    def model(
        self,
        data: PointsPerPossessionData,
        **kwargs,
    ) -> None:
        # TODO: re-examine all priors
        # TODO: hardcoding many of the values that I should later try and tune for, time permitting
        # TODO: ideas for fixed effects: days rest, back to back, etc.
        # TODO: consider adding weighting so end of season games are more important
        # TODO: latent factor analysis?
        # TODO: covid indicator?
        # ---- Use season + 1 as the max in order to predict the next season ----
        n_seasons = data.n_seasons + 1

        # ---- Fixed effects matrix ----
        n_context = data.context.shape[1]

        beta = numpyro.sample("beta", dist.Normal(0, 0.1).expand((n_context,)))
        fixed_effects = jnp.dot(data.context, beta)

        # ---- Global means & standard deviations for each offense, defense, and pace ----
        offense_global_mean = numpyro.sample("offense_global_mean", dist.Normal(0, 5))
        defense_global_mean = numpyro.sample("defense_global_mean", dist.Normal(0, 5))
        pace_global_mean = numpyro.sample("pace_global_mean", dist.Normal(0, 5))

        sigma_offense_season = numpyro.sample("sigma_offense_season", dist.Exponential(1))
        sigma_defense_season = numpyro.sample("sigma_defense_season", dist.Exponential(1))
        sigma_pace_season = numpyro.sample("sigma_pace_season", dist.Exponential(1))

        sigma_offense_team = numpyro.sample("sigma_offense_team", dist.Exponential(1))
        sigma_defense_team = numpyro.sample("sigma_defense_team", dist.Exponential(1))
        sigma_pace_team = numpyro.sample("sigma_pace_team", dist.Exponential(1))

        # ---- Season-specific means for each offense, defense, and pace ----
        offense_season_rw = numpyro.sample(
            "offense_season_rw",
            dist.GaussianRandomWalk(sigma_offense_season, num_steps=n_seasons),
        )
        defense_season_rw = numpyro.sample(
            "defense_season_rw",
            dist.GaussianRandomWalk(sigma_defense_season, num_steps=n_seasons),
        )
        pace_season_rw = numpyro.sample(
            "pace_season_rw",
            dist.GaussianRandomWalk(sigma_pace_season, num_steps=n_seasons),
        )
        # Center season means to assist with identifiability
        offense_season_mean = offense_global_mean + offense_season_rw - offense_season_rw.mean()
        defense_season_mean = defense_global_mean + defense_season_rw - defense_season_rw.mean()
        pace_season_mean = pace_global_mean + pace_season_rw - pace_season_rw.mean()

        # ----Team-specific offense, defense, and pace ratings as deviations from the mean ----
        with numpyro.plate("teams", data.n_teams):
            offense_team_rw = numpyro.sample(
                "offense_team_rw",
                dist.GaussianRandomWalk(sigma_offense_team, num_steps=n_seasons),
            )
            defense_team_rw = numpyro.sample(
                "defense_team_rw",
                dist.GaussianRandomWalk(sigma_defense_team, num_steps=n_seasons),
            )
            pace_team_rw = numpyro.sample(
                "pace_team_rw",
                dist.GaussianRandomWalk(sigma_pace_team, num_steps=n_seasons),
            )

        # Center team deltas to assist with identifiability
        offense_team_delta = offense_team_rw - offense_team_rw.mean(axis=0, keepdims=True)
        defense_team_delta = defense_team_rw - defense_team_rw.mean(axis=0, keepdims=True)
        pace_team_delta = pace_team_rw - pace_team_rw.mean(axis=0, keepdims=True)

        # ---- Combine offense, defense, and pace components ----
        offense = offense_season_mean + offense_team_delta
        defense = defense_season_mean + defense_team_delta
        pace = pace_season_mean + pace_team_delta

        # ---- Team-level ratings for  ----
        team1_offense = offense[data.team1, data.season]
        team2_offense = offense[data.team2, data.season]
        team1_defense = defense[data.team1, data.season]
        team2_defense = defense[data.team2, data.season]
        team1_pace = pace[data.team1, data.season]
        team2_pace = pace[data.team2, data.season]

        # ---- Home court advantage, both season and team ----
        with numpyro.plate("seasons", n_seasons):
            hca_season = numpyro.sample("hca_season", dist.Normal(0, 0.01))

        with numpyro.plate("teams", data.n_teams):
            hca_team = numpyro.sample("hca_team", dist.Normal(0, 0.01))

        # Center team home court advantage to assist with identifiability
        hca_team = hca_team - hca_team.mean()

        # ---- Expected points per possession for each team ----
        # These are on the log scale to assist with identifiability
        log_team1_ppp = (
            team1_offense
            - team2_defense
            + fixed_effects
            + (1 - data.is_neutral)
            * (hca_season[data.season] + hca_team[data.team1])
            * data.is_team1_home
        )
        log_team2_ppp = (
            team2_offense
            - team1_defense
            + fixed_effects
            + (1 - data.is_neutral)
            * (hca_season[data.season] + hca_team[data.team2])
            * data.is_team2_home
        )
        team1_ppp = jnp.exp(log_team1_ppp)
        team2_ppp = jnp.exp(log_team2_ppp)

        # ---- Expected pace, possessions, and scores ----
        # These are on the log scale to assist with identifiability
        log_expected_pace = (team1_pace + team2_pace) / 2
        expected_pace = jnp.exp(log_expected_pace)
        possessions = expected_pace * data.minutes
        team1_score = team1_ppp * possessions
        team2_score = team2_ppp * possessions

        # ---- Likelihood functions for possessions and scores ----
        sigma_possessions = numpyro.sample("sigma_possessions", dist.Exponential(1))
        phi_score = numpyro.sample("phi_score", dist.Exponential(0.01))

        numpyro.sample(
            "possessions",
            dist.TruncatedNormal(possessions, sigma_possessions, low=0.0),
            obs=data.avg_poss,
        )
        numpyro.sample(
            "team1_score",
            dist.NegativeBinomial2(team1_score, phi_score),
            obs=data.team1_score,
        )
        numpyro.sample(
            "team2_score",
            dist.NegativeBinomial2(team2_score, phi_score),
            obs=data.team2_score,
        )
