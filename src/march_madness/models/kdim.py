"""K-dimensional Bradley Terry model."""

from dataclasses import dataclass

import jax.numpy as jnp


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
