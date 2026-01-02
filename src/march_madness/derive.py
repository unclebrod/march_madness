"""Haversine formula to calculate distances between two lat/lng points."""

import polars as pl

EARTH_RADIUS_MILES = 3959.0


def haversine(
    lat1: str,
    lon1: str,
    lat2: str,
    lon2: str,
) -> pl.Expr:
    """Vectorized Haversine distance (miles) as a Polars expression."""
    lat1_rad = pl.col(lat1).radians()
    lon1_rad = pl.col(lon1).radians()
    lat2_rad = pl.col(lat2).radians()
    lon2_rad = pl.col(lon2).radians()

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (dlat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2).sin().pow(2)

    c = 2 * pl.arctan2(a.sqrt(), (1 - a).sqrt())

    return EARTH_RADIUS_MILES * c


def derive_team_locations(game_team_box_scores: pl.DataFrame) -> pl.DataFrame:
    """Derive team locations based on game locations."""
    existing_team_locations = (
        game_team_box_scores.filter(pl.col("team1_loc").eq("H"), pl.col("city").is_not_null())
        .group_by("team1_id", "season")
        .agg(pl.col("city", "state").mode())
        .with_columns(
            pl.col("city").list.sort().list.get(0),
            # I am assuming the modes for city and state align - generally true
            pl.col("state").list.sort().list.get(0),
        )
    )
    missing_team_locations = (
        game_team_box_scores.select(["season", "team1_id"])
        .unique()
        .join(
            existing_team_locations.select(["season", "team1_id"]).unique(),
            on=["season", "team1_id"],
            how="anti",
        )
    )
    team_locations = (
        pl.concat([existing_team_locations, missing_team_locations], how="diagonal_relaxed")
        .rename({"team1_id": "team_id"})
        .sort(["season", "team_id"])
        .with_columns(pl.col("city", "state").backward_fill().over(["team_id"]))
    )
    cities = (
        game_team_box_scores.select(["city", "state", "lat", "lng", "elevation"])
        .unique()
        .drop_nulls()
    )
    return team_locations.join(cities, on=["city", "state"], how="left")


def generate_ncaaw_homecourt() -> pl.DataFrame:
    regions = ["W", "X", "Y", "Z"]
    seeds = list(range(1, 17))  # Seeds 1-16

    matchups = []

    # First round matchups
    first_round_winners = {}  # Store possible winners for R2 matchups
    for region in regions:
        first_round_winners[region] = []
        for i in range(8):  # 1 vs 16, 2 vs 15, ..., 8 vs 9
            high_seed = seeds[i]
            low_seed = seeds[-(i + 1)]
            slot = f"R1{region}{i + 1}"
            team_a = f"{region}{high_seed:02d}"
            team_b = f"{region}{low_seed:02d}"
            has_home_court = int(high_seed <= 4)  # Top 4 seeds host

            matchups.append([slot, team_a, team_b, has_home_court])
            first_round_winners[region].append((team_a, team_b))  # Save for R2

    # Second round matchups (explicitly listing all combinations)
    for region in regions:
        for i in range(4):  # Each R2 slot is winner of (1 vs 16) vs winner of (8 vs 9), etc.
            slot = f"R2{region}{i + 1}"
            team_a_options = first_round_winners[region][i]  # Winner of high-seed game
            team_b_options = first_round_winners[region][8 - i - 1]  # Winner of low-seed game

            for team_a in team_a_options:
                for team_b in team_b_options:
                    # Home court logic: Only retains home advantage if original top 4 seed wins
                    orig_host_seed = int(team_a[1:])  # Extract seed from "W01"
                    has_home_court = int(orig_host_seed <= 4)  # Home if original host won

                    matchups.append([slot, team_a, team_b, has_home_court])

    return pl.DataFrame(
        matchups,
        schema=["Slot", "StrongSeed", "WeakSeed", "is_team1_home"],
        orient="row",
    )
