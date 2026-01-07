from collections import defaultdict
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel

from march_madness.settings import (
    DATA_DIR,
    FINAL_FOUR_REGION_SETTINGS,
    OUTPUT_DIR,
    WITHIN_REGION_STANDARD_SEED_SETTINGS,
)
from march_madness.utils import (
    current_season,
    derive_rest_days,
    derive_team_locations,
    generate_ncaaw_homecourt,
    haversine,
)

pd.set_option("future.no_silent_downcasting", True)

T = TypeVar("T", bound=BaseModel)


def _poss_expr(team_str: str, multiplier: float = 0.475) -> pl.Expr:
    """Expression for calculating an estimated number of possessions."""
    return (
        pl.col(f"{team_str}_fga")
        .sub(pl.col(f"{team_str}_orb"))
        .add(pl.col(f"{team_str}_tov"))
        .add(pl.col(f"{team_str}_fta").mul(pl.lit(multiplier)))
    )


def _fill_null_location(
    null_col: str,
    team1_loc_col: str = "team1_loc",
) -> pl.Expr:
    """Expression to fill in null location data on home/away status."""
    return (
        pl.when(pl.col(null_col).is_null())
        .then(
            pl.when(pl.col(team1_loc_col).eq("H"))
            .then(pl.col(f"team1_{null_col}"))
            .when(pl.col(team1_loc_col).eq("A"))
            .then(pl.col(f"team2_{null_col}"))
        )
        .otherwise(pl.col(null_col))
    )


class FileConfig(BaseModel):
    """File configuration convenience class."""

    filename: str
    requires_league: bool
    men_only: bool = False


class DataConfig(Generic[T]):
    """Container for keeping track of file loading configurations."""

    cities: T = FileConfig(filename="Cities", requires_league=False)
    cities_geocoded: T = FileConfig(filename="CitiesGeocoded", requires_league=False)
    conferences: T = FileConfig(filename="Conferences", requires_league=False)
    conference_tourney_games: T = FileConfig(
        filename="ConferenceTourneyGames", requires_league=True
    )
    game_cities: T = FileConfig(filename="GameCities", requires_league=True)
    massey_ordinals: T = FileConfig(filename="MasseyOrdinals", requires_league=True, men_only=True)
    ncaa_tourney_compact_results: T = FileConfig(
        filename="NCAATourneyCompactResults", requires_league=True
    )
    ncaa_tourney_detailed_results: T = FileConfig(
        filename="NCAATourneyDetailedResults", requires_league=True
    )
    ncaa_tourney_seed_round_slots: T = FileConfig(
        filename="NCAATourneySeedRoundSlots", requires_league=True, men_only=True
    )
    ncaa_tourney_seeds: T = FileConfig(filename="NCAATourneySeeds", requires_league=True)
    ncaa_tourney_slots: T = FileConfig(filename="NCAATourneySlots", requires_league=True)
    regular_season_compact_results: T = FileConfig(
        filename="RegularSeasonCompactResults", requires_league=True
    )
    regular_season_detailed_results: T = FileConfig(
        filename="RegularSeasonDetailedResults", requires_league=True
    )
    seasons: T = FileConfig(filename="Seasons", requires_league=True)
    secondary_tourney_compact_results: T = FileConfig(
        filename="SecondaryTourneyCompactResults", requires_league=True
    )
    secondary_tourney_teams: T = FileConfig(filename="SecondaryTourneyTeams", requires_league=True)
    team_coaches: T = FileConfig(filename="TeamCoaches", requires_league=True, men_only=True)
    team_conferences: T = FileConfig(filename="TeamConferences", requires_league=True)
    teams: T = FileConfig(filename="Teams", requires_league=True)
    team_spellings: T = FileConfig(filename="TeamSpellings", requires_league=True)
    sample_submissions_stage_1: T = FileConfig(
        filename="SampleSubmissionStage1", requires_league=False
    )
    sample_submissions_stage_2: T = FileConfig(
        filename="SampleSubmissionStage2", requires_league=False
    )
    seed_benchmark_stage_1: T = FileConfig(filename="SeedBenchmarkStage1", requires_league=False)


class DataLoader(BaseModel):
    """Class for easily loading data given a configuration."""

    league: Literal["M", "W"] = "M"
    data_dir: Path = DATA_DIR

    def load_data(self, config: FileConfig) -> pl.DataFrame:
        if self.league != "M" and config.men_only:
            raise ValueError("The selected data is only available for men.")
        prefix = self.league if config.requires_league else ""
        return pl.read_csv(DATA_DIR / f"{prefix}{config.filename}.csv")


class DataConstructor:
    """Class responsible for loading and constructing data for use downstream."""

    def __init__(self, league: Literal["M", "W"] = "M", **kwargs):
        self.league = league
        self.data_loader = DataLoader(league=league, **kwargs)
        self.data_config = DataConfig()

    def load_game_box_scores(self) -> pl.DataFrame:
        """Return box score and other game-level information with one row per game."""
        # TODO: can we incorporate SecondaryTourneyCompactResults? only has final scores with no box scores
        # TODO: can we incorporate city/travel? have game-city location but don't no where teams are located
        # TODO: consider incorporating coaches?
        teams = self.data_loader.load_data(self.data_config.teams)
        regular_season_results = self.data_loader.load_data(
            self.data_config.regular_season_detailed_results
        )
        playoff_results = self.data_loader.load_data(self.data_config.ncaa_tourney_detailed_results)
        secondary_results = self.data_loader.load_data(
            self.data_config.secondary_tourney_compact_results
        )
        conferences = self.data_loader.load_data(self.data_config.team_conferences)
        conference_tourney_games = self.data_loader.load_data(
            self.data_config.conference_tourney_games
        )
        seasons = self.data_loader.load_data(self.data_config.seasons)
        game_cities = self.data_loader.load_data(self.data_config.game_cities)
        cities = self.data_loader.load_data(self.data_config.cities_geocoded)
        renamer = {
            "Season": "season",
            "DayNum": "day_num",
            "NumOT": "num_ot",
            "WTeamID": "team1_id",
            "WScore": "team1_score",
            "WLoc": "team1_loc",
            "WFGM": "team1_fgm",
            "WFGA": "team1_fga",
            "WFGM3": "team1_fg3m",
            "WFGA3": "team1_fg3a",
            "WFTM": "team1_ftm",
            "WFTA": "team1_fta",
            "WOR": "team1_orb",
            "WDR": "team1_drb",
            "WAst": "team1_ast",
            "WTO": "team1_tov",
            "WStl": "team1_stl",
            "WBlk": "team1_blk",
            "WPF": "team1_pf",
            "WTeamConf": "team1_conf_abbr",
            "LTeamID": "team2_id",
            "LScore": "team2_score",
            "LFGM": "team2_fgm",
            "LFGA": "team2_fga",
            "LFGM3": "team2_fg3m",
            "LFGA3": "team2_fg3a",
            "LFTM": "team2_ftm",
            "LFTA": "team2_fta",
            "LOR": "team2_orb",
            "LDR": "team2_drb",
            "LAst": "team2_ast",
            "LTO": "team2_tov",
            "LStl": "team2_stl",
            "LBlk": "team2_blk",
            "LPF": "team2_pf",
            "LTeamConf": "team2_conf_abbr",
            "SecondaryTourney": "secondary_tourney",
        }
        df = (
            pl.concat(
                [
                    regular_season_results.with_columns(is_ncaa_tourney=0, is_secondary_tourney=0),
                    playoff_results.with_columns(is_ncaa_tourney=1, is_secondary_tourney=0),
                    secondary_results.with_columns(is_ncaa_tourney=0, is_secondary_tourney=1),
                ],
                how="diagonal_relaxed",
            )
            .join(
                conferences.rename({"ConfAbbrev": "WTeamConf", "TeamID": "WTeamID"}),
                how="left",
                on=["Season", "WTeamID"],
            )
            .join(
                conferences.rename({"ConfAbbrev": "LTeamConf", "TeamID": "LTeamID"}),
                how="left",
                on=["Season", "LTeamID"],
            )
            .join(
                teams.rename(
                    {
                        "TeamID": "WTeamID",
                        "TeamName": "team1_name",
                        "FirstD1Season": "team1_first_d1_season",
                        "LastD1Season": "team1_last_d1_season",
                    }
                ),
                how="left",
                on=["WTeamID"],
            )
            .join(
                teams.rename(
                    {
                        "TeamID": "LTeamID",
                        "TeamName": "team2_name",
                        "FirstD1Season": "team2_first_d1_season",
                        "LastD1Season": "team2_last_d1_season",
                    }
                ),
                how="left",
                on=["LTeamID"],
            )
            .join(
                conference_tourney_games.rename({"ConfAbbrev": "conf_tourney_abbr"}),
                how="left",
                on=["Season", "DayNum", "WTeamID", "LTeamID"],
            )
            .join(
                seasons.rename(
                    {
                        "DayZero": "day_zero",
                        "RegionW": "region_w",
                        "RegionX": "region_x",
                        "RegionY": "region_y",
                        "RegionZ": "region_z",
                    }
                ),
                on=["Season"],
                how="left",
            )
            .join(
                game_cities.rename({"CRType": "cr_type", "CityID": "city_id"}),
                on=["Season", "DayNum", "WTeamID", "LTeamID"],
                how="left",
            )
            .join(
                cities,
                on=["city_id"],
                how="left",
            )
            .rename(renamer)
            .sort(["season", "day_num", "team1_id"])
            .with_columns(
                team2_loc=pl.col("team1_loc").replace({"H": "A", "A": "H"}),
                is_conf_tourney=pl.col("conf_tourney_abbr").is_not_null().cast(int),
                day_zero=pl.col("day_zero").str.to_date("%m/%d/%Y"),
                days_into_season=pl.col("day_num").sub(pl.col("day_num").min().over("season")),
            )
            .with_row_index("game_id")
            .with_columns(
                days_into_season_norm=pl.col("days_into_season").truediv(
                    pl.col("days_into_season").max().over("season")
                ),
                days_into_season_sq=pl.col("days_into_season").pow(2),
                date=pl.col("day_zero").add(pl.duration(days=pl.col("day_num"))),
                minutes=pl.lit(40).add(pl.col("num_ot").mul(pl.lit(5))),
                spread=pl.col("team2_score").sub(pl.col("team1_score")),
                team1_home=pl.col("team1_loc").eq("H").cast(int),
                team2_home=pl.col("team2_loc").eq("H").cast(int),
                is_neutral=pl.col("team1_loc").eq("N").cast(int),
                team1_win=pl.col("team1_score").gt(pl.col("team2_score")).cast(int),
                team2_win=pl.col("team2_score").gt(pl.col("team1_score")).cast(int),
                team1_poss=_poss_expr("team1"),
                team2_poss=_poss_expr("team2"),
            )
            .with_columns(
                days_into_season_norm_sq=pl.col("days_into_season_norm").pow(2),
                team1_ppp=pl.col("team1_score").truediv(pl.col("team1_poss")),
                team2_ppp=pl.col("team2_score").truediv(pl.col("team2_poss")),
                team1_pace=pl.col("team1_poss").truediv(pl.col("minutes")),
                team2_pace=pl.col("team2_poss").truediv(pl.col("minutes")),
                avg_poss=pl.col("team1_poss").add(pl.col("team2_poss")).truediv(2),
            )
            .with_columns(
                team1_ppp=pl.col("team1_score").truediv(pl.col("avg_poss")),
                team2_ppp=pl.col("team2_score").truediv(pl.col("avg_poss")),
                avg_pace=pl.col("avg_poss").truediv(pl.col("minutes")),
            )
            .sort(["season", "game_id", "day_num", "team1_id"])
        )
        if self.league == "M":
            # TODO: check if this changes - as of now, these are men's only data
            coaches = self.data_loader.load_data(self.data_config.team_coaches)
            massey_ordinals = (
                self.data_loader.load_data(self.data_config.massey_ordinals)
                .with_columns(
                    SystemName=pl.format("ranking_{}", pl.col("SystemName").str.to_lowercase())
                )
                .pivot(
                    on=["SystemName"],
                    index=["Season", "RankingDayNum", "TeamID"],
                    values=["OrdinalRank"],
                )
                .rename({"Season": "season", "TeamID": "team_id", "RankingDayNum": "day_num"})
            )
            rank_cols = [x for x in massey_ordinals.columns if x.startswith("ranking_")]
            massey_ordinals = massey_ordinals.sort(["season", "team_id", "day_num"]).with_columns(
                *[pl.col(x).forward_fill().over(["season", "team_id"]) for x in rank_cols]
            )

            df = (
                df.join_where(
                    coaches,
                    pl.col("season").eq(pl.col("Season")),
                    pl.col("team1_id").eq(pl.col("TeamID")),
                    pl.col("day_num").ge(pl.col("FirstDayNum")),
                    pl.col("day_num").le(pl.col("LastDayNum")),
                )
                .drop(["Season", "TeamID", "FirstDayNum", "LastDayNum"])
                .rename({"CoachName": "team1_coach"})
                .join_where(
                    coaches,
                    pl.col("season").eq(pl.col("Season")),
                    pl.col("team1_id").eq(pl.col("TeamID")),
                    pl.col("day_num").ge(pl.col("FirstDayNum")),
                    pl.col("day_num").le(pl.col("LastDayNum")),
                )
                .drop(["Season", "TeamID", "FirstDayNum", "LastDayNum"])
                .rename({"CoachName": "team2_coach"})
                .sort(["season", "team1_id", "day_num"])
                .join_asof(
                    massey_ordinals.rename(
                        {x: f"team1_{x}" for x in rank_cols} | {"team_id": "team1_id"}
                    ),
                    by=["season", "team1_id"],
                    on="day_num",
                    strategy="backward",
                )
                .join_asof(
                    massey_ordinals.rename(
                        {x: f"team2_{x}" for x in rank_cols} | {"team_id": "team2_id"}
                    ),
                    by=["season", "team2_id"],
                    on="day_num",
                    strategy="backward",
                )
                .sort(["season", "game_id", "day_num", "team1_id"])
            )

        return df

    def load_game_team_box_scores(self) -> pl.DataFrame:
        """Return box score and other game-level information with one row per team-game (two rows per game)."""
        game_box_scores = self.load_game_box_scores()
        rename_dict = {
            col: col.replace("team1_", "TEMP_")
            .replace("team2_", "team1_")
            .replace("TEMP_", "team2_")
            for col in game_box_scores.columns
            if col.startswith("team1_") or col.startswith("team2_")
        }
        game_team_box_scores = pl.concat(
            [
                game_box_scores,
                game_box_scores.with_columns(spread=pl.col("spread").mul(pl.lit(-1)))
                .rename(rename_dict)
                .select(game_box_scores.columns),
            ],
            how="vertical",
        ).sort("season", "game_id", "day_num", "team1_id")
        team_locations = derive_team_locations(game_team_box_scores)
        team_rest = derive_rest_days(game_team_box_scores)
        return (
            game_team_box_scores.join(
                team_locations.rename(
                    {
                        "team_id": "team1_id",
                        "city": "team1_city",
                        "state": "team1_state",
                        "lat": "team1_lat",
                        "lng": "team1_lng",
                        "elevation": "team1_elevation",
                    }
                ),
                on=["season", "team1_id"],
                how="left",
            )
            .join(
                team_locations.rename(
                    {
                        "team_id": "team2_id",
                        "city": "team2_city",
                        "state": "team2_state",
                        "lat": "team2_lat",
                        "lng": "team2_lng",
                        "elevation": "team2_elevation",
                    }
                ),
                on=["season", "team2_id"],
                how="left",
            )
            .join(
                team_rest.rename(
                    {
                        "team_id": "team1_id",
                        "b2b": "team1_b2b",
                        "2in3": "team1_2in3",
                        "3in4": "team1_3in4",
                        "4in5": "team1_4in5",
                    }
                ),
                on=["season", "team1_id", "game_id"],
                how="left",
            )
            .join(
                team_rest.rename(
                    {
                        "team_id": "team2_id",
                        "b2b": "team2_b2b",
                        "2in3": "team2_2in3",
                        "3in4": "team2_3in4",
                        "4in5": "team2_4in5",
                    }
                ),
                on=["season", "team2_id", "game_id"],
                how="left",
            )
            .with_columns(
                city=_fill_null_location(null_col="city"),
                state=_fill_null_location(null_col="state"),
                lat=_fill_null_location(null_col="lat"),
                lng=_fill_null_location(null_col="lng"),
                elevation=_fill_null_location(null_col="elevation"),
            )
            .with_columns(
                team1_travel_distance=pl.when(pl.col("team1_loc").eq("H"))
                .then(0)
                .otherwise(haversine(lat1="team1_lat", lon1="team1_lng", lat2="lat", lon2="lng")),
                team2_travel_distance=pl.when(pl.col("team2_loc").eq("H"))
                .then(0)
                .otherwise(haversine(lat1="team2_lat", lon1="team2_lng", lat2="lat", lon2="lng")),
            )
            .with_columns(
                # Will use convential of team2 - team1 for consistency with spread
                **{
                    "travel_distance_diff": pl.col("team2_travel_distance").sub(
                        pl.col("team1_travel_distance")
                    ),
                    "elevation_diff": pl.col("team2_elevation").sub(pl.col("team1_elevation")),
                    "b2b_diff": pl.col("team2_b2b").sub(pl.col("team1_b2b")),
                    "2in3_diff": pl.col("team2_2in3").sub(pl.col("team1_2in3")),
                    "3in4_diff": pl.col("team2_3in4").sub(pl.col("team1_3in4")),
                    "4in5_diff": pl.col("team2_4in5").sub(pl.col("team1_4in5")),
                },
            )
            .with_columns(
                log_travel_distance_diff=pl.when(pl.col("travel_distance_diff").gt(0))
                .then(pl.col("travel_distance_diff").log1p())
                .otherwise(pl.col("travel_distance_diff").mul(pl.lit(-1)).log1p().mul(pl.lit(-1))),
                log_elevation_diff=pl.when(pl.col("elevation_diff").gt(0))
                .then(pl.col("elevation_diff").log1p())
                .otherwise(pl.col("elevation_diff").mul(pl.lit(-1)).log1p().mul(pl.lit(-1))),
                fatigue_effect_1=pl.col("b2b_diff"),
            )
            .with_columns(
                fatigue_effect_2=pl.col("2in3_diff").add(pl.col("fatigue_effect_1")),
            )
            .with_columns(
                fatigue_effect_3=pl.col("3in4_diff").add(pl.col("fatigue_effect_2")),
            )
            .with_columns(
                fatigue_effect_4=pl.col("4in5_diff").add(pl.col("fatigue_effect_3")),
            )
        )

    def generate_test_data(self):
        sample_submission = (
            self.data_loader.load_data(self.data_config.sample_submissions_stage_2)
            .with_columns(
                id_split=pl.col("ID").str.split("_"),
            )
            .with_columns(
                season=pl.col("id_split").list.get(0).cast(int),
                team1_id=pl.col("id_split").list.get(1).cast(int),
                team2_id=pl.col("id_split").list.get(2).cast(int),
                minutes=pl.lit(40),
                loc=pl.lit("N"),
                is_neutral=pl.lit(1),
                is_team1_home=pl.lit(0),
                is_team2_home=pl.lit(0),
                is_conf_tourney=pl.lit(0),
                is_ncaa_tourney=pl.lit(1),
                team1_loc=pl.lit("N"),
                team2_loc=pl.lit("N"),
            )
            .filter(
                pl.col("team1_id").cast(str).str.starts_with("1" if self.league == "M" else "3")
            )
        )
        rename_dict = {
            col: col.replace("team1_", "TEMP_")
            .replace("team2_", "team1_")
            .replace("TEMP_", "team2_")
            for col in sample_submission.columns
            if col.startswith("team1_") or col.startswith("team2_")
        }
        test_df = pl.concat(
            [
                sample_submission,
                sample_submission.rename(rename_dict).select(sample_submission.columns),
            ],
            how="vertical",
        ).drop(["Pred", "id_split"])
        if self.league == "W":
            # In the women's tourney, higher seeds have home court advantage in the first two rounds
            tourney_seeds = (
                self.data_loader.load_data(self.data_config.ncaa_tourney_seeds)
                .filter(pl.col("Season").eq(current_season()))
                .drop("Season")
            )
            homecourt_df = (
                generate_ncaaw_homecourt()
                .join(
                    tourney_seeds.rename({"Seed": "StrongSeed", "TeamID": "team1_id"}),
                    how="left",
                    on="StrongSeed",
                )
                .join(
                    tourney_seeds.rename({"Seed": "WeakSeed", "TeamID": "team2_id"}),
                    how="left",
                    on="WeakSeed",
                )
            )
            hca = (
                pl.concat(
                    [
                        homecourt_df,
                        homecourt_df.with_columns(
                            team2_id=pl.col("team1_id"),
                            team1_id=pl.col("team2_id"),
                            is_team2_home=pl.col("is_team1_home"),
                        ).drop("is_team1_home"),
                    ],
                    how="diagonal_relaxed",
                )
                .with_columns(
                    pl.col("is_team1_home").fill_null(0),
                    pl.col("is_team2_home").fill_null(0),
                )
                .with_columns(
                    is_neutral=pl.lit(0),
                    team1_loc=pl.when(pl.col("is_team1_home").eq(1))
                    .then(pl.lit("H"))
                    .otherwise(pl.lit("A")),
                    team2_loc=pl.when(pl.col("is_team2_home").eq(1))
                    .then(pl.lit("H"))
                    .otherwise(pl.lit("A")),
                )
            )
            test_df = test_df.update(
                hca.drop(["Slot", "StrongSeed", "WeakSeed"]),
                on=["team1_id", "team2_id"],
                how="left",
            )
        return test_df

    def get_final_submission(self, suffix: str | None = None) -> pl.DataFrame:
        sfx = f"_{suffix}" if suffix else ""
        df_list = []
        for league in ["M", "W"]:
            submission = pl.read_csv(OUTPUT_DIR / f"{league}/submission{sfx}.csv")
            df_list.append(submission)
        final_submission = pl.concat(df_list, how="vertical")
        final_submission.write_csv(OUTPUT_DIR / f"final_submission{sfx}.csv")

    def generate_bracket(self, suffix: str | None = None, *, save: bool = True) -> pl.DataFrame:
        """Generate a bracket for the NCAA tournament."""
        sfx = f"_{suffix}" if suffix else ""
        # tourney_slots = self.data_loader.load_data(self.data_config.ncaa_tourney_slots).filter(pl.col("Season").eq(2025), pl.col(""))
        teams = self.data_loader.load_data(self.data_config.teams)
        tourney_seeds = (
            self.data_loader.load_data(self.data_config.ncaa_tourney_seeds)
            .filter(pl.col("Season").eq(current_season()))
            .join(
                teams.select("TeamID", "TeamName"),
                how="left",
                on="TeamID",
            )
            .with_columns(pl.col("Seed").str.slice(0, 3))
            .with_columns(
                Region=pl.col("Seed").str.slice(
                    0, 1
                ),  # Extract region from seed (e.g., "W01" -> "W")
                Seed=pl.col("Seed")
                .str.slice(1, 3)
                .cast(int),  # Extract seed number (e.g., "W01" -> 1)
            )
        )
        predictions = pl.read_csv(OUTPUT_DIR / f"{self.league}/submission{sfx}.csv")
        if self.league == "M":
            # losers - Mt. Saint Mary's (1110), Texas (1400), San Diego St (1361), St. Francis (1384)
            exclude = [1110, 1400, 1361, 1384]
        else:
            # losers - Princeton (3343), UC San Diego (3471) [gonna assume William & Mary (3456) and Washington (3449)]
            exclude = [3343, 3471, 3456, 3449]
        tourney_seeds = tourney_seeds.filter(~pl.col("TeamID").is_in(exclude))
        team_list = tourney_seeds["TeamName"].to_list()
        team_id_list = tourney_seeds["TeamID"].to_list()
        team_map = dict(zip(team_list, team_id_list, strict=False))

        def _get_win_prob(team: str, opp_team: str) -> float:
            team_id = team_map[team]
            opp_team_id = team_map[opp_team]
            if team_id < opp_team_id:
                game_id = f"2025_{team_id}_{opp_team_id}"
                # lower id means we take prediction directly
                win_prob = predictions.filter(pl.col("ID").eq(game_id))["Pred"].item()
            else:
                game_id = f"2025_{opp_team_id}_{team_id}"
                # higher id means we take 1 - prediction
                win_prob = 1 - predictions.filter(pl.col("ID").eq(game_id))["Pred"].item()
            return win_prob

        matchup_dict = defaultdict(list)
        for team in team_list:
            for opp_team in team_list:
                if team == opp_team:
                    prob = None
                else:
                    prob = _get_win_prob(team, opp_team)
                matchup_dict[team].append(prob)

        matchup_df = pd.DataFrame(matchup_dict, index=team_list).T

        round_df = pd.DataFrame(
            data=np.tile(np.nan, (len(team_list), len(team_list))),
            index=team_list,
            columns=team_list,
        )
        base_df = pd.DataFrame(tourney_seeds, columns=tourney_seeds.columns)
        for i, row in base_df.iterrows():
            region_round_map = WITHIN_REGION_STANDARD_SEED_SETTINGS[row["Seed"]]
            region_round_map[row["Seed"]] = np.nan
            base_rounds = base_df["Seed"].replace(region_round_map)

            # Set self to null, necessary since sometimes will have the same seed as a play-in
            base_rounds.loc[i] = np.nan

            # Set Final Four matchups
            final_four_rounds = base_df["Region"] == FINAL_FOUR_REGION_SETTINGS[row["Region"]]
            base_rounds.loc[final_four_rounds] = base_rounds.max() + 1
            # In standard 64 bracket (no play ins) this should sum to 16 (all teams from paired region)

            # Set Championship matchups
            championship_rounds = (base_df["Region"] != row["Region"]) & (~final_four_rounds)
            base_rounds.loc[championship_rounds] = (
                base_rounds.max() + 1
            )  # only +1 since now this has some Final Four
            # In standard 64 bracket (no play ins) this should sum to 32 (all teams from other side of bracket)

            round_df[row["TeamName"]] = pd.Series(base_rounds.values, index=team_list)

        round_df = round_df.T

        advance_df = pd.DataFrame(index=round_df.index)
        # TODO: will need to change R0 and range if first four
        # For now, this essentially skips the round=0 created by some of the round generators
        advance_df["R0"] = 1.0
        for round_num in range(1, 6 + 1):
            advance_series = pd.Series(dtype=float)
            for team_name, _ in round_df.iterrows():
                team_round_row = round_df.loc[team_name, :]
                round_opponents = team_round_row.loc[team_round_row == round_num].index.tolist()
                advance_pr = (
                    advance_df.loc[round_opponents, f"R{round_num - 1}"]
                    * matchup_df.loc[team_name, round_opponents]
                ).sum()
                advance_series[team_name] = (
                    advance_df.loc[team_name, f"R{round_num - 1}"] * advance_pr
                )
            advance_df[f"R{round_num}"] = advance_series

        advance_df = advance_df.sort_values("R6", ascending=False)
        advance_df.to_csv(OUTPUT_DIR / f"{self.league}/advance{sfx}.csv")
