from pathlib import Path
from typing import Generic, Literal, TypeVar

import polars as pl
from pydantic import BaseModel

from march_madness import DATA_DIR

T = TypeVar("T", bound=BaseModel)


def _poss_expr(team_str: str, multiplier: float = 0.475) -> pl.Expr:
    """Expression for calculating an estimated number of possessions."""
    return (
        pl.col(f"{team_str}_fga")
        .sub(pl.col(f"{team_str}_orb"))
        .add(pl.col(f"{team_str}_tov"))
        .add(pl.col(f"{team_str}_fta").mul(pl.lit(multiplier)))
    )


class FileConfig(BaseModel):
    """File configuration convenience class."""

    filename: str
    requires_league: bool
    men_only: bool = False


class DataConfig(Generic[T]):
    """Container for keeping track of file loading configurations."""

    cities: T = FileConfig(filename="Cities", requires_league=False)
    conferences: T = FileConfig(filename="Conferences", requires_league=False)
    conference_tourney_games: T = FileConfig(filename="ConferenceTourneyGames", requires_league=True)
    game_cities: T = FileConfig(filename="GameCities", requires_league=True)
    massey_ordinals: T = FileConfig(filename="MasseyOrdinals", requires_league=True, men_only=True)
    ncaa_tourney_compact_results: T = FileConfig(filename="NCAATourneyCompactResults", requires_league=True)
    ncaa_tourney_detailed_results: T = FileConfig(filename="NCAATourneyDetailedResults", requires_league=True)
    ncaa_tourney_seed_round_slots: T = FileConfig(
        filename="NCAATourneySeedRoundSlots", requires_league=True, men_only=True
    )
    ncaa_tourney_seeds: T = FileConfig(filename="NCAATourneySeeds", requires_league=True)
    ncaa_tourney_slots: T = FileConfig(filename="NCAATourneySlots", requires_league=True)
    regular_season_compact_results: T = FileConfig(filename="RegularSeasonCompactResults", requires_league=True)
    regular_season_detailed_results: T = FileConfig(filename="RegularSeasonDetailedResults", requires_league=True)
    seasons: T = FileConfig(filename="Seasons", requires_league=False)
    secondary_tourney_compact_results: T = FileConfig(filename="SecondaryTourneyCompactResults", requires_league=True)
    secondary_tourney_teams: T = FileConfig(filename="SecondaryTourneyTeams", requires_league=True)
    team_coaches: T = FileConfig(filename="TeamCoaches", requires_league=True, men_only=True)
    team_conferences: T = FileConfig(filename="TeamConferences", requires_league=True)
    teams: T = FileConfig(filename="Teams", requires_league=True)
    team_spellings: T = FileConfig(filename="TeamSpellings", requires_league=True)
    sample_submissions_stage_1: T = FileConfig(filename="SampleSubmissionStage1", requires_league=False)
    sample_submissions_stage_2: T = FileConfig(filename="SampleSubmissionStage2", requires_league=False)
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
        regular_season_results = self.data_loader.load_data(self.data_config.regular_season_detailed_results)
        playoff_results = self.data_loader.load_data(self.data_config.ncaa_tourney_detailed_results)
        conferences = self.data_loader.load_data(self.data_config.team_conferences)
        conference_tourney_games = self.data_loader.load_data(self.data_config.conference_tourney_games)
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
        }
        return (
            pl.concat(
                [
                    regular_season_results.with_columns(is_ncaa_tourney=0),
                    playoff_results.with_columns(is_ncaa_tourney=1),
                ],
                how="vertical",
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
                conference_tourney_games.rename({"ConfAbbrev": "conf_tourney_abbr"}),
                how="left",
                on=["Season", "DayNum", "WTeamID", "LTeamID"],
            )
            .rename(renamer)
            .sort(["season", "day_num", "team1_id"])
            .with_columns(
                team2_loc=pl.col("team1_loc").replace({"H": "A", "A": "H"}),
                is_conf_tourney=pl.col("conf_tourney_abbr").is_not_null().cast(int),
            )
            .with_row_index("game_id")
            .with_columns(
                minutes=pl.lit(40).add(pl.col("num_ot").mul(pl.lit(5))),
                spread=pl.col("team2_score").sub(pl.col("team1_score")),
                is_team1_home=pl.col("team1_loc").eq("H").cast(int),
                is_team2_home=pl.col("team2_loc").eq("H").cast(int),
                is_neutral=pl.col("team1_loc").eq("N").cast(int),
                is_team1_win=pl.col("team1_score").gt(pl.col("team2_score")).cast(int),
                is_team2_win=pl.col("team2_score").gt(pl.col("team1_score")).cast(int),
                team1_poss=_poss_expr("team1"),
                team2_poss=_poss_expr("team2"),
            )
            .with_columns(
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

    def load_game_team_box_scores(self) -> pl.DataFrame:
        """Return box score and other game-level information with one row per team-game (two rows per game)."""
        game_box_scores = self.load_game_box_scores()
        rename_dict = {
            col: col.replace("team1_", "TEMP_").replace("team2_", "team1_").replace("TEMP_", "team2_")
            for col in game_box_scores.columns
            if col.startswith("team1_") or col.startswith("team2_")
        }
        return pl.concat(
            [
                game_box_scores,
                game_box_scores.with_columns(spread=pl.col("spread").mul(pl.lit(-1)))
                .rename(rename_dict)
                .select(game_box_scores.columns),
            ],
            how="vertical",
        ).sort("season", "game_id", "day_num", "team1_id")

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
            .filter(pl.col("team1_id").cast(str).str.starts_with("1" if self.league == "M" else "3"))
        )
        rename_dict = {
            col: col.replace("team1_", "TEMP_").replace("team2_", "team1_").replace("TEMP_", "team2_")
            for col in sample_submission.columns
            if col.startswith("team1_") or col.startswith("team2_")
        }
        return pl.concat(
            [
                sample_submission,
                sample_submission.rename(rename_dict).select(sample_submission.columns),
            ],
            how="vertical",
        ).drop(["Pred", "id_split"])
