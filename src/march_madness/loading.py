from pathlib import Path

import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[2]  # go up two levels to march_madness
DATA_DIR = ROOT_DIR / "data"


def _poss_expr(team_str: str, multiplier: float = 0.475) -> pl.Expr:
    return (
        pl.col(f"{team_str}_fga")
        .sub(pl.col(f"{team_str}_orb"))
        .add(pl.col(f"{team_str}_tov"))
        .add(pl.col(f"{team_str}_fta").mul(pl.lit(multiplier)))
    )


def load_box_scores(league: str = "M"):
    """Return a dataframe, in wide format (one row per game) combining all data from the specified subdirectory.

    Parameters
    ----------
    league : {'M', 'W'}, default='M'
        League to load data for, where 'M' is men's and 'W' is women's.

    Returns
    -------
    pl.DataFrame

    Notes
    -----
    # TODO: can we incorporate SecondaryTourneyCompactResults? only has final scores with no box scores
    # TODO: can we incorporate city/travel? have game-city location but don't no where teams are located
    # TODO: consider incorporating coaches?
    """
    regular_season_results = pl.read_csv(DATA_DIR / f"{league}RegularSeasonDetailedResults.csv")
    playoff_results = pl.read_csv(DATA_DIR / f"{league}NCAATourneyDetailedResults.csv")
    conferences = pl.read_csv(DATA_DIR / f"{league}TeamConferences.csv")
    conference_tourney_games = pl.read_csv(DATA_DIR / f"{league}ConferenceTourneyGames.csv")
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
            team1_poss=_poss_expr("team1"),
            team2_poss=_poss_expr("team2"),
        )
        .with_columns(
            # team1_ppp=pl.col("team1_score").truediv(pl.col("team1_poss")),
            # team2_ppp=pl.col("team2_score").truediv(pl.col("team2_poss")),
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
