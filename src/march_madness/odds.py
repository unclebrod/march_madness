"""Extract odds data from ESPN's API and save to CSV."""

from typing import Any

import polars as pl
import requests

from march_madness.settings import OUTPUT_DIR


def extract_odds(odds: dict[str, Any]) -> list[dict[str, Any]]:
    return {
        "provider_name": odds.get("provider", {}).get("name"),
        "details": odds.get("details"),
        "total": odds.get("overUnder"),
        "spread": odds.get("spread"),
        "home_team": odds.get("homeTeamOdds", {}).get("team", {}).get("displayName"),
        "away_team": odds.get("awayTeamOdds", {}).get("team", {}).get("displayName"),
        "home_moneyline": odds.get("moneyline", {}).get("home", {}).get("close", {}).get("odds"),
        "away_moneyline": odds.get("moneyline", {}).get("away", {}).get("close", {}).get("odds"),
        "home_spread_line": odds.get("pointSpread", {}).get("home", {}).get("close", {}).get("line"),
        "home_spread_odds": odds.get("pointSpread", {}).get("home", {}).get("close", {}).get("odds"),
        "away_spread_line": odds.get("pointSpread", {}).get("away", {}).get("close", {}).get("line"),
        "away_spread_odds": odds.get("pointSpread", {}).get("away", {}).get("close", {}).get("odds"),
        "total_over_odds": odds.get("total", {}).get("over", {}).get("close", {}).get("odds"),
        "total_under_odds": odds.get("total", {}).get("under", {}).get("close", {}).get("odds"),
    }


def _breakeven_probs_expr(col: pl.Expr) -> pl.Expr:
    return pl.when(col > 0).then(100 / (col + 100)).otherwise(-col / (-col + 100))


def main(
    sport: str = "basketball",
    league: str = "mens-college-basketball",
):
    odds_list: list[dict[str, Any]] = []
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
    dates = ["20260319", "20260320"]  # hard-coded; maybe can make more flexible later
    for date in dates:
        response = requests.get(url, params={"dates": date})
        data = response.json()
        for event in data.get("events", []):
            for competition in event.get("competitions", []):
                for odds in competition.get("odds", []):
                    odds_list.append(extract_odds(odds))

    if odds_list:
        df = (
            pl.DataFrame(odds_list)
            .with_columns(
                home_breakeven_prob=_breakeven_probs_expr(
                    pl.col("home_moneyline").str.replace("+", "", literal=True).cast(pl.Float64)
                ),
                away_breakeven_prob=_breakeven_probs_expr(
                    pl.col("away_moneyline").str.replace("+", "", literal=True).cast(pl.Float64)
                ),
            )
            .with_columns(
                home_win_prob=pl.col("home_breakeven_prob")
                / (pl.col("home_breakeven_prob") + pl.col("away_breakeven_prob")),
                away_win_prob=pl.col("away_breakeven_prob")
                / (pl.col("home_breakeven_prob") + pl.col("away_breakeven_prob")),
            )
        )
        df.write_csv(OUTPUT_DIR / "M/espn_odds.csv")
