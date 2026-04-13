# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair",
#     "marimo",
#     "polars",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import Any

    import altair as alt
    import marimo as mo
    import polars as pl

    return Any, alt, mo, pl


@app.cell
def _(alt):
    alt.renderers.set_embed_options(theme="fivethirtyeight")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # March Machine Learning Mania
    ---
    Broderick Turner
    """)
    return


@app.cell
def _(mo):
    image_path = mo.notebook_location() / "public" / "bball-logo.png"
    mo.image(str(image_path))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This site provides an overview of my yearly submission to [Kaggle's March Machine Learning Mania Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026). In the 2026 competition, I finished 129 out of 3462 competitors, and I'm hoping to do be even better in the years to come. This notebook contains:

    - 🔍 An overview of how my model works
    - 📊 Visuals of model outputs
    - 🏀 A lookup tool for matchup projections

    This notebook/site were created with [Marimo](https://marimo.io/) notebook, an interactive and lightweight Python environment designed for enhanced data exploration and storytelling. Plots are created using [Altair](https://altair-viz.github.io/). You can skip between sections using the menu (lines) on the right. It is best viewed on a desktop.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Index

    **Part 1**: [Context](#1-context)
    - [1.1 Model Overview](#11-model-overview)
    - [1.2 About the Data](#12-about-the-data)
    - [1.3 References](#13-references)

    **Part 2**: [Interactive Dashboard](#2-interactive-dashboard)
    - [2.1 What to Know](#21-what-to-know)
    - [2.2 Matchups](#22-matchups)
    - [2.3 Advancement Probabilities](#23-advancement-probabilities)
    - [2.4 Team Ratings](#24-team-ratings)
    - [2.5 Coefficient Inspection](#25-coefficient-inspection)

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Context

    Learn more about the context of the project, including modeling and data considerations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Model Overview
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model is a probabilistic model, built using the [numpyro](https://num.pyro.ai/en/stable/) library. For each game, I calculate the number of possessions from the box score data using [Ken Pomeroy's methodology](https://kenpom.com/blog/the-possession/). The model estimates a few key ratings (random effects) for each team, all of which control for the opponent:
    - 🚀 Offensive (points scored per possession)
    - 🤺 Defensive (points scored against per possession)
    - 👟 Pace (possessions per minute)

    Points per possession are a linear combination of a team's offensive rating, the opponent's defensive rating, and fixed effects (including features for rest, travel, and tournament indicators). These same fixed effects also inform the pace ratings, though with different coefficients. Home court advantage is calculated as a random effect to account for factors such as altitude.

    The model uses these ratings in order to estimate each team's score for a game. Possessions are modeled using a Poisson distribution; each team's score is modeled using a negative binomial which is similar to the Poisson except it has an extra parameter that allows the variance to be different than the mean.

    Ratings are on a season-team level. They are modeled using a Gaussian walk, which allows a given season to serve as a prior for the next.

    There is also some within-season weighting so that games at the end of the season count a bit more than games at the beginning.

    The model data is on the game-team level, meaning there are two rows present for a given game where each team is represented once as the defensive team and once as the offensive.

    The models for men's and women's basketball are trained separately on their respective data sources, but are otherwise exactly the same. Hyperparameters were tuned on men's data but are used for both men's and women's, largely for convenience and because of local hardward limitations.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 About the Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Kaggle provides a wealth of data](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) to competition entrants. These models are primarily trained using box score data (including regular season, secondary tournament, and tournament results). It does not account for things like tournament seeding in hopse of focusing purely on what the model belives is the quality of the teams.

    Outside of the provided data, I leverage the game-city level data. Using [Google's geocoding API](https://developers.google.com/maps/documentation/geocoding/guides-v3/overview), I look up where games occur as well as where each team is primarily located. This data is used to derive travel features (both distance traveled as well as changes in elevation).

    Some versions of the model have considered coaches and conference information, but as of now these are not incorporated.
    ___
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 References
    - 🥈 [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview) - the Kaggle competition site
    - 🌎 [Google Geocoding API](https://developers.google.com/maps/documentation/geocoding/guides-v3/overview) - for location data used in the model
    - 🔥 [NumPyro Documentation](https://num.pyro.ai/en/stable/) - great documentation with practical probabilistic programming vignettes
    - 🏀 [KenPom Ratings](https://kenpom.com/) - a constant sanity check to make sure ratings were reasonable
    - 🍃 [Marimo Gallery](https://marimo.io/gallery/) - awesome examples showcasing marimo's capabilities

    ---
    """)
    return


@app.cell
def _(mo):
    league = mo.ui.radio(
        label="League:",
        options={
            "Men's": "M",
            "Women's": "W",
        },
        value="Men's",
    )
    model = mo.ui.radio(
        label="Model:",
        options={
            "Points Per Possession": "ppp",
            # TODO: will fill these out in time
        },
        value="Points Per Possession",
    )
    mo.sidebar(
        mo.vstack(
            [
                league,
                model,
            ]
        )
    )
    return league, model


@app.cell
def _(league, mo, model):
    output_dir = mo.notebook_location() / "public" / league.value / model.value
    SEASON = 2026
    return SEASON, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Interactive Dashboard

    Inspect and interact with model outputs to see how the teams in the tournament are viewed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 What to Know

    - ⬅️ The global settings in the sidebar to the left apply to all of the below insights. Toggle these as desired.
    - 🧲 Changing these global settings, or any of the section-specific ones, will automatically update the visuals for you.
    - 🧰 Most plots have tooltips, so you can hover over a point and inspect the associated values.
    - 🚨 If something breaks, send me a message! I'll take a look.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Matchups

    Take a look at predicted outcomes for any of the 2025-2026 potential tournament matchups. A negative spread means that the first listed team is favored; a positive spread means they are an underdog. Note that there may be slight differences among totals/spreads/points because of how samples are collected.
    """)
    return


@app.cell
def _(output_dir, pl):
    matchup_df = pl.read_csv(str(output_dir / "preds.csv"))
    team_df = pl.concat(
        [
            matchup_df.select(
                pl.col("team1_name").alias("team_name"),
                pl.col("team1_id").alias("team_id"),
            ),
            matchup_df.select(pl.col("team2_name").alias("team_name"), pl.col("team2_id").alias("team_id")),
        ],
        how="vertical",
    ).unique()
    team_map = dict(zip(team_df["team_name"], team_df["team_id"], strict=True))
    return matchup_df, team_df, team_map


@app.cell
def _(Any, pl):
    def get_item(df: pl.DataFrame, col: str) -> Any:
        return df.select(pl.col(col))[col].item()

    return (get_item,)


@app.cell
def _(team_df):
    team_names = team_df["team_name"].unique().sort().to_list()
    return (team_names,)


@app.cell
def _(mo, team_names):
    first_team = mo.ui.dropdown(
        options=team_names,
        value="Duke",
        allow_select_none=False,
    )
    return (first_team,)


@app.cell
def _(first_team, mo, team_names):
    second_team = mo.ui.dropdown(
        options=[x for x in team_names if x != first_team.value],
        value="North Carolina",
        allow_select_none=False,
    )
    return (second_team,)


@app.cell
def _(first_team, mo, second_team):
    mo.md(f"""
    {first_team} vs. {second_team}
    """)
    return


@app.cell
def _(SEASON, first_team, get_item, matchup_df, pl, second_team, team_map):
    team1_id = team_map[first_team.value]
    team2_id = team_map[second_team.value]
    teams_in_order = sorted([team1_id, team2_id])
    matchup = matchup_df.filter(pl.col("ID").eq(f"{SEASON}_{teams_in_order[0]}_{teams_in_order[1]}"))
    team1_name = get_item(matchup, "team1_name")
    team2_name = get_item(matchup, "team2_name")
    return matchup, team1_name, team2_name


@app.cell
def _(get_item, matchup, mo):
    def get_team_results_element(team_number: int, team_name: str):
        return mo.vstack(
            [
                mo.md(f"**{team_name.upper()}**"),
                mo.md(f"Win Probability: {round(get_item(matchup, f'team{team_number}_win_prob'), 3)}"),
                mo.md(f"Score: {round(get_item(matchup, f'team{team_number}_score'), 2)}"),
                mo.md(f"Points Per Possession: {round(get_item(matchup, f'team{team_number}_ppp'), 2)}"),
            ]
        )

    return (get_team_results_element,)


@app.cell
def _(get_item, get_team_results_element, matchup, mo, team1_name, team2_name):
    team_1_results = get_team_results_element(team_number=1, team_name=team1_name)
    team_2_results = get_team_results_element(team_number=2, team_name=team2_name)
    game_level_results = mo.vstack(
        [
            mo.md("**GAME PROJECTIONS**"),
            mo.md(f"Spread: {round(get_item(matchup, 'spread'), 2)}"),
            mo.md(f"Total: {round(get_item(matchup, 'total'), 2)}"),
            mo.md(f"Number of Possessions: {round(get_item(matchup, 'possessions'), 2)}"),
        ]
    )
    matchup_display = mo.hstack(
        [
            team_1_results,
            team_2_results,
            game_level_results,
        ]
    )
    return (matchup_display,)


@app.cell
def _(matchup_display):
    matchup_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 Advancement Probabilities

    See the probabilities of teams advancing to a given round based on model matchup predictions.
    """)
    return


@app.cell
def _(output_dir, pl):
    advance_df = pl.read_csv(str(output_dir / "advance.csv")).rename({"": "team_name"})
    return (advance_df,)


@app.cell
def _():
    round_map = {
        "Round of 64": "R0",
        "Round of 32": "R1",
        "Sweet 16": "R2",
        "Elite 8": "R3",
        "Final Four": "R4",
        "Championship": "R5",
        "Champion": "R6",
    }
    inverse_round_map = {v: k for k, v in round_map.items()}
    return inverse_round_map, round_map


@app.cell
def _(mo, round_map):
    round_selection = mo.ui.radio(
        label="Select a round (probability of making it to the selected round):",
        options=round_map,
        value="Champion",
    )
    return (round_selection,)


@app.cell
def _(alt, mo, pl):
    def create_advancement_plot(df: pl.DataFrame, rnd: str, inverse_round_map: dict[str, str]):
        chart = (
            alt.Chart(df.sort(rnd, descending=True))
            .mark_bar()
            .encode(
                y=alt.Y("team_name", sort=alt.EncodingSortField(field=rnd, order="descending"), title="Team Name"),
                x=alt.X(rnd, title="Advancement Probability"),
                # tooltip=[alt.Tooltip(f"{rnd}:Q")],  # TODO: tooltip not working in sandbox mode for some reason
            )
            .properties(
                width="container",
                title=f"Probability of Advacement to the {inverse_round_map[rnd]}",
            )
        )
        return mo.ui.altair_chart(
            chart=chart,
            chart_selection=False,
            legend_selection=False,
        )

    return (create_advancement_plot,)


@app.cell
def _(advance_df, create_advancement_plot, inverse_round_map, round_selection):
    create_advancement_plot(
        df=advance_df,
        rnd=round_selection.value,
        inverse_round_map=inverse_round_map,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.4 Team Ratings

    These ratings are team-season specific. Choose the season and ratings of interest to display.
    """)
    return


@app.cell
def _(output_dir, pl):
    ratings_df = pl.read_csv(str(output_dir / "season_team_coefs.csv"))
    return (ratings_df,)


@app.cell
def _(SEASON, mo, ratings_df):
    season_options = ratings_df["season"].unique().sort(descending=True).to_list()
    team_options = ratings_df["team_name"].unique().sort(descending=False).to_list()

    season = mo.ui.dropdown(
        label="Season (the year in which the season ended):",
        options=season_options,
        value=SEASON,
        allow_select_none=False,
    )
    teams = mo.ui.multiselect(
        label="Teams (by default, all are included):",
        options=team_options,
        value=team_options,
    )
    mo.vstack(
        [
            season,
            teams,
        ]
    )
    return season, teams


@app.cell
def _(alt, mo, pl):
    def create_team_season_rating_plot(df: pl.DataFrame, season: int, rating: str, teams: list[str]):
        season_team_df = df.filter(
            pl.col("season").eq(season), pl.col("name").eq(rating), pl.col("team_name").is_in(teams)
        ).sort("percentile_500", descending=True)
        tooltip = [
            "name",
            "percentile_025",
            "percentile_500",
            "percentile_975",
        ]
        bar = (
            alt.Chart(season_team_df)
            .mark_errorbar()
            .encode(
                alt.X("percentile_025:Q").title("Value"),
                alt.X2("percentile_975:Q"),
                alt.Y("team_name:N", sort=alt.EncodingSortField(field="-percentile_500")).title("Team Name"),
                tooltip=tooltip,
            )
        )
        point = (
            alt.Chart(season_team_df)
            .mark_point(
                filled=True,
                color="black",
            )
            .encode(
                alt.X("percentile_500:Q"),
                alt.Y("team_name:N", sort=alt.EncodingSortField(field="-percentile_500")),
                tooltip=tooltip,
            )
        )
        zero_df = pl.DataFrame({"x": 0, "label": "x=0"})
        vline = alt.Chart(zero_df).mark_rule(color="black", strokeDash=[4, 2]).encode(x="x:Q")
        chart = (bar + point + vline).properties(
            title="Season Team Ratings (95% Credible Intervals)",
            width="container",
        )
        return mo.ui.altair_chart(
            chart=chart,
            chart_selection=False,
            legend_selection=False,
        )

    return (create_team_season_rating_plot,)


@app.cell
def _(mo):
    rating_options_map = {
        "Offense": "offense_team_rw",
        "Defense": "defense_team_rw",
        "Pace": "pace_team_rw",
    }
    rating = mo.ui.dropdown(
        options=rating_options_map,
        value="Offense",
        allow_select_none=False,
    )
    mo.md(f"""
    Select a rating to view its value for all available teams: {rating}
    """)
    return rating, rating_options_map


@app.cell
def _(create_team_season_rating_plot, rating, ratings_df, season, teams):
    create_team_season_rating_plot(
        df=ratings_df,
        season=season.value,
        rating=rating.value,
        teams=teams.value,
    )
    return


@app.cell
def _(alt, mo, pl):
    def create_comp_plot(df: pl.DataFrame, x: str, y: str, season: int, teams: list[str]):
        mapper = {
            "offense_team_rw": "Offense",
            "defense_team_rw": "Defense",
            "pace_team_rw": "Pace",
        }
        base_df = df.filter(
            pl.col("season").eq(season),
            pl.col("team_name").is_in(teams),
            pl.col("team_name").is_not_null(),
        )
        new_x, new_y = mapper[x], mapper[y]
        tooltip = ["team_name", new_x, new_y, "distance"]
        comp_df = (
            base_df.filter(pl.col("name").eq(x))
            .select(
                pl.col("percentile_500").alias(x),
                pl.col("team_name"),
            )
            .join(
                base_df.filter(pl.col("name").eq(y)).select(pl.col("percentile_500").alias(y), pl.col("team_name")),
                on="team_name",
                how="inner",
            )
            .rename(mapper, strict=False)
            .with_columns(
                distance=(pl.col(new_x).pow(2).add(pl.col(new_y).pow(2))).sqrt(),
            )
        )
        point = (
            alt.Chart(comp_df)
            .mark_point()
            .encode(
                alt.X(f"{new_x}:Q"),
                alt.Y(f"{new_y}:Q"),
                tooltip=tooltip,
            )
        )
        text_df = comp_df.filter(
            pl.col(new_x).ge(0),
            pl.col(new_y).ge(0),
        ).top_k(k=10, by="distance")
        text = (
            alt.Chart(text_df)
            .mark_text(dx=10, dy=10, size=8)
            .encode(
                alt.X(f"{new_x}:Q"),
                alt.Y(f"{new_y}:Q"),
                tooltip=tooltip + ["distance"],
                text="team_name",
            )
        )
        vline = (
            alt.Chart(pl.DataFrame({f"{new_x}": 0})).mark_rule(color="black", strokeDash=[4, 2]).encode(x=f"{new_x}:Q")
        )
        hline = (
            alt.Chart(pl.DataFrame({f"{new_y}": 0})).mark_rule(color="black", strokeDash=[4, 2]).encode(y=f"{new_y}:Q")
        )
        chart = (text + point + vline + hline).properties(title=f"{new_y} vs. {new_x}", width="container", height=600)
        return mo.ui.altair_chart(
            chart=chart,
            chart_selection=False,
            legend_selection=False,
        )

    return (create_comp_plot,)


@app.cell
def _(mo, rating_options_map):
    comp_x = mo.ui.dropdown(
        options=rating_options_map,
        value="Offense",
        allow_select_none=False,
    )
    comp_y = mo.ui.dropdown(
        options=rating_options_map,
        value="Defense",
        allow_select_none=False,
    )
    mo.md(f"""
    Compare one rating against another by selecting an x value ({comp_x}) and y value ({comp_y}).
    """)
    return comp_x, comp_y


@app.cell
def _(comp_x, comp_y, create_comp_plot, ratings_df, season, teams):
    create_comp_plot(
        df=ratings_df,
        x=comp_x.value,
        y=comp_y.value,
        season=season.value,
        teams=teams.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.5 Coefficient Inspection

    The model has a number of coefficients/parameters that are output. Take a look and see how much each of the contribute to the model. The scales of all of these are not the same, so I leave it up to you to decide what you want to explore. Here's what to know about the prefixes:

    - 🅱️ `beta` prefixes usually denote a fixed effect. `pace` are features used to calculate pace; `ppp`, points per possession.
    - 🔼 `sigma` prefixes usually denote a standard deviation from a particular distribution.
    - ➗ `global_mean` and `mu` coefficients usually denote some type of population/parameter average or mean.
    """)
    return


@app.cell
def _(output_dir, pl):
    coef_df = pl.read_csv(str(output_dir / "coefs.csv")).sort("percentile_500", descending=True)
    return (coef_df,)


@app.cell
def _(coef_df, mo):
    # Filter coefficients based on user selection
    coef_options = coef_df["name"].unique().sort().to_list()
    selected_coefs = mo.ui.multiselect(
        label="Select coefficients to display:",
        options=coef_options,
        value=[x for x in coef_options if x.startswith("beta")],
    )
    selected_coefs
    return (selected_coefs,)


@app.cell
def _(alt, mo, pl):
    def create_coef_plot_chart(df: pl.DataFrame, selected_coefs: list[str]):
        selected_coef_df = df.filter(pl.col("name").is_in(selected_coefs))
        tooltip = [
            "name",
            "percentile_025",
            "percentile_500",
            "percentile_975",
        ]
        bar = (
            alt.Chart(selected_coef_df)
            .mark_errorbar()
            .encode(
                alt.X("percentile_025:Q").title("Value"),
                alt.X2("percentile_975:Q"),
                alt.Y("name:N", sort=alt.EncodingSortField(field="-percentile_500")).title("Coefficient"),
                tooltip=tooltip,
            )
        )
        point = (
            alt.Chart(selected_coef_df)
            .mark_point(
                filled=True,
                color="black",
            )
            .encode(
                alt.X("percentile_500:Q").title("Value"),
                alt.Y("name:N", sort=alt.EncodingSortField(field="-percentile_500")).title("Coefficient"),
                tooltip=tooltip,
            )
        )
        vline = alt.Chart(pl.DataFrame({"x": 0})).mark_rule(color="black", strokeDash=[4, 2]).encode(x="x:Q")
        chart = (bar + point + vline).properties(
            title="Coefficient Values (95% Credible Intervals)",
            width="container",
        )
        return mo.ui.altair_chart(
            chart=chart,
            chart_selection=False,
            legend_selection=False,
        )

    return (create_coef_plot_chart,)


@app.cell
def _(coef_df, create_coef_plot_chart, selected_coefs):
    create_coef_plot_chart(df=coef_df, selected_coefs=selected_coefs.value)
    return


if __name__ == "__main__":
    app.run()
