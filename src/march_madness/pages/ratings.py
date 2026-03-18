import altair as alt
import polars as pl
import streamlit as st

from march_madness.utils import current_season

st.title("Team-Season Ratings")

ratings_df = pl.read_csv(f"{st.session_state.output_dir}/season_team_coefs.csv")

# Filters for seasons
season_options = ratings_df["season"].unique().sort(descending=True).to_list()
season = st.sidebar.radio(
    "Season:",
    season_options,
    index=season_options.index(current_season() if current_season() in season_options else season_options[-1]),
)

# Filter for a coefficient
coef_options = ratings_df["name"].unique().sort(descending=False).to_list()
coef = st.sidebar.radio(
    "Coefficient:",
    coef_options,
    index=coef_options.index("offense_team_rw" if "offense_team_rw" in coef_options else coef_options[0]),
)
ratings_df = ratings_df.filter(
    pl.col("season").eq(season),
    pl.col("name").eq(coef),
).sort("percentile_500", descending=True)

bar = (
    alt.Chart(ratings_df)
    .mark_errorbar()
    .encode(
        alt.X("percentile_025:Q").title("Value"),
        alt.X2("percentile_975:Q"),
        alt.Y("team_name:N", sort=alt.EncodingSortField(field="-percentile_500")).title("Team Name"),
    )
)
point = (
    alt.Chart(ratings_df)
    .mark_point(
        filled=True,
        color="black",
    )
    .encode(
        alt.X("percentile_500:Q"),
        alt.Y("team_name:N", sort=alt.EncodingSortField(field="-percentile_500")),
    )
)
chart = bar + point
st.altair_chart(chart, use_container_width=True, theme=None)

st.dataframe(
    ratings_df.select(
        "name",
        "team_name",
        "mean",
        "percentile_025",
        "percentile_500",
        "percentile_975",
        "std",
        # "first_d1_season",
        # "last_d1_season",
    ).sort("percentile_500", descending=True)
)
