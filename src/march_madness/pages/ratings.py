import altair as alt
import polars as pl
import streamlit as st

st.title("Team-Season Ratings")

ratings = pl.read_csv(f"{st.session_state.output_dir}/ratings.csv")
season_options = ratings["season"].unique().sort(descending=True).to_list()
season = st.sidebar.radio(
    "Season:",
    season_options,
    index=season_options.index("2025"),
)
coef_options = ratings["coefficient"].unique().sort(descending=False).to_list()
coef = st.sidebar.radio(
    "Coefficient:",
    coef_options,
    index=coef_options.index("offense_ppp"),
)
ratings_df = ratings.filter(pl.col("season").eq(season), pl.col("coefficient").eq(coef)).sort(
    "value_50", descending=True
)

bar = (
    alt.Chart(ratings_df)
    .mark_errorbar()
    .encode(
        alt.X("value_025:Q").title("Value"),
        alt.X2("value_975:Q"),
        alt.Y("team_name:N", sort=alt.EncodingSortField(field="-value_50")).title("Team Name"),
    )
)

point = (
    alt.Chart(ratings_df)
    .mark_point(
        filled=True,
        color="black",
    )
    .encode(
        alt.X("value_50:Q"),
        alt.Y("team_name:N", sort=alt.EncodingSortField(field="-value_50")),
    )
)

chart = bar + point
st.altair_chart(chart, use_container_width=True, theme=None)

st.dataframe(ratings_df)
