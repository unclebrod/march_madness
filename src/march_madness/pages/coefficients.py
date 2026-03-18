import altair as alt
import polars as pl
import streamlit as st

alt.renderers.set_embed_options(theme="dark")

st.title("Model Coefficients")

coef_df = pl.read_csv(f"{st.session_state.output_dir}/coefs.csv").sort("percentile_500", descending=True)

# Filter coefficients based on user selection
coef_options = coef_df["name"].unique().sort().to_list()
selected_coefs = st.multiselect("Select coefficients to display:", options=coef_options, default=coef_options)
coef_df = coef_df.filter(pl.col("name").is_in(selected_coefs))

bar = (
    alt.Chart(coef_df)
    .mark_errorbar()
    .encode(
        alt.X("percentile_025:Q").title("Value"),
        alt.X2("percentile_975:Q"),
        alt.Y("name:N", sort=alt.EncodingSortField(field="-percentile_500")).title("Coefficient"),
    )
)
point = (
    alt.Chart(coef_df)
    .mark_point(
        filled=True,
        color="black",
    )
    .encode(
        alt.X("percentile_500:Q"),
        alt.Y("name:N", sort=alt.EncodingSortField(field="-percentile_500")),
    )
)
chart = bar + point
st.altair_chart(chart, width="stretch", theme=None)

st.dataframe(
    coef_df.select(
        "name",
        "mean",
        "percentile_025",
        "percentile_500",
        "percentile_975",
        "std",
    ).sort("percentile_500", descending=True)
)
