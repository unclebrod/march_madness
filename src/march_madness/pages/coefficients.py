import altair as alt
import polars as pl
import streamlit as st

alt.renderers.set_embed_options(theme="dark")

st.title("Model Coefficients")

coefs = pl.read_csv(f"{st.session_state.output_dir}/coefs.csv").sort("value_50", descending=True)

bar = (
    alt.Chart(coefs)
    .mark_errorbar()
    .encode(
        alt.X("value_025:Q").title("Value"),
        alt.X2("value_975:Q"),
        alt.Y("coefficient:N", sort=alt.EncodingSortField(field="-value_50")).title("Coefficient"),
    )
)

point = (
    alt.Chart(coefs)
    .mark_point(
        filled=True,
        color="black",
    )
    .encode(
        alt.X("value_50:Q"),
        alt.Y("coefficient:N", sort=alt.EncodingSortField(field="-value_50")),
    )
)

chart = bar + point
st.altair_chart(chart, use_container_width=True, theme=None)

st.dataframe(coefs)
