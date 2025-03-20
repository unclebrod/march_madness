import altair as alt
import polars as pl
import streamlit as st

from march_madness.model import MarchMadnessModel

st.title("Model Diagnostics")

model = MarchMadnessModel.load(st.session_state.league)

burnoff = 500
losses = model.results.losses.tolist()[500:]
max_domain = burnoff + len(losses)
loss_df = pl.DataFrame(
    {
        "x": list(range(burnoff, max_domain)),
        "loss": losses,
    }
)

chart = (
    alt.Chart(loss_df)
    .mark_line()
    .encode(
        alt.X("x:Q", scale=alt.Scale(zero=False, domain=[burnoff, max_domain]), title="Iteration"),
        alt.Y("loss:Q", title="Loss"),
    )
)

st.altair_chart(chart, use_container_width=True, theme=None)
