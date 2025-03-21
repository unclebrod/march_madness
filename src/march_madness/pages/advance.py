import streamlit as st
import polars as pl
import altair as alt
from march_madness import OUTPUT_DIR, DATA_DIR

st.title("Bracket Advancement")

advance = pl.read_csv(OUTPUT_DIR / f"{st.session_state.output_dir}/advance.csv")
jl = pl.read_csv(DATA_DIR / f"{st.session_state.league.lower()}_kaggle_preds.csv")
kenpom = pl.read_csv(DATA_DIR / "AdvanceKenpom2025.csv")
rs = pl.read_csv(DATA_DIR / "AdvanceRS22025.csv")

rounds = [f"R{x}" for x in range(0, 7)]
round = st.sidebar.radio("Round:", options=rounds, index=len(rounds) - 1)

df_list = []
brackets = [
    advance,
    kenpom,
    rs,
    # jl,
]
names = [
    "BT",
    "KenPom",
    "RS2",
    # "JL",
]
for bracket, name in zip(brackets, names):
    # st.dataframe(bracket.rename({"": "Team"}))
    df = bracket.rename({"": "Team"}).select(pl.col("Team"), pl.col(round)).with_columns(Name=pl.lit(name)).with_columns(
        pl.col("Team").str.to_lowercase().str.replace_all(r"[^\w\s]", "").str.replace(" ", "")
    )
    df_list.append(df)

df = pl.concat(df_list, how="vertical")
mean_df = df.group_by("Team").agg(pl.col(round).mean()).with_columns(Name=pl.lit("Mean"))
df = pl.concat([df, mean_df], how="vertical")

if st.session_state.league == "W":
    df.filter(pl.col("Name").is_in(["JL", "BT"]))

chart = alt.Chart(df).mark_bar().encode(
    alt.Y("Team", sort=alt.EncodingSortField(field=round, op="mean", order="descending")),
    alt.X(round),
    alt.Color("Name"),
    alt.YOffset("Name"),
)

st.altair_chart(chart)
st.dataframe(df)

st.dataframe(advance)
