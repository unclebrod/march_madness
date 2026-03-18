from typing import Any

import polars as pl
import streamlit as st

from march_madness.loader import DataConfig, DataLoader
from march_madness.utils import current_season

SEASON = current_season()


def get_item(df: pl.DataFrame, col: str) -> Any:
    return df.select(pl.col(col))[col].item()


st.title("Predicted Matchups")

matchup_df = pl.read_csv(f"{st.session_state.output_dir}/preds.csv")

data_loader = DataLoader(league=st.session_state.league)
data_config = DataConfig()
teams = data_loader.load_data(data_config.teams)
team_names = teams["TeamName"].to_list()
team_ids = teams["TeamID"].to_list()

team_map = dict(zip(team_names, team_ids, strict=False))

with st.form("matchups"):
    team1 = st.selectbox("Team 1", options=team_names)
    team2 = st.selectbox("Team 2", options=team_names)
    submitted = st.form_submit_button("Submit")
    if submitted:
        if team1 == team2:
            st.warning("Please select two different teams.")
            st.stop()
        team1_id = team_map[team1]
        team2_id = team_map[team2]
        teams_in_order = sorted([team1_id, team2_id])
        matchup = matchup_df.filter(pl.col("ID").eq(f"{SEASON}_{teams_in_order[0]}_{teams_in_order[1]}"))
        # st.dataframe(matchup)

        cols = st.columns(2)
        team1 = get_item(matchup, "team1_name")
        team2 = get_item(matchup, "team2_name")
        with cols[0]:
            st.metric(f"{team1} Win Probability", round(get_item(matchup, "team1_win_prob"), 3), border=True)
            st.metric(
                f"{team1} Score (Mean)",
                round(get_item(matchup, "team1_score"), 2),
                border=True,
            )
            # st.metric(f"{team1} Score (Median)", get_item(matchup, "team1_score"), border=True)
            # st.metric(f"{team1} Score (2.5%)", get_item(matchup, "team1_score_025"), border=True)
            # st.metric(f"{team1} Score (97.5%)", get_item(matchup, "team1_score_975"), border=True)
        with cols[1]:
            st.metric(f"{team2} Win Probability", round(get_item(matchup, "team2_win_prob"), 3), border=True)
            st.metric(
                f"{team2} Score (Mean)",
                round(get_item(matchup, "team2_score"), 2),
                border=True,
            )
            # st.metric(f"{team2} Score (Median)", get_item(matchup, "team2_score"), border=True)
            # st.metric(f"{team2} Score (2.5%)", get_item(matchup, "team2_score_025"), border=True)
            # st.metric(f"{team2} Score (97.5%)", get_item(matchup, "team2_score_975"), border=True)

        estimated_possessions = get_item(matchup, "possessions")
        estimated_spread = get_item(matchup, "spread")
        estimated_total = get_item(matchup, "total")

        st.metric("Estimated Spread", round(estimated_spread, 2), border=True)
        st.metric("Estimated Total", round(estimated_total, 2), border=True)
        st.metric("Estimated Possessions", value=estimated_possessions, border=True)
