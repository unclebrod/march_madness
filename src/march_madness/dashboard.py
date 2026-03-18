"""Streamlit dashboard for March Madness analysis."""

import streamlit as st

from march_madness.settings import OUTPUT_DIR

ROOT = "src/march_madness/pages"


def main():
    st.set_page_config(
        page_title="March Madness",
        page_icon=":basketball:",
        layout="wide",
    )

    st.title("March Madness Dashboard")

    if "league" not in st.session_state:
        st.session_state.league = "M"
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = None
    league = st.sidebar.radio("League:", ["M", "W"])
    model = st.sidebar.radio("Model:", ["ppp", "elo"])

    st.session_state.league = league
    st.session_state.model = model

    st.session_state.output_dir = OUTPUT_DIR / league / model
    st.session_state.output_base = OUTPUT_DIR

    pages = [
        st.Page(
            f"{ROOT}/coefficients.py",
            title="Model Coefficients",
            icon="📊",
        ),
        st.Page(
            f"{ROOT}/ratings.py",
            title="Team-Season Ratings",
            icon="📈",
        ),
        st.Page(
            f"{ROOT}/diagnostics.py",
            title="Model Diagnostics",
            icon="🔦",
        ),
        st.Page(
            f"{ROOT}/matchups.py",
            title="Matchups",
            icon="🤼",
        ),
        st.Page(
            f"{ROOT}/advance.py",
            title="Advancement",
            icon="🥇",
        ),
    ]

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
