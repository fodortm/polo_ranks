import pandas as pd
import streamlit as st

from app.ui import (
    _on_dashboard_time_window_change,
    _on_dashboard_whole_season_toggle,
    _to_schedule_adjusted_games,
    compute_schedule_adjusted_hybrid,
)


def test_to_schedule_adjusted_games_converts_schema():
    games = pd.DataFrame(
        [
            {"team1": "Alpha", "team2": "Bravo", "score1": 3, "score2": 2},
            {"team1": "Bravo", "team2": "Charlie", "score1": 1, "score2": 1},
        ]
    )

    converted = _to_schedule_adjusted_games(games)

    assert list(converted.columns) == ["team_a", "team_b", "goals_a", "goals_b"]
    assert len(converted) == 2


def test_compute_schedule_adjusted_hybrid_smoke():
    games = pd.DataFrame(
        [
            {"team1": "Alpha", "team2": "Bravo", "score1": 3, "score2": 2},
            {"team1": "Alpha", "team2": "Charlie", "score1": 2, "score2": 1},
            {"team1": "Bravo", "team2": "Charlie", "score1": 1, "score2": 1},
        ]
    )

    out = compute_schedule_adjusted_hybrid(games, params={})

    assert out["order"]
    assert not out["table"].empty
    assert out["model"] is not None


def test_dashboard_timeframe_controls_smoke():
    st.session_state.clear()
    st.query_params.clear()

    st.session_state["dashboard_time_window"] = "Last 4 weeks"
    _on_dashboard_time_window_change()
    assert st.query_params["whole_season"] == "0"

    st.session_state["dashboard_whole_season_toggle"] = True
    _on_dashboard_whole_season_toggle()
    assert st.session_state["dashboard_time_window"] == "All"
    assert st.query_params["whole_season"] == "1"

    st.session_state["dashboard_whole_season_toggle"] = False
    _on_dashboard_whole_season_toggle()
    assert st.session_state["dashboard_time_window"] == "Last 4 weeks"
    assert st.query_params["whole_season"] == "0"
