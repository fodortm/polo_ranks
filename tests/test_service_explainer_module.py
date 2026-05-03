import pandas as pd

from domain.service import build_team_explainer_card
from domain.stats import compute_stats, compute_sos


def test_explainer_handles_missing_data_defaults():
    card = build_team_explainer_card("Unknown", {}, {}, {}, {})
    assert card["confidence_label"] == "Low"
    assert card["recency_label"] == "Early sample"
    assert len(card["factors"]) == 5


def test_explainer_tied_team_record_and_statements():
    games = pd.DataFrame([
        {"team1": "A", "score1": 2, "team2": "B", "score2": 2},
        {"team1": "A", "score1": 1, "team2": "C", "score2": 1},
        {"team1": "A", "score1": 3, "team2": "D", "score2": 3},
    ])
    stats, h2h = compute_stats(games)
    sos = compute_sos(stats)
    card = build_team_explainer_card("A", stats, sos, h2h, {"A": {"imputed": 0, "games": 3}})
    assert card["record"] == "0-0-3"
    assert all("summary" in f for f in card["factors"])


def test_explainer_sparse_history_sets_low_confidence():
    games = pd.DataFrame([
        {"team1": "A", "score1": 4, "team2": "B", "score2": 0},
    ])
    stats, h2h = compute_stats(games)
    sos = compute_sos(stats)
    card = build_team_explainer_card("A", stats, sos, h2h, {"A": {"imputed": 1, "games": 1}})
    assert card["confidence_label"] == "Low"
    assert card["recency_label"] == "Early sample"
