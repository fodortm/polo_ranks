import pandas as pd

from domain.service import build_team_resume
from domain.stats import compute_stats, compute_sos


def test_build_team_resume_orders_wins_and_losses_deterministically():
    games = pd.DataFrame(
        [
            {"team1": "A", "score1": 5, "team2": "B", "score2": 3, "date": "2026-01-01", "site": "home", "opp_rank": 1},
            {"team1": "A", "score1": 4, "team2": "C", "score2": 2, "date": "2026-01-02", "site": "away", "opp_rank": 2},
            {"team1": "A", "score1": 2, "team2": "D", "score2": 1, "date": "2026-01-03", "site": "neutral", "opp_rank": 3},
            {"team1": "A", "score1": 1, "team2": "E", "score2": 3, "date": "2026-01-04", "site": "home", "opp_rank": 4},
            {"team1": "A", "score1": 2, "team2": "F", "score2": 4, "date": "2026-01-05", "site": "away", "opp_rank": 5},
            {"team1": "A", "score1": 2, "team2": "G", "score2": 5, "date": "2026-01-06", "site": "home"},
        ]
    )
    stats, h2h = compute_stats(games)
    sos = compute_sos(stats)
    bcar = {"B": 90.0, "C": 85.0, "D": 80.0, "E": 82.0, "F": 75.0}

    resume = build_team_resume("A", games, bcar, stats, h2h, sos)

    assert resume["sections"] == ["Top wins", "Worst losses", "Recent form", "Strength of schedule"]
    assert resume["summary"]["record"] == "3-3-0"
    assert [w["opponent"] for w in resume["top_wins"]][:3] == ["B", "C", "D"]
    assert [l["opponent"] for l in resume["worst_losses"]][:3] == ["F", "E", "G"]
    assert resume["top_wins"][0]["location"] == "home"
    assert resume["top_wins"][0]["match_date"] == "2026-01-01"
    assert resume["top_wins"][0]["opp_rank_at_match"] == 1


def test_build_team_resume_stable_tiebreak_order_and_empty_states():
    games = pd.DataFrame(
        [
            {"team1": "A", "score1": 3, "team2": "X", "score2": 2, "date": "2026-02-01"},
            {"team1": "A", "score1": 3, "team2": "W", "score2": 2, "date": "2026-02-02"},
            {"team1": "A", "score1": 3, "team2": "Y", "score2": 4, "date": "2026-02-03"},
            {"team1": "A", "score1": 3, "team2": "Z", "score2": 4, "date": "2026-02-04"},
        ]
    )
    stats, h2h = compute_stats(games)
    sos = compute_sos(stats)
    bcar = {"W": 80.0, "X": 80.0, "Y": 70.0, "Z": 70.0}

    resume = build_team_resume("A", games, bcar, stats, h2h, sos)

    assert [w["opponent"] for w in resume["top_wins"]] == ["W", "X"]
    assert [l["opponent"] for l in resume["worst_losses"]] == ["Y", "Z"]

    no_games = pd.DataFrame([], columns=["team1", "score1", "team2", "score2"])
    empty_stats, empty_h2h = compute_stats(no_games)
    empty_resume = build_team_resume("A", no_games, {}, empty_stats, empty_h2h, {})
    assert empty_resume["empty_states"]["top_wins"] == "No wins available yet."
    assert empty_resume["empty_states"]["recent_form"] == "No matches logged yet."
