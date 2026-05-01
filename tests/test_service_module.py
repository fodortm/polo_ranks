import pandas as pd

from domain.service import build_team_resume
from domain.stats import compute_stats, compute_sos


def test_build_team_resume_orders_wins_and_losses_deterministically():
    games = pd.DataFrame(
        [
            {"team1": "A", "score1": 5, "team2": "B", "score2": 3},
            {"team1": "A", "score1": 4, "team2": "C", "score2": 2},
            {"team1": "A", "score1": 2, "team2": "D", "score2": 1},
            {"team1": "A", "score1": 1, "team2": "E", "score2": 3},
            {"team1": "A", "score1": 2, "team2": "F", "score2": 4},
            {"team1": "A", "score1": 2, "team2": "G", "score2": 5},
        ]
    )
    stats, h2h = compute_stats(games)
    sos = compute_sos(stats)
    bcar = {"B": 90.0, "C": 85.0, "D": 80.0, "E": 82.0, "F": 75.0}

    resume = build_team_resume("A", games, bcar, stats, h2h, sos)

    assert resume["summary"]["record"] == "3-3-0"
    assert resume["summary"]["goal_diff"] == -2
    assert resume["summary"]["notable_streak"] == "L3"
    assert [w["opponent"] for w in resume["top_wins"]][:3] == ["B", "C", "D"]
    assert [l["opponent"] for l in resume["worst_losses"]][:3] == ["F", "E", "G"]
    assert [l["opponent"] for l in resume["best_losses"]][:3] == ["G", "E", "F"]
    assert [w["opponent"] for w in resume["worst_wins"]][:3] == ["D", "C", "B"]
