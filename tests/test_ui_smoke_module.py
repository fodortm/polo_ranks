import pandas as pd

from app.ui import _to_schedule_adjusted_games, compute_schedule_adjusted_hybrid


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
