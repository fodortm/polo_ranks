import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from domain.matchup_model import PoissonAttackDefenseMatchupModel


def _load_matchup_games() -> pd.DataFrame:
    return pd.read_csv("tests/data/matchup_poisson_fixtures.csv")


def test_probabilities_sum_to_one():
    model = PoissonAttackDefenseMatchupModel().fit(_load_matchup_games())
    out = model.predict_matchup("Alpha", "Bravo", venue="neutral", max_goals=10)
    total = out["p_win"] + out["p_draw"] + out["p_loss"]
    assert abs(total - 1.0) < 1e-6


def test_expected_goals_positive_and_venue_shift_directional():
    model = PoissonAttackDefenseMatchupModel().fit(_load_matchup_games())
    neutral = model.predict_matchup("Alpha", "Bravo", venue="neutral")
    home = model.predict_matchup("Alpha", "Bravo", venue="home")
    away = model.predict_matchup("Alpha", "Bravo", venue="away")

    assert neutral["expected_goals_a"] > 0 and neutral["expected_goals_b"] > 0
    assert home["expected_goals_a"] > neutral["expected_goals_a"] > away["expected_goals_a"]


def test_top_scorelines_sorted_by_probability():
    model = PoissonAttackDefenseMatchupModel().fit(_load_matchup_games())
    out = model.predict_matchup("Alpha", "Charlie", venue="home")

    probs = [row["probability"] for row in out["top_scorelines"]]
    assert probs == sorted(probs, reverse=True)


def test_interval_outputs_are_ordered_and_non_empty():
    model = PoissonAttackDefenseMatchupModel().fit(_load_matchup_games())
    out = model.predict_matchup("Bravo", "Delta", venue="neutral")

    keys = ["goal_diff_interval_50", "goal_diff_interval_80", "total_goals_interval_50", "total_goals_interval_80"]
    for key in keys:
        interval = out[key]
        assert isinstance(interval, tuple)
        assert len(interval) == 2
        assert interval[0] <= interval[1]
