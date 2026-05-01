import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from domain.hybrid_ranking import HybridRankingConfig, ScheduleAdjustedGoalStrengthRanker


def _load_hybrid_games() -> pd.DataFrame:
    return pd.read_csv("tests/data/hybrid_ranking_fixtures.csv")


def test_prior_shrinkage_tiny_vs_larger_samples():
    games = pd.DataFrame(
        [{"team_a": "Tiny", "team_b": "Opp1", "goals_a": 4, "goals_b": 2, "venue": "neutral"}]
        + [{"team_a": "Large", "team_b": "Opp2", "goals_a": 4, "goals_b": 2, "venue": "neutral"} for _ in range(10)]
    )
    model = ScheduleAdjustedGoalStrengthRanker(HybridRankingConfig(ridge_lambda=4.0, k0=8.0)).fit(games)
    table = model.rankings_table().set_index("team")

    assert table.loc["Tiny", "games"] < table.loc["Large", "games"]
    assert abs(table.loc["Tiny", "prior"]) < abs(table.loc["Large", "prior"])


def test_robust_margin_transform_monotonicity_and_blowout_compression():
    cfg = HybridRankingConfig(margin_scale=1.0)
    transformed = [
        (gd, float((1 if gd >= 0 else -1) * np.log1p(abs(gd) / cfg.margin_scale)))  # type: ignore[attr-defined]
        for gd in [0, 1, 2, 4, 8]
    ]
    vals = [x[1] for x in transformed]

    assert vals == sorted(vals)
    assert ((vals[4] - vals[3]) / 4.0) < (vals[2] - vals[1])


def test_ridge_anchoring_moves_toward_priors():
    games = _load_hybrid_games()
    weak = ScheduleAdjustedGoalStrengthRanker(HybridRankingConfig(ridge_lambda=0.01, k0=8.0)).fit(games)
    strong = ScheduleAdjustedGoalStrengthRanker(HybridRankingConfig(ridge_lambda=100.0, k0=8.0)).fit(games)

    weak_gap = abs(weak.theta_[weak.team_to_idx_["Alpha"]] - weak.priors_[weak.team_to_idx_["Alpha"]])
    strong_gap = abs(strong.theta_[strong.team_to_idx_["Alpha"]] - strong.priors_[strong.team_to_idx_["Alpha"]])

    assert strong_gap < weak_gap


def test_sos_dynamic_weighting_increases_with_games_played():
    games = _load_hybrid_games()
    model = ScheduleAdjustedGoalStrengthRanker(HybridRankingConfig(k0=8.0, lambda_sos_max=0.25)).fit(games)

    alpha = model.team_to_idx_["Alpha"]
    echo = model.team_to_idx_["Echo"]
    assert model.games_played_[echo] > model.games_played_[alpha]
    assert model.lambda_sos_[echo] > model.lambda_sos_[alpha]


def test_sov_rewards_quality_residual_more_than_weak_expected_blowout():
    games = pd.DataFrame(
        [
            {"team_a": "Strong", "team_b": "Weak", "goals_a": 3, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Strong", "team_b": "Weak", "goals_a": 3, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Hero", "team_b": "Strong", "goals_a": 2, "goals_b": 1, "venue": "neutral"},
            {"team_a": "Hero", "team_b": "Weak", "goals_a": 2, "goals_b": 0, "venue": "neutral"},
        ]
    )
    model = ScheduleAdjustedGoalStrengthRanker(HybridRankingConfig(ridge_lambda=8.0, lambda_sov=0.5, lambda_var=0.0)).fit(games)

    sov = model.rankings_table().set_index("team")["sov"]
    assert sov["Hero"] > sov["Strong"]


def test_sov_repeated_weak_wins_do_not_beat_fewer_high_surprise_strong_wins():
    games = pd.DataFrame(
        [
            {"team_a": "Power", "team_b": "WeakA", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Power", "team_b": "WeakB", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Power", "team_b": "WeakC", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Power", "team_b": "WeakD", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Power", "team_b": "WeakE", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Upset", "team_b": "Power", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Upset", "team_b": "Power", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
        ]
    )
    teams = sorted(set(games["team_a"]).union(games["team_b"]))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    theta = np.array([1.5 if team == "Power" else (-1.0 if team.startswith("Weak") else 0.0) for team in teams], dtype=float)
    residuals = np.array([0.08, 0.09, 0.10, 0.08, 0.09, 0.45, 0.40], dtype=float)

    ranker = ScheduleAdjustedGoalStrengthRanker(HybridRankingConfig())
    sov = ranker._compute_sov(games, residuals=residuals, theta=theta, team_to_idx=team_to_idx)

    assert sov[team_to_idx["Upset"]] > sov[team_to_idx["Power"]]


def test_sov_is_non_negative_and_stable_with_duplicated_weak_games():
    base_games = pd.DataFrame(
        [
            {"team_a": "Power", "team_b": "WeakA", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Power", "team_b": "WeakB", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
            {"team_a": "Upset", "team_b": "Power", "goals_a": 1, "goals_b": 0, "venue": "neutral"},
        ]
    )
    duplicated_games = pd.concat([base_games, base_games.iloc[[0, 1]]], ignore_index=True)

    teams = sorted(set(duplicated_games["team_a"]).union(duplicated_games["team_b"]))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    theta = np.array([1.5 if team == "Power" else (-1.0 if team.startswith("Weak") else 0.0) for team in teams], dtype=float)

    base_residuals = np.array([0.1, 0.1, 0.4], dtype=float)
    duplicated_residuals = np.array([0.1, 0.1, 0.4, 0.1, 0.1], dtype=float)

    ranker = ScheduleAdjustedGoalStrengthRanker(HybridRankingConfig())
    base_sov = ranker._compute_sov(base_games, residuals=base_residuals, theta=theta, team_to_idx=team_to_idx)
    duplicated_sov = ranker._compute_sov(duplicated_games, residuals=duplicated_residuals, theta=theta, team_to_idx=team_to_idx)

    assert (base_sov >= 0).all()
    assert (duplicated_sov >= 0).all()
    assert abs(duplicated_sov[team_to_idx["Power"]] - base_sov[team_to_idx["Power"]]) < 1e-9


def test_uncertainty_bounds_and_ci_are_sensible():
    games = _load_hybrid_games()
    model = ScheduleAdjustedGoalStrengthRanker().fit(games)
    table = model.rankings_table()

    assert (table["rating_se"] > 0).all()
    assert (table["rating_ci_high"] > table["rating_ci_low"]).all()
    assert table["confidence"].between(0, 100).all()


def test_low_sample_teams_have_larger_se_than_high_information_teams():
    games = _load_hybrid_games()
    model = ScheduleAdjustedGoalStrengthRanker().fit(games)
    table = model.rankings_table().set_index("team")

    assert table.loc["Alpha", "games"] < table.loc["Echo", "games"]
    assert table.loc["Alpha", "rating_se"] > table.loc["Echo", "rating_se"]


def test_predict_matchup_requires_fit():
    model = ScheduleAdjustedGoalStrengthRanker()
    try:
        model.predict_matchup("Alpha", "Bravo")
        raise AssertionError("Expected RuntimeError when calling predict_matchup before fit")
    except RuntimeError as exc:
        assert "fit before calling predict_matchup" in str(exc)


def test_predict_matchup_validates_venue():
    games = _load_hybrid_games()
    model = ScheduleAdjustedGoalStrengthRanker().fit(games)

    try:
        model.predict_matchup("Alpha", "Bravo", venue="mars")  # type: ignore[arg-type]
        raise AssertionError("Expected ValueError for invalid venue")
    except ValueError as exc:
        assert "venue must be one of" in str(exc)


def test_predict_matchup_schema_and_deterministic_scoreline_ordering():
    games = _load_hybrid_games()
    model = ScheduleAdjustedGoalStrengthRanker().fit(games)
    pred = model.predict_matchup("Alpha", "Bravo", venue="neutral", max_goals=8)

    assert set(pred.keys()) == {
        "expected_goals_a",
        "expected_goals_b",
        "p_win",
        "p_draw",
        "p_loss",
        "top_scorelines",
        "goal_diff_interval_50",
        "goal_diff_interval_80",
        "total_goals_interval_50",
        "total_goals_interval_80",
        "matchup_confidence",
    }

    scorelines = pred["top_scorelines"]
    assert isinstance(scorelines, list)
    for idx in range(1, len(scorelines)):
        prev = scorelines[idx - 1]
        curr = scorelines[idx]
        prev_key = (-prev["probability"], prev["score_a"], prev["score_b"])
        curr_key = (-curr["probability"], curr["score_a"], curr["score_b"])
        assert prev_key <= curr_key
