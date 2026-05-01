import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app import ui


def _sample_stats():
    return {
        "A": {"wins": 8, "losses": 2, "ties": 0, "gf": 30, "ga": 15, "games": 10, "win_pct": 0.800},
        "B": {"wins": 7, "losses": 3, "ties": 0, "gf": 28, "ga": 18, "games": 10, "win_pct": 0.700},
        "C": {"wins": 6, "losses": 4, "ties": 0, "gf": 25, "ga": 22, "games": 10, "win_pct": 0.600},
    }


def _sample_h2h():
    return {
        ("A", "B"): {"wins": 1, "games": 1},
        ("B", "A"): {"wins": 0, "games": 1},
        ("A", "C"): {"wins": 1, "games": 1},
        ("C", "A"): {"wins": 0, "games": 1},
        ("B", "C"): {"wins": 1, "games": 1},
        ("C", "B"): {"wins": 0, "games": 1},
    }


def _sample_team_imputation():
    return {
        "A": {"imputed": 0, "games": 10},
        "B": {"imputed": 0, "games": 10},
        "C": {"imputed": 0, "games": 10},
    }


def test_ensemble_exports_and_includes_bcar_weight_priority():
    teams = ["A", "B", "C"]
    orders = {
        "Win": ["A", "B", "C"],
        "Pyth": ["A", "B", "C"],
        "AdjPyth": ["A", "B", "C"],
        "BCAR": ["A", "C", "B"],
        "Elo": ["A", "C", "B"],
    }

    df = ui.build_calibrated_ensemble(
        teams,
        orders,
        _sample_stats(),
        _sample_h2h(),
        {"A": 0.65, "B": 0.55, "C": 0.45},
        _sample_team_imputation(),
    )

    assert "Weight BCAR" in df.columns
    assert "Norm Weight BCAR" in df.columns

    row = df.set_index("Team").loc["A"]
    assert row["Norm Weight Elo"] > row["Norm Weight BCAR"] > row["Norm Weight AdjPyth"] > row["Norm Weight Pyth"] > row["Norm Weight Win"]


def test_content_defaults_to_ensemble_for_rank_tables():
    section_defaults = {"Team Profile": "Profile", "Rank Tables": "Ensemble (Primary)", "Sectionals": "Sectionals"}
    available_sections = {
        "Team Profile": ["Profile"],
        "Rank Tables": ["Ensemble (Primary)", "Win%", "Pythag", "AdjPyth", "Elo", "BCAR"],
        "Sectionals": ["Sectionals"],
    }

    assert section_defaults["Rank Tables"] == "Ensemble (Primary)"
    assert available_sections["Rank Tables"][0] == "Ensemble (Primary)"


def test_sort_modes_include_sos_and_switching_preserves_deterministic_ties():
    sort_modes = ["Ensemble rank", "Elo", "BCAR", "AdjPyth", "Pyth", "Win%", "SOS"]
    assert "SOS" in sort_modes

    teams = ["A", "B", "C"]
    stats = {
        "A": {"win_pct": 0.50},
        "B": {"win_pct": 0.50},
        "C": {"win_pct": 0.40},
    }
    sos = {"A": 0.60, "B": 0.60, "C": 0.30}
    ensemble_df = pd.DataFrame(
        {
            "Team": ["A", "B", "C"],
            "Rank": [1, 2, 3],
            "Calibrated Score": [0.9, 0.8, 0.7],
        }
    )
    elo = {"A": 1500, "B": 1500, "C": 1400}
    bcar_table = pd.DataFrame({"Team": ["A", "B", "C"], "BCAR Score": [1.0, 1.0, 0.8]})
    adj_vals = {"A": 0.70, "B": 0.70, "C": 0.60}
    pyth_vals = {"A": 0.70, "B": 0.70, "C": 0.60}

    tied_pair_orders = {}
    for mode in sort_modes:
        ordered = ui.sort_teams_by_mode(mode, teams, stats, sos, ensemble_df, elo=elo, bcar_table=bcar_table, adj_vals=adj_vals, pyth_vals=pyth_vals)
        tied_pair_orders[mode] = tuple(t for t in ordered if t in {"A", "B"})

    assert all(pair == ("A", "B") for pair in tied_pair_orders.values())


def test_backward_compatible_weights_when_bcar_missing_from_config():
    weights = ui.sanitize_ensemble_weights({"Elo": 0.4, "AdjPyth": 0.2, "Pyth": 0.2, "Win": 0.2})

    assert "BCAR" in weights
    assert weights["BCAR"] > 0
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_hybrid_symbols_importable_from_domain_ranking():
    from domain.hybrid_ranking import HybridRankingConfig, ScheduleAdjustedGoalStrengthRanker
    from domain.ranking import HybridRankingConfig as ExportedHybridRankingConfig
    from domain.ranking import ScheduleAdjustedGoalStrengthRanker as ExportedScheduleAdjustedGoalStrengthRanker

    assert ExportedHybridRankingConfig is HybridRankingConfig
    assert ExportedScheduleAdjustedGoalStrengthRanker is ScheduleAdjustedGoalStrengthRanker
