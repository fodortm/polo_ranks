import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config.model_params import resolve_hybrid_and_matchup_configs


def test_resolve_aliases_and_home_advantage_mapping():
    hybrid_cfg, matchup_cfg = resolve_hybrid_and_matchup_configs(
        {
            "epsilon": 1e-4,
            "margin_scale_s": 2.5,
            "poisson_regularization": 0.75,
            "home_advantage": 0.33,
        }
    )

    assert hybrid_cfg.eps == pytest.approx(1e-4)
    assert hybrid_cfg.margin_scale == pytest.approx(2.5)
    assert matchup_cfg.ridge_lambda == pytest.approx(0.75)
    assert matchup_cfg.home_advantage_ridge == pytest.approx(0.33)


def test_resolve_defaults_when_params_missing():
    hybrid_cfg, matchup_cfg = resolve_hybrid_and_matchup_configs({})

    assert hybrid_cfg.eps == pytest.approx(1e-6)
    assert hybrid_cfg.margin_scale == pytest.approx(4.0)
    assert matchup_cfg.ridge_lambda == pytest.approx(0.2)
    assert matchup_cfg.home_advantage_ridge == pytest.approx(0.1)


def test_resolve_rejects_unknown_fields():
    with pytest.raises(ValueError, match=r"Unknown config field\(s\): made_up"):
        resolve_hybrid_and_matchup_configs({"made_up": 1})


def test_resolve_rejects_malformed_numeric_field():
    with pytest.raises(ValueError, match="must be numeric"):
        resolve_hybrid_and_matchup_configs({"epsilon": "not-a-number"})
