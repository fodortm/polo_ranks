from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from domain.hybrid_ranking import HybridRankingConfig
from domain.matchup_model import MatchupModelConfig

DEFAULT_PYTHAG_EXP = 2
DEFAULT_LOGISTIC_K = 10
DEFAULT_LOGISTIC_X0 = 0.5
DEFAULT_IMPUTED_MODE = "full"
DEFAULT_IMPUTED_WEIGHT = 1.0

HYBRID_DEFAULTS = {
    "k0": 0.55,
    "epsilon": 1e-6,
    "margin_scale_s": 4.0,
    "ridge_lambda": 0.08,
    "lambda_sos_max": 0.25,
    "lambda_sov": 0.35,
    "lambda_var": 0.15,
    "home_advantage": 0.1,
    "poisson_regularization": 0.2,
}

_HYBRID_ALIASES = {
    "epsilon": "eps",
    "margin_scale_s": "margin_scale",
}
_MATCHUP_ALIASES = {
    "poisson_regularization": "ridge_lambda",
}


def _coerce_numeric(value: Any, key: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field '{key}' must be numeric; got {value!r}.") from exc
    if not (numeric == numeric and numeric not in (float('inf'), float('-inf'))):
        raise ValueError(f"Config field '{key}' must be a finite number; got {value!r}.")
    return numeric


def resolve_hybrid_and_matchup_configs(params: Mapping[str, Any] | None) -> tuple[HybridRankingConfig, MatchupModelConfig]:
    """Map persisted config keys (including legacy aliases) into model configs."""
    source = dict(HYBRID_DEFAULTS)
    if params:
        source.update(dict(params))

    hybrid_keys = {field.name for field in fields(HybridRankingConfig)}
    matchup_keys = {field.name for field in fields(MatchupModelConfig)}
    allowed = set(HYBRID_DEFAULTS)
    allowed.update(hybrid_keys)
    allowed.update(matchup_keys)
    allowed.update(_HYBRID_ALIASES)
    allowed.update(_MATCHUP_ALIASES)
    unknown = sorted(set(params or {}).difference(allowed))
    if unknown:
        raise ValueError(
            "Unknown config field(s): " + ", ".join(unknown) + ". "
            "Supported fields include hybrid keys plus aliases: "
            "epsilon->eps, margin_scale_s->margin_scale, poisson_regularization->ridge_lambda."
        )

    hybrid_values: dict[str, Any] = {}
    for key, value in source.items():
        target = _HYBRID_ALIASES.get(key, key)
        if target in hybrid_keys:
            hybrid_values[target] = _coerce_numeric(value, key) if target != "use_home_indicator" else bool(value)

    home_advantage = _coerce_numeric(source.get("home_advantage", HYBRID_DEFAULTS["home_advantage"]), "home_advantage")

    matchup_values: dict[str, Any] = {}
    for key, value in source.items():
        target = _MATCHUP_ALIASES.get(key, key)
        if target in matchup_keys:
            matchup_values[target] = _coerce_numeric(value, key)

    # Persisted home_advantage is treated as a numeric prior strength for the Poisson home term.
    matchup_values.setdefault("home_advantage_ridge", home_advantage)

    return HybridRankingConfig(**hybrid_values), MatchupModelConfig(**matchup_values)
