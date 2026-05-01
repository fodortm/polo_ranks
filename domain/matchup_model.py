from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from domain.hybrid_ranking import ScheduleAdjustedGoalStrengthRanker


@dataclass(frozen=True)
class MatchupModelConfig:
    ridge_lambda: float = 1.0
    home_advantage_ridge: float = 0.5
    max_iter: int = 200
    tol: float = 1e-6


class PoissonAttackDefenseMatchupModel:
    """Poisson attack/defense model for matchup forecasts.

    Venue is interpreted from team_a perspective:
    - "home": team_a receives home advantage.
    - "away": team_a plays away (team_b receives home advantage).
    - "neutral": no home advantage applied.
    """

    REQUIRED_COLUMNS = {"team_a", "team_b", "goals_a", "goals_b"}

    def __init__(
        self,
        config: Optional[MatchupModelConfig] = None,
        ranking_model: Optional[ScheduleAdjustedGoalStrengthRanker] = None,
    ) -> None:
        self.config = config or MatchupModelConfig()
        self.ranking_model = ranking_model
        self._is_fit = False

    def fit(self, games_df: pd.DataFrame) -> "PoissonAttackDefenseMatchupModel":
        self._validate_schema(games_df)
        games = games_df.reset_index(drop=True).copy()

        teams = sorted(set(games["team_a"]).union(games["team_b"]))
        n_teams = len(teams)
        team_to_idx = {team: idx for idx, team in enumerate(teams)}

        mu = np.log(max(float(games[["goals_a", "goals_b"]].to_numpy().mean()), 1e-6))
        alpha = np.zeros(n_teams)
        delta = np.zeros(n_teams)
        h = 0.0

        for _ in range(self.config.max_iter):
            g_mu = 0.0
            g_alpha = np.zeros(n_teams)
            g_delta = np.zeros(n_teams)
            g_h = 0.0

            h_mu = 0.0
            h_alpha = np.zeros(n_teams)
            h_delta = np.zeros(n_teams)
            h_h = 0.0

            for _, row in games.iterrows():
                i = team_to_idx[row["team_a"]]
                j = team_to_idx[row["team_b"]]
                ya = float(row["goals_a"])
                yb = float(row["goals_b"])

                venue = str(row.get("venue", "neutral")).strip().lower()
                home_term = 1.0 if venue in {"home", "a_home", "team_a_home"} else -1.0 if venue in {"away", "b_home", "team_b_home"} else 0.0

                eta_a = mu + alpha[i] - delta[j] + h * home_term
                eta_b = mu + alpha[j] - delta[i]
                lam_a = exp(eta_a)
                lam_b = exp(eta_b)

                r_a = ya - lam_a
                r_b = yb - lam_b

                g_mu += r_a + r_b
                h_mu += lam_a + lam_b

                g_alpha[i] += r_a
                g_alpha[j] += r_b
                h_alpha[i] += lam_a
                h_alpha[j] += lam_b

                g_delta[j] -= r_a
                g_delta[i] -= r_b
                h_delta[j] += lam_a
                h_delta[i] += lam_b

                g_h += home_term * r_a
                h_h += (home_term * home_term) * lam_a

            g_alpha -= self.config.ridge_lambda * alpha
            g_delta -= self.config.ridge_lambda * delta
            g_h -= self.config.home_advantage_ridge * h

            h_alpha += self.config.ridge_lambda
            h_delta += self.config.ridge_lambda
            h_h += self.config.home_advantage_ridge

            new_mu = mu + g_mu / max(h_mu, 1e-9)
            new_alpha = alpha + g_alpha / np.maximum(h_alpha, 1e-9)
            new_delta = delta + g_delta / np.maximum(h_delta, 1e-9)
            new_h = h + g_h / max(h_h, 1e-9)

            # identifiability constraint: sum-to-zero anchoring
            new_alpha -= np.mean(new_alpha)
            new_delta -= np.mean(new_delta)

            step = max(
                abs(new_mu - mu),
                float(np.max(np.abs(new_alpha - alpha))),
                float(np.max(np.abs(new_delta - delta))),
                abs(new_h - h),
            )
            mu, alpha, delta, h = new_mu, new_alpha, new_delta, new_h
            if step < self.config.tol:
                break

        self.teams_ = teams
        self.team_to_idx_ = team_to_idx
        self.mu_ = float(mu)
        self.alpha_ = alpha
        self.delta_ = delta
        self.h_ = float(h)
        self._is_fit = True
        return self

    def predict_matchup(self, team_a: str, team_b: str, venue: Literal["home", "away", "neutral"] = "neutral", max_goals: int = 10) -> Dict[str, object]:
        if not self._is_fit:
            raise RuntimeError("Model must be fit before predict_matchup().")
        if team_a not in self.team_to_idx_ or team_b not in self.team_to_idx_:
            raise ValueError("Both team_a and team_b must be present in training data")
        if venue not in {"home", "away", "neutral"}:
            raise ValueError("venue must be one of: home, away, neutral")

        i = self.team_to_idx_[team_a]
        j = self.team_to_idx_[team_b]
        home_term = 1.0 if venue == "home" else -1.0 if venue == "away" else 0.0

        lam_a = exp(self.mu_ + self.alpha_[i] - self.delta_[j] + self.h_ * home_term)
        lam_b = exp(self.mu_ + self.alpha_[j] - self.delta_[i])

        p_a = self._poisson_probs(lam_a, max_goals)
        p_b = self._poisson_probs(lam_b, max_goals)
        joint = np.outer(p_a, p_b)

        p_draw = float(np.trace(joint))
        p_win = float(np.tril(joint, k=-1).sum())
        p_loss = float(np.triu(joint, k=1).sum())

        top_scorelines = []
        for ga in range(max_goals + 1):
            for gb in range(max_goals + 1):
                top_scorelines.append({"score_a": ga, "score_b": gb, "probability": float(joint[ga, gb])})
        top_scorelines = sorted(top_scorelines, key=lambda x: x["probability"], reverse=True)[:8]

        gd_grid, gd_pmf = self._goal_diff_pmf(joint)
        tg_grid, tg_pmf = self._total_goals_pmf(joint)

        ranking_conf = self._matchup_confidence(team_a, team_b)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "venue": venue,
            "expected_goals_a": float(lam_a),
            "expected_goals_b": float(lam_b),
            "p_win": p_win,
            "p_draw": p_draw,
            "p_loss": p_loss,
            "top_scorelines": top_scorelines,
            "goal_diff_interval_50": self._central_interval(gd_grid, gd_pmf, 0.50),
            "goal_diff_interval_80": self._central_interval(gd_grid, gd_pmf, 0.80),
            "total_goals_interval_50": self._central_interval(tg_grid, tg_pmf, 0.50),
            "total_goals_interval_80": self._central_interval(tg_grid, tg_pmf, 0.80),
            "matchup_confidence": ranking_conf,
        }

    @staticmethod
    def _poisson_probs(lam: float, max_goals: int) -> np.ndarray:
        probs = np.zeros(max_goals + 1, dtype=float)
        probs[0] = exp(-lam)
        for k in range(1, max_goals + 1):
            probs[k] = probs[k - 1] * lam / k
        tail = max(0.0, 1.0 - probs.sum())
        probs[-1] += tail
        return probs

    @staticmethod
    def _goal_diff_pmf(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_goals = joint.shape[0] - 1
        diffs = np.arange(-max_goals, max_goals + 1)
        pmf = np.zeros_like(diffs, dtype=float)
        for a in range(max_goals + 1):
            for b in range(max_goals + 1):
                pmf[a - b + max_goals] += joint[a, b]
        return diffs, pmf

    @staticmethod
    def _total_goals_pmf(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_goals = joint.shape[0] - 1
        totals = np.arange(0, 2 * max_goals + 1)
        pmf = np.zeros_like(totals, dtype=float)
        for a in range(max_goals + 1):
            for b in range(max_goals + 1):
                pmf[a + b] += joint[a, b]
        return totals, pmf

    @staticmethod
    def _central_interval(grid: np.ndarray, pmf: np.ndarray, mass: float) -> tuple[float, float]:
        lo_q = (1.0 - mass) / 2.0
        hi_q = 1.0 - lo_q
        cdf = np.cumsum(pmf)
        lo_idx = int(np.searchsorted(cdf, lo_q, side="left"))
        hi_idx = int(np.searchsorted(cdf, hi_q, side="left"))
        lo_idx = min(max(lo_idx, 0), len(grid) - 1)
        hi_idx = min(max(hi_idx, 0), len(grid) - 1)
        return float(grid[lo_idx]), float(grid[hi_idx])

    def _matchup_confidence(self, team_a: str, team_b: str) -> float:
        if self.ranking_model is None or not getattr(self.ranking_model, "_is_fit", False):
            return float("nan")
        conf_map = dict(zip(self.ranking_model.teams_, self.ranking_model.confidence_))
        if team_a not in conf_map or team_b not in conf_map:
            return float("nan")
        return float((conf_map[team_a] + conf_map[team_b]) / 2.0)

    def _validate_schema(self, games_df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(games_df.columns)
        if missing:
            raise ValueError(f"games_df missing required columns: {sorted(missing)}")
