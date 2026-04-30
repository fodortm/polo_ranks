from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HybridRankingConfig:
    ridge_lambda: float = 4.0
    k0: float = 8.0
    eps: float = 1e-6
    margin_scale: float = 1.0
    lambda_sos_max: float = 0.25
    lambda_sov: float = 0.2
    lambda_var: float = 0.1
    use_home_indicator: bool = True


class ScheduleAdjustedGoalStrengthRanker:
    """Opponent-adjusted team ranking model using a prior-centered ridge fit."""

    REQUIRED_COLUMNS = {"team_a", "team_b", "goals_a", "goals_b"}
    OPTIONAL_COLUMNS = {"date", "venue"}

    def __init__(self, config: Optional[HybridRankingConfig] = None) -> None:
        self.config = config or HybridRankingConfig()
        self._is_fit = False

    def fit(self, games_df: pd.DataFrame, game_weights: Optional[Iterable[float]] = None) -> "ScheduleAdjustedGoalStrengthRanker":
        self._validate_schema(games_df)
        games = games_df.copy()

        teams = sorted(set(games["team_a"]).union(games["team_b"]))
        team_to_idx = {team: idx for idx, team in enumerate(teams)}
        n_games = len(games)
        n_teams = len(teams)

        weights = self._resolve_weights(n_games=n_games, game_weights=game_weights)

        games_played, gf, ga = self._team_goal_totals(games, teams)
        priors = (games_played / (games_played + self.config.k0)) * np.log((gf + self.config.eps) / (ga + self.config.eps))

        X = np.zeros((n_games, n_teams + (1 if self.config.use_home_indicator else 0)), dtype=float)
        y = np.zeros(n_games, dtype=float)

        for row_idx, row in games.reset_index(drop=True).iterrows():
            i = team_to_idx[row["team_a"]]
            j = team_to_idx[row["team_b"]]
            X[row_idx, i] = 1.0
            X[row_idx, j] = -1.0

            gd = float(row["goals_a"] - row["goals_b"])
            y[row_idx] = np.sign(gd) * np.log1p(abs(gd) / self.config.margin_scale)

            if self.config.use_home_indicator:
                X[row_idx, -1] = self._home_indicator(row)

        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        yw = y * sqrt_w

        reg = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        np.fill_diagonal(reg[:n_teams, :n_teams], self.config.ridge_lambda)

        prior_rhs = np.zeros(X.shape[1], dtype=float)
        prior_rhs[:n_teams] = self.config.ridge_lambda * priors

        lhs = Xw.T @ Xw + reg
        rhs = Xw.T @ yw + prior_rhs
        theta = np.linalg.solve(lhs, rhs)

        expectation = X @ theta
        residuals = y - expectation

        sos = self._compute_sos(games, theta[:n_teams], team_to_idx, weights)
        lambda_sos = self.config.lambda_sos_max * (games_played / (games_played + self.config.k0))
        sov = self._compute_sov(games, residuals, theta[:n_teams], team_to_idx)
        volatility = self._compute_volatility(games, residuals, team_to_idx)

        rating = theta[:n_teams] + lambda_sos * sos + self.config.lambda_sov * sov - self.config.lambda_var * volatility

        self.teams_ = teams
        self.team_to_idx_ = team_to_idx
        self.idx_to_team_ = {v: k for k, v in team_to_idx.items()}
        self.theta_ = theta
        self.priors_ = priors
        self.games_played_ = games_played
        self.gf_ = gf
        self.ga_ = ga
        self.weights_ = weights
        self.design_matrix_ = X
        self.response_ = y
        self.residuals_ = residuals
        self.sos_ = sos
        self.lambda_sos_ = lambda_sos
        self.sov_ = sov
        self.volatility_ = volatility
        self.rating_ = rating
        self.normal_matrix_ = lhs
        self.normal_rhs_ = rhs

        self._is_fit = True
        return self

    def rankings(self) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("Model must be fit before calling rankings().")

        frame = pd.DataFrame(
            {
                "team": self.teams_,
                "rating": self.rating_,
                "theta": self.theta_[: len(self.teams_)],
                "prior": self.priors_,
                "games": self.games_played_,
                "sos": self.sos_,
                "sov": self.sov_,
                "volatility": self.volatility_,
            }
        )
        return frame.sort_values("rating", ascending=False).reset_index(drop=True)

    def _validate_schema(self, games_df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(games_df.columns)
        if missing:
            raise ValueError(f"games_df missing required columns: {sorted(missing)}")

    def _resolve_weights(self, n_games: int, game_weights: Optional[Iterable[float]]) -> np.ndarray:
        if game_weights is None:
            return np.ones(n_games, dtype=float)
        weights = np.asarray(list(game_weights), dtype=float)
        if len(weights) != n_games:
            raise ValueError("game_weights must match number of rows in games_df")
        if np.any(weights <= 0):
            raise ValueError("game_weights values must be > 0")
        return weights

    def _team_goal_totals(self, games: pd.DataFrame, teams: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_teams = len(teams)
        team_to_idx = {team: idx for idx, team in enumerate(teams)}
        games_played = np.zeros(n_teams, dtype=float)
        gf = np.zeros(n_teams, dtype=float)
        ga = np.zeros(n_teams, dtype=float)

        for _, row in games.iterrows():
            a = team_to_idx[row["team_a"]]
            b = team_to_idx[row["team_b"]]
            ga_a, ga_b = float(row["goals_a"]), float(row["goals_b"])

            games_played[a] += 1
            games_played[b] += 1
            gf[a] += ga_a
            ga[a] += ga_b
            gf[b] += ga_b
            ga[b] += ga_a

        return games_played, gf, ga

    def _compute_sos(self, games: pd.DataFrame, theta: np.ndarray, team_to_idx: Dict[str, int], weights: np.ndarray) -> np.ndarray:
        n_teams = len(team_to_idx)
        weighted_opp_sum = np.zeros(n_teams, dtype=float)
        weighted_games = np.zeros(n_teams, dtype=float)

        for g_idx, row in games.reset_index(drop=True).iterrows():
            w = float(weights[g_idx])
            a = team_to_idx[row["team_a"]]
            b = team_to_idx[row["team_b"]]
            weighted_opp_sum[a] += w * theta[b]
            weighted_opp_sum[b] += w * theta[a]
            weighted_games[a] += w
            weighted_games[b] += w

        return np.divide(weighted_opp_sum, weighted_games, out=np.zeros_like(weighted_opp_sum), where=weighted_games > 0)

    def _compute_sov(self, games: pd.DataFrame, residuals: np.ndarray, theta: np.ndarray, team_to_idx: Dict[str, int]) -> np.ndarray:
        n_teams = len(team_to_idx)
        sov_sum = np.zeros(n_teams, dtype=float)
        win_count = np.zeros(n_teams, dtype=float)

        for g_idx, row in games.reset_index(drop=True).iterrows():
            a = team_to_idx[row["team_a"]]
            b = team_to_idx[row["team_b"]]
            quality_a = max(theta[b], 0.0)
            quality_b = max(theta[a], 0.0)
            res = residuals[g_idx]

            if row["goals_a"] > row["goals_b"] and res > 0:
                sov_sum[a] += res * (1.0 + quality_a)
                win_count[a] += 1
            elif row["goals_b"] > row["goals_a"] and -res > 0:
                sov_sum[b] += (-res) * (1.0 + quality_b)
                win_count[b] += 1

        return np.divide(sov_sum, win_count, out=np.zeros_like(sov_sum), where=win_count > 0)

    def _compute_volatility(self, games: pd.DataFrame, residuals: np.ndarray, team_to_idx: Dict[str, int]) -> np.ndarray:
        n_teams = len(team_to_idx)
        bucket = [[] for _ in range(n_teams)]
        for g_idx, row in games.reset_index(drop=True).iterrows():
            a = team_to_idx[row["team_a"]]
            b = team_to_idx[row["team_b"]]
            r = float(residuals[g_idx])
            bucket[a].append(r)
            bucket[b].append(r)

        volatility = np.zeros(n_teams, dtype=float)
        for i, values in enumerate(bucket):
            if values:
                volatility[i] = float(np.std(values))
        return volatility

    @staticmethod
    def _home_indicator(row: pd.Series) -> float:
        venue = row.get("venue")
        if venue is None or pd.isna(venue):
            return 0.0
        venue_norm = str(venue).strip().lower()
        if venue_norm in {"home", "a_home", "team_a_home"}:
            return 1.0
        if venue_norm in {"away", "b_home", "team_b_home"}:
            return -1.0
        return 0.0
