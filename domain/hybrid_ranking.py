from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional

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
        self._matchup_model = None
        self.fit_warnings_: list[str] = []

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
        self.weights_ = weights

        reg = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        np.fill_diagonal(reg[:n_teams, :n_teams], self.config.ridge_lambda)

        prior_rhs = np.zeros(X.shape[1], dtype=float)
        prior_rhs[:n_teams] = self.config.ridge_lambda * priors

        lhs = Xw.T @ Xw + reg
        rhs = Xw.T @ yw + prior_rhs
        solve_used_pinv = False
        try:
            theta = np.linalg.solve(lhs, rhs)
            lhs_inv = np.linalg.inv(lhs)
        except np.linalg.LinAlgError:
            solve_used_pinv = True
            lhs_inv = np.linalg.pinv(lhs)
            theta = lhs_inv @ rhs

        expectation = X @ theta
        residuals = y - expectation

        sigma2, dof = self._estimate_sigma2(Xw=Xw, residuals=residuals, lhs_inv=lhs_inv)
        covariance = sigma2 * lhs_inv
        rating_se = np.sqrt(np.clip(np.diag(covariance)[:n_teams], a_min=0.0, a_max=None))

        se_prior = self._se_prior()
        confidence = np.clip(100.0 * (1.0 - (rating_se / se_prior)), 0.0, 100.0)

        sos = self._compute_sos(games, theta[:n_teams], team_to_idx, weights)
        lambda_sos = self.config.lambda_sos_max * (games_played / (games_played + self.config.k0))
        sov = self._compute_sov(games, residuals, theta[:n_teams], team_to_idx)
        volatility = self._compute_volatility(games, residuals, team_to_idx)

        rating = theta[:n_teams] + lambda_sos * sos + self.config.lambda_sov * sov - self.config.lambda_var * volatility
        rating_ci_low = rating - 1.96 * rating_se
        rating_ci_high = rating + 1.96 * rating_se

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
        self.rating_se_ = rating_se
        self.rating_ci_low_ = rating_ci_low
        self.rating_ci_high_ = rating_ci_high
        self.confidence_ = confidence
        self.sigma2_ = sigma2
        self.dof_ = dof
        self.covariance_ = covariance
        self.normal_matrix_ = lhs
        self.normal_rhs_ = rhs
        self.solve_used_pinv_ = solve_used_pinv
        self.fit_warnings_ = []
        if solve_used_pinv:
            self.fit_warnings_.append("normal-equation solve used pseudo-inverse fallback")
        if np.any(self.games_played_ < 2):
            self.fit_warnings_.append("very small sample size for one or more teams")

        # lazy import to avoid circular import at module load time
        from domain.matchup_model import PoissonAttackDefenseMatchupModel

        self._matchup_model = PoissonAttackDefenseMatchupModel(ranking_model=self).fit(games)
        self._is_fit = True
        return self

    def predict_matchup(
        self,
        team_a: str,
        team_b: str,
        venue: Literal["home", "away", "neutral"] = "neutral",
        max_goals: int = 10,
    ) -> dict[str, object]:
        if not self._is_fit or self._matchup_model is None:
            raise RuntimeError("Model must be fit before calling predict_matchup().")
        if venue not in {"home", "away", "neutral"}:
            raise ValueError("venue must be one of: home, away, neutral")

        prediction = self._matchup_model.predict_matchup(
            team_a=team_a,
            team_b=team_b,
            venue=venue,
            max_goals=max_goals,
        )

        ordered_scorelines = sorted(
            prediction["top_scorelines"],
            key=lambda x: (-x["probability"], x["score_a"], x["score_b"]),
        )

        return {
            "expected_goals_a": prediction["expected_goals_a"],
            "expected_goals_b": prediction["expected_goals_b"],
            "p_win": prediction["p_win"],
            "p_draw": prediction["p_draw"],
            "p_loss": prediction["p_loss"],
            "top_scorelines": ordered_scorelines,
            "goal_diff_interval_50": prediction["goal_diff_interval_50"],
            "goal_diff_interval_80": prediction["goal_diff_interval_80"],
            "total_goals_interval_50": prediction["total_goals_interval_50"],
            "total_goals_interval_80": prediction["total_goals_interval_80"],
            "matchup_confidence": prediction["matchup_confidence"],
        }

    def rankings(self) -> pd.DataFrame:
        return self.rankings_table()

    def rankings_table(self) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("Model must be fit before calling rankings_table().")

        frame = pd.DataFrame(
            {
                "team": self.teams_,
                "rating": self.rating_,
                "rating_se": self.rating_se_,
                "rating_ci_low": self.rating_ci_low_,
                "rating_ci_high": self.rating_ci_high_,
                "confidence": self.confidence_,
                "theta": self.theta_[: len(self.teams_)],
                "prior": self.priors_,
                "games": self.games_played_,
                "sos": self.sos_,
                "sov": self.sov_,
                "volatility": self.volatility_,
            }
        )
        return frame.sort_values("rating", ascending=False).reset_index(drop=True)

    def _estimate_sigma2(self, Xw: np.ndarray, residuals: np.ndarray, lhs_inv: np.ndarray) -> tuple[float, float]:
        weighted_sse = float(np.sum((np.sqrt(self.weights_) * residuals) ** 2))
        leverage = np.trace(Xw @ lhs_inv @ Xw.T)
        dof = max(float(np.sum(self.weights_) - leverage), 1.0)
        sigma2 = max(weighted_sse / dof, self.config.eps)
        return sigma2, dof

    def _se_prior(self) -> float:
        if self.config.ridge_lambda <= 0:
            return 1.0
        return float(np.sqrt(1.0 / self.config.ridge_lambda))

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
