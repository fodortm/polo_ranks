from __future__ import annotations

from typing import Any
from datetime import datetime, timezone


def _quality_score(team: str, bcar_scores: dict[str, float], stats: dict[str, dict[str, Any]], sos: dict[str, float]) -> float:
    if team in bcar_scores:
        return float(bcar_scores[team])
    win_pct = float(stats.get(team, {}).get("win_pct", 0.0))
    sos_val = float(sos.get(team, 0.0))
    return win_pct * 100.0 + sos_val * 10.0


def _result_streak(games: list[dict[str, Any]]) -> str:
    if not games:
        return "N/A"
    symbols = []
    for g in games:
        margin = g["margin"]
        symbols.append("W" if margin > 0 else "L" if margin < 0 else "T")
    last = symbols[-1]
    cnt = 1
    for s in reversed(symbols[:-1]):
        if s != last:
            break
        cnt += 1
    return f"{last}{cnt}"


def _location_tag(game_row: Any, is_home_slot: bool) -> str:
    if hasattr(game_row, "site"):
        site_val = str(getattr(game_row, "site") or "").strip().lower()
        if site_val in {"neutral", "n"}:
            return "neutral"
    return "home" if is_home_slot else "away"


def _match_context_tag(game_row: Any) -> str:
    if hasattr(game_row, "date") and getattr(game_row, "date") is not None:
        return str(getattr(game_row, "date"))
    if hasattr(game_row, "week") and getattr(game_row, "week") is not None:
        return f"W{getattr(game_row, 'week')}"
    return "date-unknown"


def build_team_resume(team, games_df, bcar_scores, stats, h2h, sos):
    team_games: list[dict[str, Any]] = []
    for row in games_df.itertuples(index=False):
        if row.team1 != team and row.team2 != team:
            continue
        is_home_slot = row.team1 == team
        opp = row.team2 if is_home_slot else row.team1
        team_score = int(row.score1 if is_home_slot else row.score2)
        opp_score = int(row.score2 if is_home_slot else row.score1)
        margin = team_score - opp_score
        quality = _quality_score(opp, bcar_scores, stats, sos)
        team_games.append(
            {
                "opponent": opp,
                "team_score": team_score,
                "opp_score": opp_score,
                "margin": margin,
                "quality": quality,
                "location": _location_tag(row, is_home_slot),
                "match_date": _match_context_tag(row),
                "opp_rank_at_match": (
                    int(getattr(row, "opp_rank"))
                    if hasattr(row, "opp_rank") and getattr(row, "opp_rank") is not None and str(getattr(row, "opp_rank")).lower() != "nan"
                    else None
                ),
            }
        )

    wins = [g for g in team_games if g["margin"] > 0]
    losses = [g for g in team_games if g["margin"] < 0]

    top_wins = sorted(
        wins,
        key=lambda g: (-g["quality"], -g["margin"], g["opp_score"], g["opponent"], g["match_date"]),
    )[:5]
    worst_losses = sorted(
        losses,
        key=lambda g: (g["quality"], g["margin"], -g["team_score"], g["opponent"], g["match_date"]),
    )[:5]

    recent_form = sorted(team_games, key=lambda g: g["match_date"], reverse=True)[:5]
    sos_context = sorted(team_games, key=lambda g: (-g["quality"], g["opponent"], g["match_date"]))[:5]

    tstats = stats.get(team, {})
    opponents = sorted({g["opponent"] for g in team_games if g["opponent"] in bcar_scores}, key=lambda t: (-bcar_scores[t], t))
    ranked_w = ranked_l = ranked_t = 0
    for opp in opponents:
        rec = h2h.get((team, opp), {"wins": 0, "games": 0})
        ranked_w += int(rec.get("wins", 0))
        ranked_g = int(rec.get("games", 0))
        ranked_l += max(ranked_g - int(rec.get("wins", 0)), 0)
    ranked_t = max(len([g for g in team_games if g["opponent"] in set(opponents)]) - ranked_w - ranked_l, 0)

    summary = {
        "record": f"{int(tstats.get('wins', 0))}-{int(tstats.get('losses', 0))}-{int(tstats.get('ties', 0))}",
        "goal_diff": int(tstats.get("gd", 0)),
        "notable_streak": _result_streak(team_games),
        "ranked_opponent_split": f"{ranked_w}-{ranked_l}-{ranked_t}",
    }

    return {
        "team": team,
        "sections": ["Top wins", "Worst losses", "Recent form", "Strength of schedule"],
        "top_wins": top_wins,
        "worst_losses": worst_losses,
        "recent_form": recent_form,
        "strength_of_schedule": sos_context,
        "summary": summary,
        "empty_states": {
            "top_wins": "No wins available yet." if not top_wins else "",
            "worst_losses": "No losses available yet." if not worst_losses else "",
            "recent_form": "No matches logged yet." if not recent_form else "",
            "strength_of_schedule": "No opponent-quality data yet." if not sos_context else "",
        },
    }


def _clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _trend_bucket(recent_form_score: float) -> str:
    if recent_form_score >= 60:
        return "rising"
    if recent_form_score <= 40:
        return "falling"
    return "flat"


def _build_standing_narrative(team: str, bcar_rank: int | None, bcar_total: int | None, trend_label: str, resume_signal: str, as_of_utc: str) -> str:
    if bcar_rank is not None and bcar_total is not None and bcar_total > 0:
        position_sentence = f"{team} is currently #{bcar_rank} in BCAR out of {bcar_total} teams."
    else:
        position_sentence = f"{team} has a BCAR profile in progress with position still stabilizing."

    trend_sentence = {
        "rising": "Recent BCAR direction is rising based on current form and game outcomes.",
        "flat": "Recent BCAR direction is flat, with no clear week-to-week shift.",
        "falling": "Recent BCAR direction is falling based on recent outcomes.",
    }[trend_label]

    resume_sentence = f"Resume signal is {resume_signal}. Data refreshed: {as_of_utc} UTC."
    return " ".join([position_sentence, trend_sentence, resume_sentence])


def build_team_explainer_card(team: str, stats: dict[str, dict[str, Any]], sos: dict[str, float], h2h: dict[tuple[str, str], dict[str, int]], team_imputation: dict[str, dict[str, int]], as_of: datetime | None = None) -> dict[str, Any]:
    team_stats = stats.get(team, {})
    games = int(team_stats.get("games", 0))
    wins = int(team_stats.get("wins", 0))
    losses = int(team_stats.get("losses", 0))
    ties = int(team_stats.get("ties", 0))
    opponents = [opp for opp in stats if opp != team]
    opp_games = [h2h.get((team, opp), {"games": 0}).get("games", 0) for opp in opponents]
    unique_played = sum(1 for g in opp_games if g > 0)
    max_unique = max(len(opponents), 1)
    imp_games = int(team_imputation.get(team, {}).get("imputed", 0))
    total_imp_games = int(team_imputation.get(team, {}).get("games", games))
    imputation_rate = (imp_games / total_imp_games) if total_imp_games else 0.0
    results_quality = _clamp_0_100(100.0 * float(team_stats.get("win_pct", 0.0)))
    schedule_strength = _clamp_0_100(100.0 * float(sos.get(team, 0.0)))
    consistency = _clamp_0_100(100.0 * (1.0 - min(1.0, abs(float(team_stats.get("gf", 0)) - float(team_stats.get("ga", 0))) / max(games * 6.0, 1.0))))
    recent_window = min(5, games)
    recent_form_score = _clamp_0_100(50.0 + (wins - losses) * (50.0 / max(recent_window, 1))) if games else 50.0
    data_coverage = _clamp_0_100(100.0 * ((0.6 * (unique_played / max_unique)) + (0.4 * (1.0 - imputation_rate))))
    confidence = "High"
    if games < 4 or unique_played < 3 or imputation_rate > 0.35:
        confidence = "Low"
    elif games < 7 or unique_played < 5 or imputation_rate > 0.20:
        confidence = "Medium"
    freshness = "Fresh"
    if games <= 2:
        freshness = "Early sample"
    elif games <= 5:
        freshness = "Building sample"
    def statement(name: str, value: float) -> str:
        if name == "Results quality":
            return "Strong results against played opponents." if value >= 65 else "Mixed results; rank may swing with next games."
        if name == "Schedule strength":
            return "Faced a challenging schedule." if value >= 60 else "Schedule has been softer so far."
        if name == "Consistency":
            return "Game-to-game performance has been steady." if value >= 60 else "Performance has been volatile week to week."
        if name == "Recent form":
            trend = _trend_bucket(value)
            if trend == "rising":
                return "Recent form is rising."
            if trend == "falling":
                return "Recent form is falling."
            return "Recent form is flat."
        return "Coverage is broad enough to trust comparisons." if value >= 60 else "Coverage is thin; treat rank as provisional."
    factors = [
        {"name": "Results quality", "value": round(results_quality, 1)},
        {"name": "Schedule strength", "value": round(schedule_strength, 1)},
        {"name": "Consistency", "value": round(consistency, 1)},
        {"name": "Recent form", "value": round(recent_form_score, 1)},
        {"name": "Data coverage", "value": round(data_coverage, 1)},
    ]
    for factor in factors:
        factor["summary"] = statement(factor["name"], factor["value"])
    as_of_ts = as_of or datetime.now(timezone.utc)
    trend_label = _trend_bucket(recent_form_score)
    bcar_rank = team_stats.get("bcar_rank")
    bcar_total = team_stats.get("bcar_total")
    resume_signal = "strong" if (results_quality >= 65 and schedule_strength >= 60) else ("developing" if results_quality >= 50 else "mixed")
    as_of_utc = as_of_ts.isoformat()
    narrative = _build_standing_narrative(team, bcar_rank, bcar_total, trend_label, resume_signal, as_of_utc)
    return {
        "team": team,
        "factors": factors,
        "confidence_label": confidence,
        "recency_label": freshness,
        "as_of_utc": as_of_utc,
        "record": f"{wins}-{losses}-{ties}",
        "standing_narrative": narrative,
    }
