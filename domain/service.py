from __future__ import annotations

from typing import Any


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
                "impact": quality + max(0, team_score - opp_score) * 0.05,
                "negative_impact": quality + max(0, opp_score - team_score) * 0.1,
            }
        )

    wins = [g for g in team_games if g["margin"] > 0]
    losses = [g for g in team_games if g["margin"] < 0]

    top_wins = sorted(
        wins,
        key=lambda g: (-g["quality"], -g["margin"], g["opponent"], g["team_score"], g["opp_score"]),
    )[: max(3, min(len(wins), 5))]
    worst_losses = sorted(
        losses,
        key=lambda g: (-g["negative_impact"], -g["margin"], g["opponent"], g["team_score"], g["opp_score"]),
    )[: max(3, min(len(losses), 5))]

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
        "top_wins": top_wins,
        "worst_losses": worst_losses,
        "summary": summary,
    }
