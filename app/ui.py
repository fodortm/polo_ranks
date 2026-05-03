import pandas as pd
import os
import re
import math
import json
import hashlib
from io import StringIO
from collections import defaultdict
from functools import cmp_to_key
import streamlit as st
import altair as alt

from domain.parsing import discover_score_files, is_skippable_line, load_parser_config, normalize_team_name, parse_game_line_anchored
from domain.hybrid_ranking import HybridRankingConfig, ScheduleAdjustedGoalStrengthRanker
from domain.service import build_team_resume, build_team_explainer_card

# ---------------- Constants ---------------- #
SCORES_CSV = "scores.csv"
CONFIG_JSON = "model_config.json"
SCORES_GLOB_SUFFIX = "_scores_illpolo.txt"
DATA_DIR = "."
SEMANTIC_COLORS = {
    "positive": "#2E8540",
    "negative": "#B50909",
    "neutral": "#6B7280",
}
TYPOGRAPHY_SCALE = {"title": "##", "subtitle": "####"}
SPACING_SCALE = {"section": "<div style='margin-top: 1.25rem;'></div>", "panel": "<div style='margin-top: 0.75rem;'></div>"}
RANK_TIER_COLORS = {"elite": "#1D4ED8", "contender": "#93C5FD", "support": "#9CA3AF"}
CHART_FORMATS = {"pct3": ".3f", "float3": ".3f", "float2": ".2f", "float1": ".1f", "int0": ".0f"}

DEFAULT_PRIMARY_METRIC = "BCAR"
DEFAULT_DASHBOARD_METRIC_LENS = "BCAR"
DEFAULT_WEEKLY_TREND_METRIC = "BCAR"
DEFAULT_SORT_MODE = "BCAR"
DEFAULT_HYBRID_UI_MODEL = "Legacy Hybrid"
LEGACY_METRIC_DEFAULTS = {"Ensemble", "Ensemble (Primary)", "Win%", "Win %"}


SECONDARY_METRIC_ORDER = ["Elo", "Adj Pyth", "Win %", "Ensemble Score"]
SECONDARY_METRIC_TOOLTIPS = {
    "Elo": "Elo rating-based strength estimate used as a secondary cross-check.",
    "Adj Pyth": "Adjusted Pythagorean expectation from scored results.",
    "Win %": "Raw win percentage; useful context but less schedule-aware.",
    "Ensemble Score": "Composite score that blends multiple secondary models.",
}

METRIC_LENS_OPTIONS = [
    "BCAR",
    "Elo",
    "Ensemble",
    "Win%",
    "Adj Pyth",
]
METRIC_OPTION_LABELS = {
    "BCAR": "BCAR (Recommended default)",
    "Elo": "ELO (Secondary / backup)",
    "Ensemble": "Ensemble Score (Secondary / backup)",
    "Win%": "Win% (Secondary / backup)",
    "Adj Pyth": "Adj Pyth (Secondary / backup)",
}
WEEKLY_TREND_OPTIONS = ["BCAR", "Elo", "Ensemble", "Win %", "Adjusted Pyth"]
WEEKLY_TREND_LABELS = {
    "BCAR": "BCAR (Recommended default)",
    "Elo": "ELO (Secondary / backup)",
    "Ensemble": "Ensemble Score (Secondary / backup)",
    "Win %": "Win % (Secondary / backup)",
    "Adjusted Pyth": "Adjusted Pyth (Secondary / backup)",
}


DEFAULT_ENSEMBLE_WEIGHTS = {
    "Elo": 0.40,
    "BCAR": 0.25,
    "AdjPyth": 0.18,
    "Pyth": 0.12,
    "Win": 0.05,
}



def _normalize_whole_season_flag(raw_value):
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    normalized = str(raw_value).strip().lower()
    return normalized in {"1", "true", "t", "yes", "y", "all", "season", "whole"}


def _init_dashboard_timeframe_state():
    whole_season_from_url = _normalize_whole_season_flag(st.query_params.get("whole_season"))
    if "dashboard_time_window" not in st.session_state:
        st.session_state["dashboard_time_window"] = "All" if whole_season_from_url else "Last 4 weeks"
    elif st.session_state["dashboard_time_window"] not in {"Last 4 weeks", "All"}:
        st.session_state["dashboard_time_window"] = "Last 4 weeks"
    if "dashboard_whole_season" not in st.session_state:
        st.session_state["dashboard_whole_season"] = st.session_state["dashboard_time_window"] == "All"


def _init_dashboard_metric_state():
    metric_from_url = str(st.query_params.get("metric", DEFAULT_DASHBOARD_METRIC_LENS)).strip()
    if metric_from_url not in METRIC_LENS_OPTIONS:
        metric_from_url = DEFAULT_DASHBOARD_METRIC_LENS
    if "dashboard_metric_lens" not in st.session_state:
        st.session_state["dashboard_metric_lens"] = metric_from_url
    elif st.session_state["dashboard_metric_lens"] not in METRIC_LENS_OPTIONS:
        st.session_state["dashboard_metric_lens"] = DEFAULT_DASHBOARD_METRIC_LENS


def _sync_dashboard_timeframe_query_params():
    dashboard_whole_season = st.session_state.get("dashboard_time_window") == "All"
    st.query_params["whole_season"] = "1" if dashboard_whole_season else "0"
    st.query_params["metric"] = st.session_state.get("dashboard_metric_lens", DEFAULT_DASHBOARD_METRIC_LENS)


def _on_dashboard_whole_season_toggle():
    dashboard_whole_season = st.session_state.get("dashboard_whole_season_toggle", False)
    st.session_state["dashboard_time_window"] = "All" if dashboard_whole_season else "Last 4 weeks"
    _track_on_change("filter_usage", "dashboard_time_window")
    _sync_dashboard_timeframe_query_params()


def _on_dashboard_time_window_change():
    _track_on_change("filter_usage", "dashboard_time_window")
    _sync_dashboard_timeframe_query_params()


def track_ui_event(event_name, **payload):
    events = st.session_state.setdefault("ui_analytics_events", [])
    events.append({"event": event_name, "payload": payload, "at": pd.Timestamp.utcnow().isoformat()})


def _track_on_change(event_name, state_key):
    track_ui_event(event_name, key=state_key, value=st.session_state.get(state_key))


def _is_admin_user():
    query_flag = str(st.query_params.get("is_admin", "")).strip().lower()
    if query_flag in {"1", "true", "t", "yes", "y"}:
        return True

    secret_flag = st.secrets.get("ui_admin_enabled", False)
    if isinstance(secret_flag, bool):
        return secret_flag
    return str(secret_flag).strip().lower() in {"1", "true", "t", "yes", "y"}

def sanitize_ensemble_weights(raw_weights):
    raw_weights = raw_weights or {}
    sanitized = {}
    for model, default_weight in DEFAULT_ENSEMBLE_WEIGHTS.items():
        raw_value = raw_weights.get(model, default_weight)
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            parsed = default_weight
        if math.isnan(parsed) or parsed < 0:
            parsed = 0.0
        sanitized[model] = parsed

    total = sum(sanitized.values())
    if total <= 0:
        return DEFAULT_ENSEMBLE_WEIGHTS.copy()
    return {model: weight / total for model, weight in sanitized.items()}

def render_typography(level, text):
    if level in TYPOGRAPHY_SCALE:
        st.markdown(f"{TYPOGRAPHY_SCALE[level]} {text}")
    elif level == "caption":
        st.caption(text)
    else:
        st.write(text)

def render_spacing(level="panel"):
    st.markdown(SPACING_SCALE.get(level, SPACING_SCALE["panel"]), unsafe_allow_html=True)

def render_primary_rank_caption_block():
    st.caption("Ensemble (Primary) resolves ties in this order: calibrated score → direct H2H tiebreak → SOS margin tiebreak → stable secondary key.")
    st.caption("SOS is used as a context/sorting signal and tiebreak ingredient; it does not independently set rank unless SOS is the selected sorting mode.")


def render_kpi_card(container, label, value, delta=None, caption=None):
    container.metric(label, value, delta=delta)
    if caption:
        container.caption(caption)



def render_team_explainer_card(card):
    st.caption(f"Ranking explainer · {card['team']} · As of {card['as_of_utc']} UTC")
    c1, c2 = st.columns(2)
    c1.metric("Confidence", card["confidence_label"])
    c2.metric("Recency", card["recency_label"])
    if card.get("standing_narrative"):
        st.info(card["standing_narrative"])
    explainer_df = pd.DataFrame(card["factors"]).rename(columns={"name": "Factor", "value": "Score", "summary": "What this means"})
    st.dataframe(explainer_df[["Factor", "Score", "What this means"]], use_container_width=True, hide_index=True)
def apply_chart_theme(chart):
    return chart.configure_axis(
        grid=True, tickColor="#D1D5DB", labelColor="#111827", titleColor="#111827"
    ).configure_legend(
        orient="bottom", titleColor="#111827", labelColor="#374151", symbolType="circle"
    ).configure_view(strokeWidth=0)

# ---------------- Parsing ---------------- #
def _parse_line(line):
    parsed = _parse_game_line_anchored(line)
    return [parsed] if parsed is not None else []

def _is_skippable_line(raw):
    return is_skippable_line(raw)

def _normalize_team_name(name):
    return normalize_team_name(name, load_parser_config())

def _discover_score_files(data_dir=DATA_DIR):
    return discover_score_files(data_dir)

def _parse_game_line_anchored(line):
    return parse_game_line_anchored(line, load_parser_config())

def _build_file_fingerprint(files):
    payload = []
    for path in files:
        stat = os.stat(path)
        payload.append(f"{os.path.basename(path)}:{int(stat.st_mtime_ns)}:{stat.st_size}")
    encoded = "|".join(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

@st.cache_data
def _load_games_pipeline_cached(data_dir, file_fingerprint):
    _ = file_fingerprint
    files = _discover_score_files(data_dir)
    lines_scanned = 0
    skipped = 0
    suspicious_unparsed = 0
    unresolved_suspicious_lines = []
    parsed_rows = []
    per_file_reports = []
    for path in files:
        file_lines_scanned = 0
        file_skipped = 0
        file_suspicious = 0
        file_games = 0
        file_suspicious_examples = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                lines_scanned += 1
                file_lines_scanned += 1
                parsed = _parse_game_line_anchored(line)
                if parsed is not None:
                    parsed_rows.append(parsed)
                    file_games += 1
                else:
                    raw = line.strip()
                    if _is_skippable_line(raw):
                        skipped += 1
                        file_skipped += 1
                    elif raw:
                        suspicious_unparsed += 1
                        file_suspicious += 1
                        unresolved_suspicious_lines.append(raw)
                        if len(file_suspicious_examples) < 5:
                            file_suspicious_examples.append(raw)
        per_file_reports.append(
            {
                "file_name": os.path.basename(path),
                "lines_scanned": file_lines_scanned,
                "games_parsed": file_games,
                "skipped": file_skipped,
                "suspicious_unparsed": file_suspicious,
                "suspicious_examples": file_suspicious_examples,
            }
        )
    games_df = pd.DataFrame(parsed_rows, columns=["team1", "score1", "team2", "score2"])
    games_before_dedup = len(games_df)
    if not games_df.empty:
        identity = games_df.copy()
        identity["team1"] = identity["team1"].str.strip().str.lower()
        identity["team2"] = identity["team2"].str.strip().str.lower()
        games_df = games_df.loc[~identity.duplicated(subset=["team1", "team2", "score1", "score2"])].reset_index(drop=True)
    qa_meta = {
        "rebuilt_at": pd.Timestamp.utcnow().isoformat(),
        "files_loaded": len(files),
        "lines_scanned": lines_scanned,
        "games_parsed": games_before_dedup,
        "skipped": skipped,
        "suspicious_unparsed": suspicious_unparsed,
        "unresolved_suspicious_lines": unresolved_suspicious_lines[:25],
        "duplicates_dropped": games_before_dedup - len(games_df),
        "per_file_reports": per_file_reports,
    }
    return games_df, qa_meta

def load_games_pipeline(data_dir=DATA_DIR):
    files = _discover_score_files(data_dir)
    file_fingerprint = _build_file_fingerprint(files)
    return _load_games_pipeline_cached(data_dir, file_fingerprint)

def clear_score_pipeline_cache():
    _load_games_pipeline_cached.clear()
    load_scores.clear()

def _week_label_from_path(path):
    name = os.path.basename(path).replace("_scores_illpolo.txt", "")
    return name.replace("_", " ").title()

def compute_weekly_rank_history(data_dir=DATA_DIR):
    files = _discover_score_files(data_dir)
    parsed_rows = []
    rank_rows = []
    for week_num, path in enumerate(files, start=1):
        week_label = _week_label_from_path(path)
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                parsed = _parse_game_line_anchored(line)
                if parsed is not None:
                    parsed_rows.append(parsed)
        week_games = pd.DataFrame(parsed_rows, columns=["team1", "score1", "team2", "score2"])
        if week_games.empty:
            continue
        scored_games = week_games.dropna(subset=["score1"])
        base_stats, _ = compute_stats(scored_games)
        all_week_teams = set(week_games["team1"]).union(set(week_games["team2"]))
        for team in all_week_teams:
            if team not in base_stats:
                base_stats[team] = {'wins':0,'losses':0,'ties':0,'gf':0,'ga':0,'games':0,'opponents':[]}
        inferred = infer_default_scores(week_games, base_stats)
        week_stats, week_h2h = compute_stats(inferred)
        week_order = rank_win_pct(week_stats, week_h2h)
        for rank, team in enumerate(week_order, start=1):
            rank_rows.append({"team": team, "rank": rank, "week_num": week_num, "week_label": week_label})
    return pd.DataFrame(rank_rows)



def build_dashboard_view_model(stats, rank_order, rank_metric_values, rank_metric_label, games_inferred, weekly_ranks, window_size=3, top_n_rank=10, trend_top_n=8, movement_top_n=8, primary_table=None, sos=None):
    """Build all dashboard-ready datasets from a single shared computation path."""
    stats = stats or {}
    rank_order = rank_order or []
    rank_metric_values = rank_metric_values or {}
    games_inferred = games_inferred if games_inferred is not None else pd.DataFrame()
    weekly_ranks = weekly_ranks if weekly_ranks is not None else pd.DataFrame()

    top_team = rank_order[0] if rank_order else None
    top_team_metric = rank_metric_values.get(top_team) if top_team else None
    total_games = int(len(games_inferred))
    scored_results = int(games_inferred["score1"].notna().sum()) if "score1" in games_inferred else 0
    inferred_results = int(games_inferred.get("is_imputed", pd.Series(False, index=games_inferred.index)).fillna(False).sum()) if total_games else 0

    biggest_riser_label = "N/A"
    biggest_faller_label = "N/A"
    history_window_df = pd.DataFrame(columns=["team", "rank", "week_num", "week_label"])
    movement_rows = pd.DataFrame(columns=["team", "latest_rank", "prior_rank", "move", "direction"])
    if not weekly_ranks.empty and all(c in weekly_ranks.columns for c in ["team", "rank", "week_num", "week_label"]):
        max_week = int(weekly_ranks["week_num"].max())
        summary_start_week = max(1, max_week - max(window_size - 1, 0))
        history_window_df = weekly_ranks[weekly_ranks["week_num"] >= summary_start_week].copy()

        summary_stats = []
        for team, grp in history_window_df.groupby("team"):
            grp = grp.sort_values("week_num")
            if len(grp) >= 2:
                rank_change = int(grp["rank"].iloc[-1] - grp["rank"].iloc[0])
                summary_stats.append({"team": team, "rank_change": rank_change})
        if summary_stats:
            summary_rank = pd.DataFrame(summary_stats)
            biggest_riser = summary_rank.sort_values("rank_change").iloc[0]
            biggest_faller = summary_rank.sort_values("rank_change", ascending=False).iloc[0]
            biggest_riser_label = f"{biggest_riser['team']} ({abs(int(biggest_riser['rank_change']))})"
            biggest_faller_label = f"{biggest_faller['team']} ({abs(int(biggest_faller['rank_change']))})"

        latest_week = int(weekly_ranks["week_num"].max())
        prior_week = max(1, latest_week - 1)
        latest_df = weekly_ranks[weekly_ranks["week_num"] == latest_week][["team", "rank"]].rename(columns={"rank": "latest_rank"})
        prior_df = weekly_ranks[weekly_ranks["week_num"] == prior_week][["team", "rank"]].rename(columns={"rank": "prior_rank"})
        movement_rows = latest_df.merge(prior_df, on="team", how="left")
        movement_rows["move"] = movement_rows["prior_rank"] - movement_rows["latest_rank"]
        movement_rows["direction"] = movement_rows["move"].apply(lambda m: "Riser" if m > 0 else ("Faller" if m < 0 else "Flat"))
        movement_rows = movement_rows.sort_values(["move", "latest_rank"], ascending=[False, True]).head(movement_top_n)

    current_rank_table = pd.DataFrame([
        {"Rank": i + 1, "Team": team, rank_metric_label: rank_metric_values.get(team, 0.0)}
        for i, team in enumerate(rank_order[: min(top_n_rank, len(rank_order))])
    ])
    if rank_metric_label == "Ensemble (Primary)" and isinstance(primary_table, pd.DataFrame) and not primary_table.empty:
        diag_cols = ["Team", "Calibrated Score", "Games Confidence", "SOS Confidence", "Composite Confidence", "Confidence Tier", "Weight Elo", "Weight BCAR", "Weight AdjPyth", "Weight Pyth", "Weight Win"]
        available_diag_cols = [c for c in diag_cols if c in primary_table.columns]
        diag_df = primary_table[available_diag_cols].copy()
        current_rank_table = current_rank_table.merge(diag_df, on="Team", how="left")
    if isinstance(sos, dict):
        current_rank_table["SOS"] = current_rank_table["Team"].map(lambda t: sos.get(t, 0.0))
    top_teams = rank_order[:min(trend_top_n, len(rank_order))]
    trend_pool = history_window_df[history_window_df["team"].isin(top_teams)].copy() if not history_window_df.empty else pd.DataFrame(columns=["team", "rank", "week_num", "week_label"])

    dist_df = pd.DataFrame({"Team": list(stats.keys()), "Win %": [stats[t]["win_pct"] for t in stats]})
    offense_defense_df = pd.DataFrame({
        "Team": list(stats.keys()),
        "Offense (GPG For)": [stats[t]["gf"] / max(stats[t]["games"], 1) for t in stats],
        "Defense (GPG Against)": [stats[t]["ga"] / max(stats[t]["games"], 1) for t in stats],
    })

    imputed_pct = (inferred_results / total_games) if total_games else 0.0
    trust_level = "High" if imputed_pct <= 0.2 else ("Moderate" if imputed_pct <= 0.4 else "Watchlist")

    return {
        "kpi_payload": {
            "top_team": top_team or "N/A",
            "top_team_metric": top_team_metric,
            "biggest_riser_label": biggest_riser_label,
            "biggest_faller_label": biggest_faller_label,
            "total_games": total_games,
            "scored_results": scored_results,
            "inferred_results": inferred_results,
        },
        "current_rank_table": current_rank_table,
        "windowed_rank_history": trend_pool,
        "rank_movement_table": movement_rows,
        "distribution_dataset": dist_df,
        "offense_defense_dataset": offense_defense_df,
        "trust_metrics": {
            "trust_level": trust_level,
            "imputed_pct": imputed_pct,
            "parsed_games": total_games,
            "team_count": len(stats),
            "confidence_progress": min(1.0, max(0.0, 1.0 - imputed_pct)),
        },
        "imputation_impact_dataset": pd.DataFrame([
            {"Category": "Parsed", "Count": scored_results},
            {"Category": "Inferred", "Count": inferred_results},
        ]),
    }

def build_team_profile_3metric_df(stats, adj_vals, elo, sos, min_games):
    rows = []
    eligible = [t for t in stats.keys() if stats[t]["games"] >= min_games and t in adj_vals and t in elo]
    win_ord_local = sorted(eligible, key=lambda t: stats[t]["win_pct"], reverse=True)
    adj_ord_local = sorted(eligible, key=lambda t: adj_vals[t], reverse=True)
    elo_ord_local = sorted(eligible, key=lambda t: elo[t], reverse=True)
    win_rank = {t: i + 1 for i, t in enumerate(win_ord_local)}
    adj_rank = {t: i + 1 for i, t in enumerate(adj_ord_local)}
    elo_rank = {t: i + 1 for i, t in enumerate(elo_ord_local)}
    for team in eligible:
        rows.append(
            {
                "Team": team,
                "WinPct": stats[team]["win_pct"],
                "AdjPyth": adj_vals[team],
                "Elo": elo[team],
                "Games": stats[team]["games"],
                "SOS": sos.get(team, 0.0),
                "WinRank": win_rank.get(team),
                "AdjRank": adj_rank.get(team),
                "EloRank": elo_rank.get(team),
            }
        )
    return pd.DataFrame(rows)
def parse_scores_text(text):
    records = []
    for line in text.splitlines():
        records.extend(_parse_line(line))
    return pd.DataFrame(records)

def parse_expert_order_text(text):
    teams = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+\s*[\)\.\-:]\s*", "", line)
        line = re.sub(r"^\s*#\s*\d+\s*", "", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if line:
            teams.append(_normalize_team_name(line))
    deduped = []
    seen = set()
    for team in teams:
        key = team.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(team)
    return deduped

def compute_expert_fit(method_order, expert_order, top_n=25):
    method_top = method_order[:top_n]
    expert_top = expert_order[:top_n]
    method_ranks = {team: idx + 1 for idx, team in enumerate(method_top)}
    expert_ranks = {team: idx + 1 for idx, team in enumerate(expert_top)}
    overlap_25 = sorted(set(method_ranks).intersection(expert_ranks))
    overlap_10_count = len(set(method_order[:10]).intersection(set(expert_order[:10])))
    mae = None
    if overlap_25:
        mae = sum(abs(method_ranks[t] - expert_ranks[t]) for t in overlap_25) / len(overlap_25)
    return {
        "top25_overlap_count": len(overlap_25),
        "top10_overlap_count": overlap_10_count,
        "mean_abs_rank_error": mae,
        "deltas": [
            {
                "Team": t,
                "Method Rank": method_ranks[t],
                "Expert Rank": expert_ranks[t],
                "Abs Delta": abs(method_ranks[t] - expert_ranks[t]),
                "Signed Delta (method-expert)": method_ranks[t] - expert_ranks[t],
            }
            for t in sorted(overlap_25, key=lambda team: abs(method_ranks[team] - expert_ranks[team]), reverse=True)
        ],
    }




def render_dashboard_metric_charts(stats, adj_vals, elo, sos, min_games_default=0):
    st.markdown("### Metric-space charts")
    max_games = max((sts["games"] for sts in stats.values()), default=1)
    min_games_plot = st.slider("Minimum games", min_value=0, max_value=max_games, value=min_games_default, step=1, key="dashboard_3d_min_games")
    team_search = st.text_input("Team search highlight", value="", key="dashboard_3d_search").strip().lower()
    profile_df = build_team_profile_3metric_df(stats, adj_vals, elo, sos, min_games_plot)
    if profile_df.empty:
        st.info("No teams meet the current minimum games threshold.")
        return
    focus_options = ["(None)"] + sorted(profile_df["Team"].tolist())
    focus_team = st.selectbox("Focus team", options=focus_options, index=0, key="dashboard_3d_focus_team")
    profile_df["SearchHit"] = profile_df["Team"].str.lower().str.contains(team_search, regex=False) if team_search else False
    profile_df["IsFocus"] = profile_df["Team"] == focus_team
    profile_df["Highlight"] = profile_df["IsFocus"] | profile_df["SearchHit"]
    elo_min, elo_max = float(profile_df["Elo"].min()), float(profile_df["Elo"].max())
    profile_df["EloNorm"] = (profile_df["Elo"] - elo_min) / max(elo_max - elo_min, 1e-6)
    sos_min, sos_max = float(profile_df["SOS"].min()), float(profile_df["SOS"].max())
    hover_sel = alt.selection_point(fields=["Team"], on="mouseover", empty=True, name="team_hover")
    base_opacity = alt.condition(hover_sel | alt.datum.Highlight, alt.value(1.0), alt.value(0.25))
    sos_color = alt.Color("SOS:Q", scale=alt.Scale(domain=[sos_min, (sos_min + sos_max)/2.0, sos_max], range=["#B91C1C", "#FACC15", "#15803D"]))

    st.subheader("Win% vs Adjusted Pythagorean")
    chart1 = alt.Chart(profile_df).mark_circle().encode(
        x=alt.X("WinPct:Q", title="Win %"), y=alt.Y("AdjPyth:Q", title="Adjusted Pythagorean"),
        size=alt.Size("EloNorm:Q", scale=alt.Scale(domain=[0, 1], range=[40, 900])), color=sos_color, opacity=base_opacity,
        tooltip=["Team:N", alt.Tooltip("WinPct:Q", format=".3f"), alt.Tooltip("AdjPyth:Q", format=".3f"), alt.Tooltip("Elo:Q", format=".1f"), alt.Tooltip("SOS:Q", format=".3f"), "Games:Q"]
    )
    st.altair_chart((chart1.add_params(hover_sel)).properties(height=420), use_container_width=True)

    gpg_df = profile_df.copy()
    gpg_df["GoalsFor"] = gpg_df["Team"].map(lambda t: stats[t]["gf"] / max(stats[t]["games"], 1))
    gpg_df["GoalsAgainst"] = gpg_df["Team"].map(lambda t: stats[t]["ga"] / max(stats[t]["games"], 1))
    st.subheader("Goals For vs Goals Against")
    chart2 = alt.Chart(gpg_df).mark_circle().encode(
        x=alt.X("GoalsFor:Q", title="Goals For / Game"), y=alt.Y("GoalsAgainst:Q", title="Goals Against / Game", scale=alt.Scale(reverse=True)),
        size=alt.Size("EloNorm:Q", scale=alt.Scale(domain=[0, 1], range=[40, 900])), color=sos_color, opacity=base_opacity,
        tooltip=["Team:N", alt.Tooltip("GoalsFor:Q", format=".2f"), alt.Tooltip("GoalsAgainst:Q", format=".2f"), alt.Tooltip("Elo:Q", format=".1f"), alt.Tooltip("SOS:Q", format=".3f"), "Games:Q"]
    )
    st.altair_chart((chart2.add_params(hover_sel)).properties(height=420), use_container_width=True)

    bcar_field = "BCAR" if "BCAR" in gpg_df.columns else ("BCAR Score" if "BCAR Score" in gpg_df.columns else None)
    if bcar_field:
        bcar_df = gpg_df.copy()
        bcar_df[bcar_field] = pd.to_numeric(bcar_df[bcar_field], errors="coerce")
        bcar_df = bcar_df.dropna(subset=["Elo", "SOS", "AdjPyth", bcar_field, "Team", "Games"])
        if not bcar_df.empty:
            st.subheader("Elo vs BCAR")
            chart3 = alt.Chart(bcar_df).mark_circle().encode(
                x=alt.X("Elo:Q", title="Elo"),
                y=alt.Y(f"{bcar_field}:Q", title="BCAR"),
                size=alt.Size("AdjPyth:Q", scale=alt.Scale(range=[40, 900])),
                color=sos_color,
                opacity=base_opacity,
                tooltip=[
                    "Team:N",
                    alt.Tooltip("Elo:Q", format=".1f"),
                    alt.Tooltip(f"{bcar_field}:Q", format=".3f"),
                    alt.Tooltip("SOS:Q", format=".3f"),
                    alt.Tooltip("AdjPyth:Q", format=".3f"),
                    "Games:Q",
                ],
            )
            st.altair_chart((chart3.add_params(hover_sel)).properties(height=420), use_container_width=True)




def build_rank_overview_chart(top_n_df, metric_label, metric_format):
    return alt.Chart(top_n_df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Team:N", sort="-x", title=None),
        x=alt.X(f"{metric_label}:Q", title=metric_label, axis=alt.Axis(format=metric_format)),
        color=alt.condition(alt.datum.Rank == 1, alt.value(RANK_TIER_COLORS["elite"]), alt.value(RANK_TIER_COLORS["contender"])),
        tooltip=[alt.Tooltip("Rank:Q", format=CHART_FORMATS["int0"]), "Team:N", alt.Tooltip(f"{metric_label}:Q", format=metric_format)],
    ).properties(title=f"Current rank bar chart (Top {len(top_n_df)})")


def build_trend_chart(trend_pool):
    return alt.Chart(trend_pool).mark_line(point=True).encode(
        x=alt.X("week_num:Q", title="Week"),
        y=alt.Y("rank:Q", title="Rank (1 is best)", scale=alt.Scale(reverse=True)),
        color=alt.Color("team:N", legend=alt.Legend(title="Top teams")),
        tooltip=["team:N", "week_label:N", alt.Tooltip("rank:Q", format=CHART_FORMATS["int0"])],
    ).properties(title="Weekly rank trajectory")


def build_movement_chart(movement_rows):
    return alt.Chart(movement_rows).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("team:N", sort=None, title=None),
        y=alt.Y("move:Q", title="Week-over-week rank change (+ is better)"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Riser", "Flat", "Faller"], range=[SEMANTIC_COLORS["positive"], SEMANTIC_COLORS["neutral"], SEMANTIC_COLORS["negative"]])),
        tooltip=["team:N", alt.Tooltip("latest_rank:Q", format=CHART_FORMATS["int0"]), alt.Tooltip("prior_rank:Q", format=CHART_FORMATS["int0"]), alt.Tooltip("move:Q", format=CHART_FORMATS["int0"]), "direction:N"],
    ).properties(title="Movement (risers/fallers)")


def build_distribution_chart(dist_df):
    return alt.Chart(dist_df).mark_bar().encode(
        x=alt.X("Win %:Q", bin=alt.Bin(maxbins=12), title="Win % bins"),
        y=alt.Y("count():Q", title="Teams"),
        tooltip=[alt.Tooltip("count():Q", title="Teams in bin", format=CHART_FORMATS["int0"])],
    ).properties(title="Metric distribution")


def build_offense_defense_chart(scatter_df, top_team):
    return alt.Chart(scatter_df).mark_circle(size=90, opacity=0.8).encode(
        x=alt.X("Offense (GPG For):Q"),
        y=alt.Y("Defense (GPG Against):Q", scale=alt.Scale(reverse=True)),
        color=alt.condition(alt.datum.Team == top_team, alt.value(RANK_TIER_COLORS["elite"]), alt.value(RANK_TIER_COLORS["support"])),
        tooltip=["Team:N", alt.Tooltip("Offense (GPG For):Q", format=CHART_FORMATS["float2"]), alt.Tooltip("Defense (GPG Against):Q", format=CHART_FORMATS["float2"])],
    ).properties(title="Offense vs defense scatter")


def render_dashboard_header_kpis(kpi, metric_lens, metric_format):
    render_spacing("section")
    render_typography("subtitle", "Band A · Hero summary")
    hero_cols = st.columns(6)
    render_kpi_card(hero_cols[0], "Current #1", kpi["top_team"], delta=(f"{metric_lens} {kpi['top_team_metric']:{metric_format}}" if kpi["top_team_metric"] is not None else None), caption=f"Best current league position by {metric_lens} ranking.")
    render_kpi_card(hero_cols[1], "Biggest riser", kpi["biggest_riser_label"], caption="Largest upward movement in the recent 3-week window.")
    render_kpi_card(hero_cols[2], "Biggest faller", kpi["biggest_faller_label"], caption="Largest downward movement in the recent 3-week window.")
    render_kpi_card(hero_cols[3], "Total games", kpi["total_games"], caption="All parsed matchups currently included in this model run.")
    render_kpi_card(hero_cols[4], "Scored games", kpi["scored_results"], caption="Games with explicit scores recorded in source files.")
    render_kpi_card(hero_cols[5], "Estimated games", kpi["inferred_results"], caption="Games where scores were inferred from model defaults.")


def render_dashboard_controls():
    ctl_a, ctl_b, ctl_c = st.columns(3)
    dashboard_whole_season = st.session_state.get("dashboard_time_window") == "All"
    ctl_a.selectbox("Sectional", options=["All Sectionals"], key="dashboard_sectional_selector", on_change=_track_on_change, args=("filter_usage", "dashboard_sectional_selector"))
    ctl_b.selectbox("Division", options=["All Divisions"], key="dashboard_division_selector", on_change=_track_on_change, args=("filter_usage", "dashboard_division_selector"))
    ctl_c.checkbox("Whole season", value=dashboard_whole_season, key="dashboard_whole_season_toggle", on_change=_on_dashboard_whole_season_toggle)
    st.selectbox(
        "Timeframe",
        options=["Last 4 weeks", "All"],
        key="dashboard_time_window",
        on_change=_on_dashboard_time_window_change,
    )
    st.selectbox(
        "Metric lens",
        options=METRIC_LENS_OPTIONS,
        key="dashboard_metric_lens",
        format_func=lambda key: METRIC_OPTION_LABELS.get(key, key),
        help="BCAR is the recommended default for ranking decisions. Secondary methods remain available for cross-checking.",
        on_change=_track_on_change,
        args=("metric_switch", "dashboard_metric_lens"),
    )
    if st.button("Reset to BCAR defaults", key="dashboard_reset_defaults"):
        st.session_state["dashboard_metric_lens"] = DEFAULT_DASHBOARD_METRIC_LENS
        st.session_state["dashboard_time_window"] = "Last 4 weeks"
        st.session_state["dashboard_whole_season_toggle"] = False
        track_ui_event("filter_usage", key="dashboard_reset_defaults", value="clicked")
        _sync_dashboard_timeframe_query_params()
    with st.expander("See all metrics", expanded=False):
        track_ui_event("metrics_expand_rendered", state="rendered")
        track_ui_event("metrics_expand_opened", opened=True) if st.session_state.get("dashboard_metrics_expanded") else None
        st.caption("Secondary metrics are shown for context only and should not be treated as equally weighted with BCAR.")
        trend_top_n = st.slider("Trend teams shown", min_value=4, max_value=25, value=8, step=1, key="dashboard_trend_top_n")
        movement_top_n = st.slider("Movement rows shown", min_value=4, max_value=25, value=8, step=1, key="dashboard_movement_top_n")
    return trend_top_n, movement_top_n


def render_rank_overview_panel(dashboard_vm, metric_label, metric_format):
    top_n_df = dashboard_vm["current_rank_table"]
    if top_n_df.empty:
        st.info("No rank table available for the current filter window.")
        return
    st.altair_chart(apply_chart_theme(build_rank_overview_chart(top_n_df, metric_label, metric_format)), use_container_width=True)
    if metric_label == "Ensemble (Primary)":
        preferred_cols = ["Rank", "Team", "Ensemble Score", "Games Confidence", "SOS Confidence", "Composite Confidence", "Confidence Tier", "Weight Elo", "Weight BCAR", "Weight AdjPyth", "Weight Pyth", "Weight Win", "SOS"]
        dashboard_table = top_n_df.rename(columns={"Calibrated Score": "Ensemble Score"})
        show_cols = [c for c in preferred_cols if c in dashboard_table.columns]
        if show_cols:
            st.dataframe(format_confidence_columns(dashboard_table[show_cols]), use_container_width=True, hide_index=True)
        render_primary_rank_caption_block()


def render_trend_panel(dashboard_vm):
    trend_pool = dashboard_vm["windowed_rank_history"]
    if trend_pool.empty:
        st.info("Weekly rank trajectory appears after multiple weekly files are available.")
        return
    st.altair_chart(apply_chart_theme(build_trend_chart(trend_pool)), use_container_width=True)


def render_movement_panel(dashboard_vm):
    movement_rows = dashboard_vm["rank_movement_table"]
    if movement_rows.empty:
        st.info("Rank movement appears when at least two ranking periods are available.")
        return
    st.altair_chart(apply_chart_theme(build_movement_chart(movement_rows)), use_container_width=True)
    st.dataframe(movement_rows.rename(columns={"team": "Team", "latest_rank": "Current Rank", "prior_rank": "Prior Rank", "move": "Δ Rank", "direction": "Direction"}), use_container_width=True, hide_index=True)


def render_distribution_panel(dashboard_vm):
    st.altair_chart(apply_chart_theme(build_distribution_chart(dashboard_vm["distribution_dataset"])), use_container_width=True)


def render_offense_defense_panel(dashboard_vm, top_team):
    st.altair_chart(apply_chart_theme(build_offense_defense_chart(dashboard_vm["offense_defense_dataset"], top_team)), use_container_width=True)


def render_trust_imputation_panel(dashboard_vm):
    trust_metrics = dashboard_vm["trust_metrics"]
    impact_df = dashboard_vm["imputation_impact_dataset"]
    st.markdown("#### Data quality / trust capsule")
    st.caption("Quick confidence read before acting on rankings.")
    st.metric("Trust level", trust_metrics["trust_level"])
    st.progress(trust_metrics["confidence_progress"])
    st.caption(f"Inferred share: {trust_metrics['imputed_pct']:.1%} · Parsed games: {trust_metrics['parsed_games']} · Teams: {trust_metrics['team_count']}")
    st.caption("Lower inferred share generally means more stable ranking confidence.")
    ensemble_table = dashboard_vm.get("primary_rank_table")
    if ensemble_table is not None and not ensemble_table.empty and "Composite Confidence" in ensemble_table.columns:
        avg_conf = float(ensemble_table["Composite Confidence"].mean())
        st.caption(f"Ensemble confidence snapshot: average composite confidence is {avg_conf:.1f}/100 ({confidence_tier_label(avg_conf)}).")
        st.caption("Composite confidence blends games volume, SOS normalization + schedule breadth, and imputation reliability.")
    if not impact_df.empty:
        st.dataframe(impact_df, hide_index=True, use_container_width=True)
# ---------------- I/O ---------------- #
@st.cache_data
def load_scores():
    games_df, _ = load_games_pipeline(DATA_DIR)
    return games_df

# ---------------- Inference ---------------- #
@st.cache_data
def infer_default_scores(games_df, stats):
    df = games_df.copy()
    if 'is_imputed' not in df.columns:
        df['is_imputed'] = df['score1'].isna()
    else:
        df['is_imputed'] = df['is_imputed'].fillna(df['score1'].isna()).astype(bool)
    mask = df['score1'].isna()
    for idx in df[mask].index:
        t1, t2 = df.at[idx,'team1'], df.at[idx,'team2']
        st1, st2 = stats[t1], stats[t2]
        if st1['games'] and st2['games']:
            avg1 = (st1['gf']/st1['games'] + st2['ga']/st2['games'])/2
            avg2 = (st2['gf']/st2['games'] + st1['ga']/st1['games'])/2
        else:
            avg1 = avg2 = 1
        loser_avg, winner_avg = sorted([avg1,avg2])
        loser_score = int(math.floor(loser_avg + 0.5))
        winner_score = int(math.floor(winner_avg + 0.5)) + 1
        # team1 always winner by "d." notation
        df.at[idx,'score1'] = winner_score
        df.at[idx,'score2'] = loser_score
        df.at[idx,'is_imputed'] = True
    return df

def add_imputation_markers(df, include_estimated_scores=True, highlight_estimated_games=True):
    view = df.copy()
    has_imputed = view.get("is_imputed", pd.Series(False, index=view.index)).fillna(False).astype(bool)
    if include_estimated_scores:
        marker = " 🧩" if highlight_estimated_games else ""
        view["Team"] = view["Team"].astype(str) + has_imputed.map(lambda x: marker if x else "")
    return view

# ---------------- Stats ---------------- #
@st.cache_data
def compute_stats(games):
    stats = {}
    h2h = {}

    def ensure_h2h_record(me, opp):
        key = (me, opp)
        if key not in h2h:
            h2h[key] = {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0}
        return h2h[key]

    teams = set(games['team1']).union(games['team2'])
    for t in teams:
        stats[t] = {'wins':0,'losses':0,'ties':0,'gf':0,'ga':0,'games':0,'opponents':[]}
    for _,r in games.iterrows():
        t1,t2,s1,s2 = r.team1, r.team2, int(r.score1), int(r.score2)
        for me,opp,ms,os in [(t1,t2,s1,s2),(t2,t1,s2,s1)]:
            stats[me]['gf'] += ms
            stats[me]['ga'] += os
            stats[me]['games'] += 1
            stats[me]['opponents'].append(opp)
            h2h_record = ensure_h2h_record(me, opp)
            h2h_record['gf'] += ms
            h2h_record['ga'] += os
        if s1>s2:
            stats[t1]['wins']+=1
            stats[t2]['losses']+=1
            ensure_h2h_record(t1, t2)['wins']+=1
        elif s2>s1:
            stats[t2]['wins']+=1
            stats[t1]['losses']+=1
            ensure_h2h_record(t2, t1)['wins']+=1
        else:
            stats[t1]['ties']+=1
            stats[t2]['ties']+=1
        ensure_h2h_record(t1, t2)['games']+=1
        ensure_h2h_record(t2, t1)['games']+=1
    for t,st in stats.items():
        tot=st['wins']+st['losses']+st['ties']
        st['win_pct']=st['wins']/tot if tot else 0
        st['gd']=st['gf']-st['ga']
    return stats,h2h

def games_df_hash(games_df):
    if games_df.empty:
        return "empty"
    ordered = games_df.sort_values(["team1", "team2", "score1", "score2"], kind="mergesort").reset_index(drop=True)
    payload = ordered.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

@st.cache_data
def build_matchup_aggregate_cached(games_hash, games_csv):
    _ = games_hash
    df = pd.read_csv(StringIO(games_csv))
    agg = defaultdict(lambda: {"wins": 0, "losses": 0, "gf": 0, "ga": 0, "games": 0})
    for r in df.itertuples(index=False):
        t1, t2, s1, s2 = r.team1, r.team2, int(r.score1), int(r.score2)
        rec1 = agg[(t1, t2)]
        rec1["gf"] += s1
        rec1["ga"] += s2
        rec1["games"] += 1
        if s1 > s2:
            rec1["wins"] += 1
        elif s1 < s2:
            rec1["losses"] += 1
        rec2 = agg[(t2, t1)]
        rec2["gf"] += s2
        rec2["ga"] += s1
        rec2["games"] += 1
        if s2 > s1:
            rec2["wins"] += 1
        elif s2 < s1:
            rec2["losses"] += 1
    return dict(agg)

def build_matchup_aggregate(games_df):
    ghash = games_df_hash(games_df)
    ordered = games_df.sort_values(["team1", "team2", "score1", "score2"], kind="mergesort").reset_index(drop=True)
    return build_matchup_aggregate_cached(ghash, ordered.to_csv(index=False))

# ---------------- Metrics ---------------- #
@st.cache_data
def compute_sos(stats):
    sos={}
    for t,st in stats.items():
        opps=st['opponents']
        sos[t]=sum(stats[o]['win_pct'] for o in opps)/len(opps) if opps else 0
    return sos

@st.cache_data
def compute_pythag(games, stats, exp=2, imputed_mode="full", imputed_weight=1.0):
    p={}
    weighted_gf = defaultdict(float)
    weighted_ga = defaultdict(float)
    for _, r in games.iterrows():
        is_imputed = bool(r.get('is_imputed', False))
        for me, ms, os_ in [(r.team1, float(r.score1), float(r.score2)), (r.team2, float(r.score2), float(r.score1))]:
            if is_imputed and imputed_mode == "binary":
                if ms > os_:
                    contrib_for, contrib_against = 1.0, 0.0
                elif ms < os_:
                    contrib_for, contrib_against = 0.0, 1.0
                else:
                    contrib_for = contrib_against = 0.5
            else:
                contrib_for, contrib_against = ms, os_
            weight = imputed_weight if (is_imputed and imputed_mode == "down_weight") else 1.0
            weighted_gf[me] += contrib_for * weight
            weighted_ga[me] += contrib_against * weight
    for t,st in stats.items():
        gf,ga=weighted_gf[t],weighted_ga[t]
        p[t]=gf**exp/(gf**exp+ga**exp) if gf+ga>0 else 0
    return p

# Logistic scaling function
def logistic(x, k, x0):
    return 1/(1 + math.exp(-k * (x - x0)))

# Adjusted Pythagorean with logistic blend
@st.cache_data
def compute_adjusted_pythag(games, stats, exp=2, k=10, x0=0.5, imputed_mode="full", imputed_weight=1.0):
    adj_ms = defaultdict(float)
    adj_os = defaultdict(float)
    for _,r in games.iterrows():
        is_imputed = bool(r.get('is_imputed', False))
        for me,opp,ms,os_ in [(r.team1,r.team2,r.score1,r.score2),(r.team2,r.team1,r.score2,r.score1)]:
            # strength factor via logistic on opponent win_pct
            win_pct = stats[opp]['win_pct']
            s = logistic(win_pct, k, x0)
            # expected goals
            st1, st2 = stats[me], stats[opp]
            if st1['games'] and st2['games']:
                E_ms = (st1['gf']/st1['games'] + st2['ga']/st2['games']) / 2
                E_os = (st2['gf']/st2['games'] + st1['ga']/st1['games']) / 2
            else:
                E_ms = E_os = 1
            if is_imputed and imputed_mode == "binary":
                ms_eff = 1.0 if ms > os_ else (0.0 if ms < os_ else 0.5)
                os_eff = 1.0 - ms_eff
            else:
                ms_eff, os_eff = ms, os_
            weight = imputed_weight if (is_imputed and imputed_mode == "down_weight") else 1.0
            adj_ms[me] += (E_ms + (ms_eff - E_ms) * s) * weight
            adj_os[me] += (E_os + (os_eff - E_os) * s) * weight
    adj = {}
    for t,st in stats.items():
        g = st['games']
        agf = adj_ms[t]/g if g else 0
        aga = adj_os[t]/g if g else 0
        adj[t] = agf**exp/(agf**exp + aga**exp) if agf+aga>0 else 0
    return adj

@st.cache_data
def compute_confidence_adjusted_bayesian_rank(
    games,
    stats,
    tau0=1.0,
    m_cap=4,
    resume_a=0.35,
    resume_b=0.65,
    resume_c=0.80,
    beta=0.50,
    lambda_max=0.30,
    n_star=10.0,
):
    teams = sorted(stats.keys())
    if not teams:
        return {"order": [], "table": pd.DataFrame()}

    team_ix = {t: i for i, t in enumerate(teams)}
    offense = {t: 0.0 for t in teams}
    defense = {t: 0.0 for t in teams}
    team_games = defaultdict(int)
    opp_set = {t: set() for t in teams}

    # Penalized iterative offense/defense fit (fast approximation to Bayesian shrinkage)
    for _ in range(20):
        new_offense = {}
        new_defense = {}
        for t in teams:
            gf_terms, ga_terms = [], []
            for r in games.itertuples():
                if r.team1 == t:
                    gf_terms.append(math.log1p(float(r.score1)) - defense[r.team2])
                    ga_terms.append(math.log1p(float(r.score2)) - offense[r.team2])
                    team_games[t] += 1
                    opp_set[t].add(r.team2)
                elif r.team2 == t:
                    gf_terms.append(math.log1p(float(r.score2)) - defense[r.team1])
                    ga_terms.append(math.log1p(float(r.score1)) - offense[r.team1])
                    team_games[t] += 1
                    opp_set[t].add(r.team1)
            lam = tau0 ** 2
            new_offense[t] = sum(gf_terms) / (len(gf_terms) + lam) if gf_terms else 0.0
            new_defense[t] = sum(ga_terms) / (len(ga_terms) + lam) if ga_terms else 0.0
        offense, defense = new_offense, new_defense

    strength = {t: offense[t] - defense[t] for t in teams}
    s_vals = pd.Series(strength)
    s_med = float(s_vals.median())
    s_sd = max(float(s_vals.std(ddof=0)), 1e-6)
    sigma = {t: tau0 / math.sqrt(max(1, stats[t]["games"])) for t in teams}
    certainty = {t: max(0.0, 1.0 - (sigma[t] ** 2 / max(tau0 ** 2, 1e-6))) for t in teams}
    quality = {t: 1.0 / (1.0 + math.exp(-(strength[t] - s_med) / s_sd)) for t in teams}

    def margin_cap(m):
        return math.log1p(min(abs(float(m)), m_cap)) / math.log1p(m_cap)

    rows = []
    for t in teams:
        g = stats[t]["games"]
        lam_n = lambda_max * (1.0 - math.exp(-(g / max(n_star, 1e-6))))
        sos_num = sos_den = sov_num = sov_den = bl_num = bl_den = 0.0
        for r in games.itertuples():
            if r.team1 == t:
                opp, m = r.team2, float(r.score1) - float(r.score2)
            elif r.team2 == t:
                opp, m = r.team1, float(r.score2) - float(r.score1)
            else:
                continue
            c = certainty.get(opp, 0.0)
            w = 1.0
            sos_num += w * c * strength.get(opp, 0.0)
            sos_den += w * c
            if m > 0:
                sov_num += w * c * quality.get(opp, 0.5) * margin_cap(m)
                sov_den += w * c
            elif m < 0:
                bl_num += w * c * (1.0 - quality.get(opp, 0.5)) * margin_cap(m)
                bl_den += w * c
        sos_i = sos_num / sos_den if sos_den else 0.0
        sov_i = sov_num / sov_den if sov_den else 0.0
        bl_i = bl_num / bl_den if bl_den else 0.0
        resume_i = resume_a * sos_i + resume_b * sov_i - resume_c * bl_i
        rows.append({
            "Team": t,
            "Strength": strength[t],
            "Sigma": sigma[t],
            "SOS_BCAR": sos_i,
            "SOV_BCAR": sov_i,
            "BL_BCAR": bl_i,
            "Resume_BCAR": resume_i,
            "Lambda_BCAR": lam_n,
            "Games": g,
            "UniqueOpp": len(opp_set[t]),
            "ModelConf": max(0.0, min(1.0, 1.0 - (sigma[t] ** 2 / max(tau0 ** 2, 1e-6)))),
            "SchedConf": sos_den / max(g, 1),
            "Coverage": (g / (g + 6.0)) * (len(opp_set[t]) / (len(opp_set[t]) + 4.0)),
        })
    df = pd.DataFrame(rows)
    for col in ["Strength", "Resume_BCAR", "Sigma"]:
        sd = float(df[col].std(ddof=0))
        df[f"Z_{col}"] = (df[col] - float(df[col].mean())) / (sd if sd > 1e-9 else 1.0)
    df["BCAR Score"] = df["Z_Strength"] + df["Lambda_BCAR"] * df["Z_Resume_BCAR"] - beta * df["Z_Sigma"]
    df["Confidence"] = 100 * (df["ModelConf"] * df["SchedConf"].clip(lower=0) * df["Coverage"]).clip(lower=0) ** (1 / 3)
    df["Strength 95% Low"] = df["Strength"] - 1.96 * df["Sigma"]
    df["Strength 95% High"] = df["Strength"] + 1.96 * df["Sigma"]
    df = df.sort_values("BCAR Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    order = df["Team"].tolist()
    return {"order": order, "table": df}

@st.cache_data
def compute_hybrid_rankings(games, stats, h2h, sos, params):
    teams = sorted(stats.keys())
    if not teams:
        return {"order": [], "table": pd.DataFrame(), "ratings": {}}
    k0 = float(params.get("k0", 0.55))
    epsilon = float(params.get("epsilon", 1e-6))
    margin_scale_s = max(float(params.get("margin_scale_s", 4.0)), 1e-6)
    ridge_lambda = max(float(params.get("ridge_lambda", 0.08)), 0.0)
    lambda_sos_max = float(params.get("lambda_sos_max", 0.25))
    lambda_sov = float(params.get("lambda_sov", 0.35))
    lambda_var = float(params.get("lambda_var", 0.15))
    home_advantage = float(params.get("home_advantage", 0.0))
    poisson_reg = float(params.get("poisson_regularization", 0.2))
    bcar_bundle = compute_confidence_adjusted_bayesian_rank(games, stats)
    bcar_lookup = bcar_bundle["table"].set_index("Team") if not bcar_bundle["table"].empty else pd.DataFrame()
    rows = []
    for t in teams:
        g = max(stats[t]["games"], 1)
        gd_pg = (stats[t]["gf"] - stats[t]["ga"]) / g
        win_pct = stats[t]["win_pct"]
        sos_term = sos.get(t, 0.0)
        margins = []
        opp_strength = []
        for r in games.itertuples():
            if r.team1 == t:
                m, opp = float(r.score1) - float(r.score2), r.team2
            elif r.team2 == t:
                m, opp = float(r.score2) - float(r.score1), r.team1
            else:
                continue
            margins.append(m)
            opp_strength.append(stats.get(opp, {}).get("win_pct", 0.0))
        sov = (sum(max(0.0, m) * os_ for m, os_ in zip(margins, opp_strength)) / max(len(margins), 1)) if margins else 0.0
        var_pen = (pd.Series(margins).std(ddof=0) if len(margins) >= 2 else 0.0)
        bcar_strength = float(bcar_lookup.at[t, "Strength"]) if (not bcar_lookup.empty and t in bcar_lookup.index) else 0.0
        rating = (k0 * win_pct) + ((1.0 - k0) * math.tanh((gd_pg + home_advantage) / margin_scale_s))
        rating += lambda_sos_max * sos_term + lambda_sov * sov + 0.20 * bcar_strength
        rating -= (lambda_var * var_pen + ridge_lambda * (1.0 / math.sqrt(g + epsilon)))
        rating_se = max(epsilon, math.sqrt((var_pen + poisson_reg) / g))
        ci_low, ci_high = rating - 1.96 * rating_se, rating + 1.96 * rating_se
        confidence = max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, rating_se / max(abs(rating), 0.75)))))
        rows.append({
            "Team": t, "rating": rating, "rating_se": rating_se, "ci_low": ci_low, "ci_high": ci_high,
            "confidence": confidence, "strength_of_schedule": sos_term, "strength_of_victory": sov,
            "volatility_penalty": var_pen,
        })
    df = pd.DataFrame(rows).sort_values("rating", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return {"order": df["Team"].tolist(), "table": df, "ratings": dict(zip(df["Team"], df["rating"]))}

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def build_hybrid_matchup_projection(team_a, team_b, stats, hybrid_ratings, params):
    ga = max(stats[team_a]["games"], 1)
    gb = max(stats[team_b]["games"], 1)
    a_for = stats[team_a]["gf"] / ga
    a_against = stats[team_a]["ga"] / ga
    b_for = stats[team_b]["gf"] / gb
    b_against = stats[team_b]["ga"] / gb
    base_a = max(0.2, (a_for + b_against) / 2.0)
    base_b = max(0.2, (b_for + a_against) / 2.0)
    rating_delta = hybrid_ratings.get(team_a, 0.0) - hybrid_ratings.get(team_b, 0.0)
    boost = math.tanh(rating_delta)
    lam_a = max(0.2, base_a * (1.0 + 0.25 * boost))
    lam_b = max(0.2, base_b * (1.0 - 0.25 * boost))
    max_goals = 8
    p_win = p_draw = p_loss = 0.0
    score_probs = []
    for i in range(max_goals + 1):
        pi = poisson_pmf(i, lam_a)
        for j in range(max_goals + 1):
            p = pi * poisson_pmf(j, lam_b)
            if i > j: p_win += p
            elif i == j: p_draw += p
            else: p_loss += p
            score_probs.append((i, j, p))
    top_scores = sorted(score_probs, key=lambda x: x[2], reverse=True)[:5]
    exp_gd = lam_a - lam_b
    exp_total = lam_a + lam_b
    gd_interval = (exp_gd - 1.96 * math.sqrt(exp_total), exp_gd + 1.96 * math.sqrt(exp_total))
    tg_interval = (max(0.0, exp_total - 1.96 * math.sqrt(exp_total)), exp_total + 1.96 * math.sqrt(exp_total))
    confidence = max(0.0, min(100.0, 100.0 * abs(p_win - p_loss)))
    return {"lam_a": lam_a, "lam_b": lam_b, "p_win": p_win, "p_draw": p_draw, "p_loss": p_loss, "top_scores": top_scores, "gd_interval": gd_interval, "tg_interval": tg_interval, "confidence": confidence}


def _to_schedule_adjusted_games(games: pd.DataFrame) -> pd.DataFrame:
    cols = ["team1", "team2", "score1", "score2"]
    base = games[cols].dropna(subset=["score1", "score2"]).copy()
    return base.rename(columns={"team1": "team_a", "team2": "team_b", "score1": "goals_a", "score2": "goals_b"})


@st.cache_data
def compute_schedule_adjusted_hybrid(games: pd.DataFrame, params: dict) -> dict:
    converted = _to_schedule_adjusted_games(games)
    if converted.empty:
        return {"order": [], "table": pd.DataFrame(), "model": None}
    cfg = HybridRankingConfig(
        ridge_lambda=max(float(params.get("ridge_lambda", 4.0)), 0.0),
        k0=float(params.get("k0", 8.0)),
        eps=float(params.get("epsilon", 1e-6)),
        margin_scale=max(float(params.get("margin_scale_s", 1.0)), 1e-6),
        lambda_sos_max=float(params.get("lambda_sos_max", 0.25)),
        lambda_sov=float(params.get("lambda_sov", 0.2)),
        lambda_var=float(params.get("lambda_var", 0.1)),
        use_home_indicator=bool(params.get("use_home_indicator", True)),
    )
    model = ScheduleAdjustedGoalStrengthRanker(cfg).fit(converted)
    table = model.rankings_table().rename(
        columns={
            "team": "Team",
            "rating_ci_low": "ci_low",
            "rating_ci_high": "ci_high",
            "sos": "strength_of_schedule",
            "sov": "strength_of_victory",
            "volatility": "volatility_penalty",
        }
    )
    table["Rank"] = table.index + 1
    return {"order": table["Team"].tolist(), "table": table, "model": model}

# ---------------- Rankings ---------------- #
@st.cache_data
def rank_win_pct(stats,h2h):
    def cmp_pairwise(a,b):
        h = h2h.get((a,b),{'wins':0,'games':0})
        if h['games']:
            p = h['wins']/h['games']
            if p != 0.5:
                return -1 if p > 0.5 else 1
        if stats[a]['gd'] != stats[b]['gd']:
            return -1 if stats[a]['gd'] > stats[b]['gd'] else 1
        return -1 if a < b else (1 if a > b else 0)

    ordered = sorted(stats.keys())
    grouped_by_win_pct = defaultdict(list)
    for team in ordered:
        grouped_by_win_pct[stats[team]['win_pct']].append(team)

    final_order = []
    for win_pct in sorted(grouped_by_win_pct.keys(), reverse=True):
        group = grouped_by_win_pct[win_pct]
        if len(group) <= 1:
            final_order.extend(group)
            continue

        if len(group) == 2:
            final_order.extend(sorted(group, key=cmp_to_key(cmp_pairwise)))
            continue

        # 3+ tied teams: use mini-table among tied teams.
        mini = {
            t: {"wins": 0, "losses": 0, "ties": 0, "points": 0, "gd": 0, "games": 0}
            for t in group
        }
        for team in group:
            for opp in group:
                if team == opp:
                    continue
                rec = h2h.get((team, opp), {"wins": 0, "games": 0, "gf": 0, "ga": 0})
                if rec["games"] <= 0:
                    continue
                wins = rec["wins"]
                losses = h2h.get((opp, team), {"wins": 0}).get("wins", 0)
                ties = rec["games"] - wins - losses
                mini[team]["wins"] += wins
                mini[team]["losses"] += losses
                mini[team]["ties"] += ties
                mini[team]["games"] += rec["games"]
                mini[team]["gd"] += rec.get("gf", 0) - rec.get("ga", 0)
                mini[team]["points"] += (2 * wins) + ties

        def mini_sort_key(team):
            m = mini[team]
            mini_win_pct = (m["wins"] + 0.5 * m["ties"]) / m["games"] if m["games"] else 0.0
            return (-m["points"], -mini_win_pct, -m["gd"], -stats[team]["gd"], team)

        final_order.extend(sorted(group, key=mini_sort_key))
    return final_order
@st.cache_data
def rank_pythag(stats,p):
    return sorted(stats.keys(),key=lambda t:p[t],reverse=True)

def rank_adj_pyth(stats,games,h2h,k=10,x0=0.5, imputed_mode="full", imputed_weight=1.0):
    vals = compute_adjusted_pythag(games,stats,k=k,x0=x0, imputed_mode=imputed_mode, imputed_weight=imputed_weight)
    order = sorted(stats.keys(),key=lambda t:vals[t],reverse=True)
    final = []
    eps = 1e-4
    for t in order:
        if final:
            prev = final[-1]
            if abs(vals[prev]-vals[t])<eps:
                h=h2h.get((t,prev),{'wins':0,'games':0})
                if h['games'] and h['wins']/h['games']>0.5:
                    final[-1],t = t,prev
        final.append(t)
    return final, vals
@st.cache_data
def compute_elo(games, initial=1500, k=32, phase_k_enabled=False, early_phase_games=40, early_phase_multiplier=1.15, late_phase_multiplier=0.9):
    teams=set(games['team1']).union(games['team2'])
    R={t:initial for t in teams}
    total_games = len(games)
    phase_cutoff = max(0, min(int(early_phase_games), total_games))
    for idx, r in enumerate(games.itertuples(), start=1):
        game_k = k * (early_phase_multiplier if (phase_k_enabled and idx <= phase_cutoff) else (late_phase_multiplier if phase_k_enabled else 1.0))
        a,b,sa,sb=r.team1,r.team2,r.score1,r.score2
        ea=1/(1+10**((R[b]-R[a])/400)); eb=1-ea
        aa,ab = (1,0) if sa>sb else ((0,1) if sb>sa else (0.5,0.5))
        R[a]+=game_k*(aa-ea); R[b]+=game_k*(ab-eb)
    return R
@st.cache_data
def rank_elo(stats,elo):
    return sorted(stats.keys(),key=lambda t:elo[t],reverse=True)

def sort_teams_by_mode(mode, teams, stats, sos, ensemble_df, elo=None, bcar_table=None, adj_vals=None, pyth_vals=None):
    base_teams = [t for t in teams if t in stats]
    ensemble_rank = {}
    if ensemble_df is not None and not ensemble_df.empty and "Team" in ensemble_df.columns:
        for i, team in enumerate(ensemble_df["Team"].tolist(), start=1):
            ensemble_rank[team] = i

    bcar_score = {}
    if bcar_table is not None and not bcar_table.empty and "Team" in bcar_table.columns and "BCAR Score" in bcar_table.columns:
        bcar_score = dict(zip(bcar_table["Team"], bcar_table["BCAR Score"]))

    mode_key = (mode or "Ensemble rank").strip()
    if mode_key == "Win%":
        return sorted(base_teams, key=lambda t: (-stats[t].get("win_pct", 0.0), ensemble_rank.get(t, 9999), t))
    if mode_key == "Pyth":
        return sorted(base_teams, key=lambda t: (-(pyth_vals or {}).get(t, 0.0), ensemble_rank.get(t, 9999), t))
    if mode_key == "AdjPyth":
        return sorted(base_teams, key=lambda t: (-(adj_vals or {}).get(t, 0.0), ensemble_rank.get(t, 9999), t))
    if mode_key == "Elo":
        return sorted(base_teams, key=lambda t: (-(elo or {}).get(t, 0.0), ensemble_rank.get(t, 9999), t))
    if mode_key == "BCAR":
        return sorted(base_teams, key=lambda t: (-bcar_score.get(t, float("-inf")), ensemble_rank.get(t, 9999), t), reverse=False)
    if mode_key == "SOS":
        return sorted(base_teams, key=lambda t: (-sos.get(t, 0.0), ensemble_rank.get(t, 9999), t))
    return sorted(base_teams, key=lambda t: (ensemble_rank.get(t, 9999), t))

# ---------------- Sectional Rankings ---------------- #
SECTIONAL_SCORE_PARAMS = {
    "game_penalty_threshold": 0.8,
    "game_penalty_power": 2.0,
    "zero_sectional_penalty": 0.85,
    "sectional_penalty_threshold": 0.7,
    "h2h_weight": 0.45,
    "common_weight": 0.45,
    "win_pct_weight": 0.10,
    "sos_center": 0.5,
    "sos_scale": 2.0,
    "sectional_sos_boost": 1.1,
    "base_common_weight": 0.7,
    "common_weight_scale": 0.6,
    "goal_diff_factor_scale": 0.05,
    "shared_opponent_mode": "pairwise",
    "shared_opponent_min_teams": 2,
    "shared_games_threshold": 3,
    "shared_shrink_k": 4.0,
    "shared_metric": "win_rate",
    "reliability_floor": 0.35,
    "reliability_ceiling": 0.95,
    "reliability_shrink_k": 6.0,
    "global_prior_min_weight": 0.10,
    "global_prior_max_weight": 0.25,
    "global_prior_shrink_k": 4.0,
}

def build_team_opponent_vectors(team, matchup_agg):
    vectors = {}
    for (src, opp), rec in matchup_agg.items():
        if src != team:
            continue
        if rec["games"] <= 0:
            continue
        vectors[opp] = {
            "wins": rec["wins"],
            "losses": rec["losses"],
            "games": rec["games"],
            "gf": rec.get("gf", 0),
            "ga": rec.get("ga", 0),
            "win_rate": (rec["wins"] / rec["games"]) if rec["games"] else 0.5,
            "margin_per_game": ((rec.get("gf", 0) - rec.get("ga", 0)) / rec["games"]) if rec["games"] else 0.0,
        }
    return vectors

def compute_shared_opponent_score(team, valid_teams, team_vectors, stats, p):
    if team not in team_vectors:
        return 0.5, 0, []

    mode = p.get("shared_opponent_mode", "pairwise")
    min_teams = max(int(p.get("shared_opponent_min_teams", 2)), 2)
    metric = p.get("shared_metric", "win_rate")

    opp_coverage = defaultdict(set)
    for sectional_team in valid_teams:
        for opp in team_vectors.get(sectional_team, {}):
            if opp in valid_teams:
                continue
            opp_coverage[opp].add(sectional_team)

    if mode == "sectional":
        shared_pool = {opp for opp, covered in opp_coverage.items() if len(covered) >= min_teams}
    else:
        shared_pool = set()
        for other in valid_teams:
            if other == team:
                continue
            team_opps = set(team_vectors.get(team, {}).keys()) - set(valid_teams)
            other_opps = set(team_vectors.get(other, {}).keys()) - set(valid_teams)
            shared_pool.update(team_opps & other_opps)

    if not shared_pool:
        return 0.5, 0, []

    detail_rows = []
    weighted_metric_sum = 0.0
    total_games = 0
    for opp in sorted(shared_pool):
        rec = team_vectors[team].get(opp)
        if not rec or rec["games"] <= 0 or opp not in stats:
            continue
        games = rec["games"]
        total_games += games
        value = rec["win_rate"] if metric == "win_rate" else rec["margin_per_game"]
        opp_strength = stats[opp]["win_pct"]
        # Discount results against weak shared opponents so teams cannot
        # inflate sectional placement purely by farming low-quality common foes.
        strength_scale = min(max(opp_strength / 0.5, 0.6), 1.4)
        adjusted_value = value * strength_scale if metric == "win_rate" else value
        weighted_metric_sum += (adjusted_value * games)
        detail_rows.append({
            "Opponent": opp,
            "Record": f"{rec['wins']}-{rec['losses']}",
            "Games": games,
            "Win %": rec["win_rate"],
            "Margin/Game": rec["margin_per_game"],
            "Opp Win %": opp_strength,
            "Strength Scale": strength_scale,
            "Included Teams": sorted(opp_coverage.get(opp, [])),
        })

    if total_games == 0:
        return 0.5, 0, detail_rows

    if metric == "win_rate":
        observed_score = weighted_metric_sum / total_games
        neutral = 0.5
    else:
        avg_margin = weighted_metric_sum / total_games
        observed_score = 1.0 / (1.0 + math.exp(-avg_margin))
        neutral = 0.5

    threshold = max(int(p.get("shared_games_threshold", 3)), 1)
    shrink_k = max(float(p.get("shared_shrink_k", 4.0)), 1e-6)
    if total_games < threshold:
        shrink = total_games / (total_games + shrink_k)
        observed_score = (shrink * observed_score) + ((1 - shrink) * neutral)

    return observed_score, total_games, detail_rows


def summarize_common_opponents(team_a, team_b, matchup_agg):
    """Build common-opponent resume details for two teams.

    Returns:
        dict with:
            - shared_opponents: sorted list of shared opponents
            - team_a_record: aggregate W-L vs shared opponents
            - team_b_record: aggregate W-L vs shared opponents
            - normalized_edge_score: in [0, 1], 0.5 is neutral
            - confidence_note: reliability note when overlap is sparse
            - detail_rows: table-ready rows by shared opponent
    """

    def _record_vs(src_team, opp_team):
        nested_rec = matchup_agg.get(src_team, {}).get(opp_team)
        if nested_rec is not None:
            wins = int(nested_rec.get("wins", 0))
            games = int(nested_rec.get("games", 0))
            losses = max(games - wins, 0)
            return {"wins": wins, "losses": losses, "games": games}

        tuple_rec = matchup_agg.get((src_team, opp_team), {"wins": 0, "losses": 0, "games": 0})
        wins = int(tuple_rec.get("wins", 0))
        losses = int(tuple_rec.get("losses", 0))
        games = int(tuple_rec.get("games", wins + losses))
        return {"wins": wins, "losses": losses, "games": games}

    team_a_opponents = {k for k in matchup_agg.get(team_a, {}) if k != team_b}
    team_b_opponents = {k for k in matchup_agg.get(team_b, {}) if k != team_a}
    shared_opponents = sorted(team_a_opponents.intersection(team_b_opponents))

    detail_rows = []
    team_a_tot_wins = team_a_tot_losses = 0
    team_b_tot_wins = team_b_tot_losses = 0
    total_shared_games = 0

    for shared_opp in shared_opponents:
        rec_a = _record_vs(team_a, shared_opp)
        rec_b = _record_vs(team_b, shared_opp)
        team_a_tot_wins += rec_a["wins"]
        team_a_tot_losses += rec_a["losses"]
        team_b_tot_wins += rec_b["wins"]
        team_b_tot_losses += rec_b["losses"]
        total_shared_games += rec_a["games"] + rec_b["games"]
        detail_rows.append(
            {
                "Opponent": shared_opp,
                f"{team_a} record": f"{rec_a['wins']}-{rec_a['losses']}",
                f"{team_b} record": f"{rec_b['wins']}-{rec_b['losses']}",
            }
        )

    team_a_games = team_a_tot_wins + team_a_tot_losses
    team_b_games = team_b_tot_wins + team_b_tot_losses
    team_a_win_rate = (team_a_tot_wins / team_a_games) if team_a_games else 0.5
    team_b_win_rate = (team_b_tot_wins / team_b_games) if team_b_games else 0.5
    normalized_edge_score = 0.5 + ((team_a_win_rate - team_b_win_rate) / 2.0)
    normalized_edge_score = min(max(normalized_edge_score, 0.0), 1.0)

    confidence_note = ""
    if not shared_opponents:
        confidence_note = "No shared-opponent overlap yet."
    elif total_shared_games < 6:
        confidence_note = "Low confidence: very small shared-opponent sample."
    elif total_shared_games < 12:
        confidence_note = "Moderate confidence: shared-opponent sample is still limited."

    return {
        "shared_opponents": shared_opponents,
        "team_a_record": {"wins": team_a_tot_wins, "losses": team_a_tot_losses, "games": team_a_games},
        "team_b_record": {"wins": team_b_tot_wins, "losses": team_b_tot_losses, "games": team_b_games},
        "normalized_edge_score": normalized_edge_score,
        "confidence_note": confidence_note,
        "detail_rows": detail_rows,
    }

def compute_sectional_team_breakdown(team, sectional, stats, h2h, games, sos, matchup_agg, global_prior_scores=None, params=None):
    p = {**SECTIONAL_SCORE_PARAMS, **(params or {})}
    valid_teams = [t for t in sectional if t in stats]
    if team not in stats:
        return {
            "team": team, "sectional": list(sectional), "valid_teams": valid_teams,
            "h2h_score": 0.0, "common_opponent_score": 0.0, "win_pct": 0.0, "combined_score": float("-inf"),
            "penalties": {"game_penalty": 1.0, "sectional_penalty": 1.0},
            "h2h_details": [], "non_sectional_common_details": [], "sectional_common_details": []
        }

    avg_games = (sum(stats[t]["games"] for t in valid_teams) / len(valid_teams)) if valid_teams else 0
    team_games = stats[team]["games"]
    game_penalty = 1.0
    if avg_games > 0 and team_games < avg_games * p["game_penalty_threshold"]:
        game_penalty = (team_games / avg_games) ** p["game_penalty_power"]

    sectional_games = sum(1 for opp in valid_teams if opp != team and h2h.get((team, opp), {"games": 0})["games"] > 0)
    avg_sectional_games = (sum(1 for t in valid_teams for opp in valid_teams if t != opp and h2h.get((t, opp), {"games": 0})["games"] > 0) / len(valid_teams)) if valid_teams else 0
    sectional_penalty = 1.0
    if sectional_games == 0:
        sectional_penalty = p["zero_sectional_penalty"]
    elif avg_sectional_games > 0 and sectional_games < avg_sectional_games * p["sectional_penalty_threshold"]:
        sectional_penalty = p["zero_sectional_penalty"] + ((1 - p["zero_sectional_penalty"]) * (sectional_games / (avg_sectional_games * p["sectional_penalty_threshold"])))

    all_opp_win_pcts = [stats[opp]["win_pct"] for opp in stats if opp != team]
    avg_opp_win_pct = sum(all_opp_win_pcts) / len(all_opp_win_pcts) if all_opp_win_pcts else 0.5

    h2h_scores, h2h_details = [], []
    sectional_wins = 0
    sectional_h2h_games = 0
    for opp in valid_teams:
        if opp == team:
            continue
        r = h2h.get((team, opp), {"wins": 0, "games": 0, "gf": 0, "ga": 0})
        if r["games"] <= 0:
            h2h_scores.append(0.0)
            continue
        sectional_wins += r["wins"]
        sectional_h2h_games += r["games"]
        opp_strength, opp_sos = stats[opp]["win_pct"], sos[opp]
        sos_multiplier = 1.0 + ((opp_sos - p["sos_center"]) * p["sos_scale"])
        sos_multiplier *= p["sectional_sos_boost"]
        adjusted_strength = opp_strength * sos_multiplier
        win_pct = r["wins"] / r["games"]
        goal_diff_factor = 1.0
        if r["wins"] > 0 and r["games"] - r["wins"] > 0:
            goal_diff_factor = 1.0 + ((r.get("gf", 0) - r.get("ga", 0)) * p["goal_diff_factor_scale"])
        weighted_score = win_pct * (adjusted_strength / avg_opp_win_pct) * goal_diff_factor
        h2h_scores.append(weighted_score)
        h2h_details.append({"Opponent": opp, "Record": f"{r['wins']}-{r['games']-r['wins']}", "Win %": win_pct, "Opp Win %": opp_strength, "Opp SOS": opp_sos, "SOS Mult": sos_multiplier, "Adj Strength": adjusted_strength, "GD Factor": goal_diff_factor, "Weighted Score": weighted_score})
    h2h_score = (sum(h2h_scores) / len(h2h_scores)) if h2h_scores else 0.0

    team_vectors = {t: build_team_opponent_vectors(t, matchup_agg) for t in valid_teams}
    common_wins_weighted = 0.0
    common_games = 0
    non_sectional_details, sectional_details = [], []
    common_win_pct, common_games, non_sectional_details = compute_shared_opponent_score(team, valid_teams, team_vectors, stats, p)

    for opp in set(valid_teams):
        if opp == team or opp not in stats:
            continue
        r = h2h.get((team, opp), {"wins": 0, "games": 0, "gf": 0, "ga": 0})
        if r["games"] <= 0:
            continue
        opp_strength, opp_sos = stats[opp]["win_pct"], sos[opp]
        sos_multiplier = (1.0 + ((opp_sos - p["sos_center"]) * p["sos_scale"])) * p["sectional_sos_boost"]
        adjusted_strength = opp_strength * sos_multiplier
        weight = p["base_common_weight"] + (p["common_weight_scale"] * (adjusted_strength / avg_opp_win_pct))
        # Keep sectional matchup detail for transparency only.
        # Sectional matchups are already represented in H2H and must not
        # be double-counted in the common-opponent component.
        sectional_details.append({"Opponent": opp, "Record": f"{r['wins']}-{r['games']-r['wins']}", "Win %": (r["wins"] / r["games"]), "Opp Win %": opp_strength, "Opp SOS": opp_sos, "SOS Mult": sos_multiplier, "Adj Strength": adjusted_strength, "Weight": weight, "Weighted Score": ((r["wins"] / r["games"]) * weight)})

    common_wins_weighted = common_win_pct * common_games
    win_pct = stats[team]["win_pct"]
    global_prior_score = (global_prior_scores or {}).get(team, win_pct)
    prior_sample_reliability = sectional_games / (sectional_games + max(float(p.get("global_prior_shrink_k", 4.0)), 1e-6))
    prior_min_weight = float(p.get("global_prior_min_weight", 0.10))
    prior_max_weight = float(p.get("global_prior_max_weight", 0.25))
    prior_min_weight = min(max(prior_min_weight, 0.0), 1.0)
    prior_max_weight = min(max(prior_max_weight, prior_min_weight), 1.0)
    effective_prior_weight = prior_min_weight + ((prior_max_weight - prior_min_weight) * (1 - prior_sample_reliability))
    local_weight = 1 - effective_prior_weight
    raw_score = local_weight * ((h2h_score * p["h2h_weight"]) + (common_win_pct * p["common_weight"]) + (win_pct * p["win_pct_weight"])) + (effective_prior_weight * global_prior_score)
    fallback_score = global_prior_score

    sectional_h2h_win_pct = (sectional_wins / sectional_h2h_games) if sectional_h2h_games > 0 else 0.35
    # Add a coverage penalty so teams with sparse/no in-sectional games are
    # pulled down instead of receiving a neutral boost.
    avg_sectional_games_per_team = avg_sectional_games if avg_sectional_games > 0 else max(len(valid_teams) - 1, 1)
    sectional_game_coverage = min(sectional_h2h_games / max(avg_sectional_games_per_team, 1.0), 1.0)
    coverage_floor = 0.55
    coverage_multiplier = coverage_floor + ((1.0 - coverage_floor) * sectional_game_coverage)

    # Apply a firm guardrail from in-sectional results so teams with weak
    # direct sectional performance cannot float too high on broad-season record.
    results_multiplier = min(max(0.50 + (0.90 * sectional_h2h_win_pct), 0.50), 1.40)
    sectional_results_factor = results_multiplier * coverage_multiplier
    raw_score *= sectional_results_factor
    fallback_score *= sectional_results_factor

    penalty_reliability = game_penalty * sectional_penalty
    shrink_k = max(float(p.get("reliability_shrink_k", 6.0)), 1e-6)
    game_coverage = (team_games / (team_games + shrink_k)) if team_games >= 0 else 0.0
    sectional_coverage = (sectional_games / (sectional_games + shrink_k)) if sectional_games >= 0 else 0.0
    coverage_reliability = game_coverage * sectional_coverage

    floor = float(p.get("reliability_floor", 0.35))
    ceiling = float(p.get("reliability_ceiling", 0.95))
    floor = min(max(floor, 0.0), 1.0)
    ceiling = min(max(ceiling, floor), 1.0)
    reliability = floor + ((ceiling - floor) * (penalty_reliability * coverage_reliability))

    combined_score = (reliability * raw_score) + ((1 - reliability) * fallback_score)

    return {
        "team": team, "sectional": list(sectional), "valid_teams": valid_teams,
        "h2h_score": h2h_score, "common_opponent_score": common_win_pct, "win_pct": win_pct,
        "global_prior_score": global_prior_score, "global_prior_weight": effective_prior_weight,
        "combined_score": combined_score, "common_wins_weighted": common_wins_weighted, "common_games": common_games,
        "reliability": reliability, "raw_score": raw_score, "fallback_score": fallback_score,
        "sectional_h2h_win_pct": sectional_h2h_win_pct, "sectional_results_factor": sectional_results_factor,
        "sectional_game_coverage": sectional_game_coverage,
        "penalties": {"game_penalty": game_penalty, "sectional_penalty": sectional_penalty},
        "h2h_details": h2h_details, "non_sectional_common_details": non_sectional_details, "sectional_common_details": sectional_details,
    }

def compute_sectional_rankings(stats, h2h, games_inferred, sos, matchup_agg, global_prior_scores=None, sectional_params=None):
    sectionals = {
        "Hoffman Estates (H.S.)": ["Hersey", "Barrington", "Elk Grove", "Conant", "Hoffman Estates", "McHenry", "Fremd", "Palatine", "Meadows", "Schaumburg"],
        "Chicago (Lane)": ["Amundsen", "De La Salle", "Jones-Payton", "Juarez", "Lane", "Latin", "Senn", "St Ignatius", "Whitney Young"],
        "Oak Park (Fenwick)": ["Morton", "Northside", "St Patrick", "Taft", "Westinghouse", "York", "Leyden", "Fenwick", "Oak Park", "STC"],
        "Glenview (GBS)": ["Maine West", "Evanston", "GBS", "Prospect", "GBN", "Maine East", "Maine South", "Niles West", "Loyola", "New Trier"],
        "LaGrange (Lyons)": ["Morton", "Curie", "Kennedy", "Mt Carmel", "Solorio", "St Rita", "Lyons", "R-B", "Argo"],
        "Hinsdale (Central)": ["Metea", "Waubonsie", "Hinsdale South", "HC", "Lockport", "NC", "Neuqua", "NN", "Sandburg"],
        "New Lenox (LWC)": ["Bradley", "Chicago Ag", "Brother Rice", "H-F", "LWE", "Bremen", "LWC", "LWW", "Shepard", "Andrew"],
        "Buffalo Grove": ["BG", "Deerfield", "Warren", "Highland Park", "Lake Forest", "Libertyville", "Stevenson", "Mundelein", "VH"]
    }
    
    sectional_breakdowns = {}
    def rank_teams_in_sectional(teams, sectional_name):
        team_breakdowns = {
            team: compute_sectional_team_breakdown(team, teams, stats, h2h, games_inferred, sos, matchup_agg, global_prior_scores=global_prior_scores, params=sectional_params)
            for team in teams
        }
        for team, breakdown in team_breakdowns.items():
            if team in stats:
                stats[team].setdefault("sectional_score", {})[sectional_name] = breakdown["combined_score"]
        sectional_breakdowns[sectional_name] = team_breakdowns
        return sorted(teams, key=lambda t: team_breakdowns[t]["combined_score"], reverse=True)
    
    # Rank teams in each sectional
    sectional_rankings = {name: rank_teams_in_sectional(teams, name) for name, teams in sectionals.items()}
    
    # Calculate sectional strength
    def get_sectional_strength(teams):
        valid_teams = [t for t in teams if t in stats]
        if not valid_teams:
            return 0
        return sum(stats[t]['win_pct'] for t in valid_teams) / len(valid_teams)
    
    sectional_strengths = {name: get_sectional_strength(teams) for name, teams in sectionals.items()}
    sectional_order = sorted(sectional_strengths.keys(), key=lambda x: sectional_strengths[x], reverse=True)
    
    return sectional_rankings, sectional_order, sectional_breakdowns

# ---------------- App ---------------- #

@st.cache_data
def load_model_config():
    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["ensemble_weights"] = sanitize_ensemble_weights(config.get("ensemble_weights", {}))
    return config

def build_confidence_badge(team, stats, h2h, team_imputation, all_teams):
    games = stats[team]["games"]
    max_games = max((stats[t]["games"] for t in all_teams), default=1)
    games_ratio = games / max_games if max_games else 0
    covered = sum(1 for opp in all_teams if opp != team and h2h.get((team, opp), {"games": 0})["games"] > 0)
    coverage_ratio = covered / max(len(all_teams) - 1, 1)
    imp_ratio = (team_imputation[team]["imputed"] / team_imputation[team]["games"]) if team_imputation[team]["games"] else 0
    score = (0.45 * games_ratio) + (0.35 * coverage_ratio) + (0.20 * (1 - imp_ratio))
    if score >= 0.75:
        return "High", score, games_ratio, coverage_ratio, imp_ratio
    if score >= 0.5:
        return "Medium", score, games_ratio, coverage_ratio, imp_ratio
    return "Low", score, games_ratio, coverage_ratio, imp_ratio

def confidence_tier_label(confidence_score):
    if confidence_score >= 75:
        return "High"
    if confidence_score >= 50:
        return "Moderate"
    return "Low"




def confidence_tier_badge(tier):
    palette = {
        "High": "🟢 High",
        "Moderate": "🟡 Moderate",
        "Low": "🔴 Low",
    }
    return palette.get(str(tier), str(tier))


def format_confidence_columns(df):
    formatted = df.copy()
    for col in ["Games Confidence", "SOS Confidence", "Composite Confidence", "Confidence"]:
        if col in formatted.columns:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce").round(1)
    if "Confidence Tier" in formatted.columns:
        formatted["Confidence Tier"] = formatted["Confidence Tier"].map(confidence_tier_badge)
    return formatted

def compute_ensemble_confidence(team, stats, h2h, sos, team_imputation, all_teams):
    _, _, games_ratio, _, imp_ratio = build_confidence_badge(team, stats, h2h, team_imputation, all_teams)
    max_sos = max((sos.get(t, 0.0) for t in all_teams), default=1.0)
    min_sos = min((sos.get(t, 0.0) for t in all_teams), default=0.0)
    sos_range = max(max_sos - min_sos, 1e-9)
    sos_norm = (sos.get(team, 0.0) - min_sos) / sos_range
    _, _, unique_opp_ratio, _, _, _ = compute_resume_breadth_factor(
        team, stats, h2h, all_teams, team_imputation
    )

    games_confidence = max(0.0, min(100.0, games_ratio * 100.0))
    sos_confidence = max(0.0, min(100.0, (0.55 * sos_norm + 0.45 * unique_opp_ratio) * 100.0))
    reliability_component = max(0.0, min(1.0, 1.0 - imp_ratio))
    composite_confidence = max(
        0.0,
        min(100.0, (0.45 * (games_confidence / 100.0) + 0.35 * (sos_confidence / 100.0) + 0.20 * reliability_component) * 100.0),
    )
    return games_confidence, sos_confidence, composite_confidence


def compute_resume_breadth_factor(team, stats, h2h, all_teams, team_imputation, breadth_cfg=None):
    cfg = breadth_cfg or {}
    min_damping = float(cfg.get("min_damping", 0.85))
    max_damping = float(cfg.get("max_damping", 1.00))
    opp_weight = float(cfg.get("opponent_weight", 0.45))
    games_weight = float(cfg.get("games_weight", 0.40))
    imputation_weight = float(cfg.get("imputation_weight", 0.15))
    threshold = float(cfg.get("breadth_threshold", 0.70))
    threshold_steepness = float(cfg.get("threshold_steepness", 1.0))

    games = stats[team]["games"]
    max_games = max((stats[t]["games"] for t in all_teams), default=1)
    games_ratio = games / max_games if max_games else 0.0
    unique_opp_count = sum(1 for opp in all_teams if opp != team and h2h.get((team, opp), {"games": 0})["games"] > 0)
    max_unique_opponents = max(len(all_teams) - 1, 1)
    unique_opp_ratio = unique_opp_count / max_unique_opponents
    imp_ratio = (team_imputation[team]["imputed"] / team_imputation[team]["games"]) if team_imputation[team]["games"] else 0.0

    raw_breadth = (opp_weight * unique_opp_ratio) + (games_weight * games_ratio) + (imputation_weight * (1 - imp_ratio))
    if threshold_steepness <= 0:
        threshold_steepness = 1.0
    threshold_scaled = max(0.0, min(1.0, raw_breadth / threshold)) ** threshold_steepness if threshold > 0 else raw_breadth
    breadth_score = max(0.0, min(1.0, threshold_scaled))
    damping = min_damping + (max_damping - min_damping) * breadth_score
    damping = max(min_damping, min(max_damping, damping))
    return damping, unique_opp_count, unique_opp_ratio, games_ratio, imp_ratio, raw_breadth

def build_rank_diff(previous_orders, current_orders):
    rows = []
    for model, current in current_orders.items():
        previous = previous_orders.get(model, [])
        prev_idx = {t: i + 1 for i, t in enumerate(previous)}
        curr_idx = {t: i + 1 for i, t in enumerate(current)}
        for team, curr_rank in curr_idx.items():
            old_rank = prev_idx.get(team)
            delta = None if old_rank is None else (old_rank - curr_rank)
            rows.append({
                "Model": model,
                "Team": team,
                "Prior Rank": old_rank if old_rank is not None else "New",
                "Current Rank": curr_rank,
                "Δ Rank": delta if delta is not None else "—"
            })
    return pd.DataFrame(rows)

def build_why_rank_rows(order, metric_values, stats, h2h, sos, team_imputation):
    rows = []
    for team in order:
        rank = order.index(team) + 1
        top_h2h = max((h2h.get((team, opp), {"wins":0,"games":0}) for opp in stats if opp != team), key=lambda x: (x.get("wins",0), x.get("games",0)), default={"wins":0,"games":0})
        total_games = top_h2h.get("games", 0)
        h2h_delta = (top_h2h.get("wins", 0) / total_games - 0.5) if total_games else 0
        imp_ratio = (team_imputation[team]["imputed"] / team_imputation[team]["games"]) if team_imputation[team]["games"] else 0
        penalties = []
        if stats[team]["games"] < max(1, max(st["games"] for st in stats.values()) * 0.7):
            penalties.append("low games")
        if imp_ratio > 0.3:
            penalties.append("high imputation")
        rows.append({
            "Rank": rank,
            "Team": team,
            "Metric": round(metric_values.get(team, 0), 3),
            "Record": f"{stats[team]['wins']}-{stats[team]['losses']}-{stats[team]['ties']}",
            "H2H Delta": f"{h2h_delta:+.3f}",
            "SOS Effect": round(sos[team]-0.5, 3),
            "Penalties": ", ".join(penalties) if penalties else "None",
            "Imputation Rate": f"{imp_ratio:.1%}"
        })
    return pd.DataFrame(rows)

def rank_percentile_map(order):
    n = len(order)
    if n <= 1:
        return {team: 1.0 for team in order}
    return {team: (n - idx - 1) / (n - 1) for idx, team in enumerate(order)}

def compute_rank_tie_break_key(team, stats, sos, h2h):
    direct_h2h_win_pct = max(
        (
            (record["wins"] / record["games"])
            for (me, _), record in h2h.items()
            if me == team and record.get("games", 0) > 0
        ),
        default=0.0
    )
    sos_adjusted_margin = ((stats[team]["gf"] - stats[team]["ga"]) / max(stats[team]["games"], 1)) * sos.get(team, 0)
    stable_secondary = stats[team]["win_pct"]
    return (direct_h2h_win_pct, sos_adjusted_margin, stable_secondary)

def _build_expert_nudge_lookup(teams, expert_nudge_cfg):
    cfg = expert_nudge_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    max_abs = max(0.0, float(cfg.get("max_abs", 0.02)))
    max_rank_shift = max(0, int(cfg.get("max_rank_shift", 2)))
    raw_adjustments = cfg.get("team_adjustments", {}) or {}
    if not isinstance(raw_adjustments, dict):
        raw_adjustments = {}
    team_set = set(teams)
    normalized_team_index = {_normalize_team_name(team).lower(): team for team in teams}
    applied = {}
    for team_key, raw_value in raw_adjustments.items():
        if team_key is None:
            continue
        normalized_key = _normalize_team_name(str(team_key)).lower()
        resolved_team = normalized_team_index.get(normalized_key)
        if not resolved_team or resolved_team not in team_set:
            continue
        try:
            bounded = max(-max_abs, min(max_abs, float(raw_value)))
        except (TypeError, ValueError):
            continue
        if abs(bounded) > 0:
            applied[resolved_team] = bounded
    return enabled, max_abs, max_rank_shift, applied

def build_calibrated_ensemble(teams, orders, stats, h2h, sos, team_imputation, ensemble_base_weights=None, win_model_cap=None, ensemble_breadth_cfg=None, expert_nudge_cfg=None):
    base_weights = sanitize_ensemble_weights(ensemble_base_weights or DEFAULT_ENSEMBLE_WEIGHTS)
    model_pct = {name: rank_percentile_map(order) for name, order in orders.items()}
    rank_lookup = {name: {t: i + 1 for i, t in enumerate(order)} for name, order in orders.items()}
    rows = []
    for team in teams:
        _, confidence, games_ratio, coverage_ratio, imp_ratio = build_confidence_badge(team, stats, h2h, team_imputation, teams)
        games_confidence, sos_confidence, composite_confidence = compute_ensemble_confidence(
            team, stats, h2h, sos, team_imputation, teams
        )
        breadth_damping, unique_opp_count, unique_opp_ratio, _, _, breadth_raw_score = compute_resume_breadth_factor(
            team, stats, h2h, teams, team_imputation, ensemble_breadth_cfg
        )
        reliability = max(0.0, confidence)
        reliability_modulators = {
            "Win": 0.50 + 0.50 * coverage_ratio,
            "Pyth": 0.50 + 0.50 * games_ratio,
            "AdjPyth": 0.65 + 0.35 * (1 - imp_ratio),
            "BCAR": 0.60 + 0.40 * (1 - imp_ratio),
            "Elo": 0.50 + 0.50 * coverage_ratio,
        }
        weights = {
            model: base_weights[model] * reliability * reliability_modulators[model]
            for model in base_weights
        }
        win_cap_cfg = win_model_cap or {}
        win_cap_max = float(win_cap_cfg.get("max_multiplier", 0.85))
        win_cov_floor = float(win_cap_cfg.get("coverage_floor", 0.65))
        win_imp_ceiling = float(win_cap_cfg.get("imputation_ceiling", 0.30))
        cov_cap_ratio = min(1.0, coverage_ratio / max(win_cov_floor, 1e-9)) if win_cov_floor > 0 else 1.0
        imp_cap_ratio = min(1.0, max(0.0, (1.0 - imp_ratio) / max(1.0 - win_imp_ceiling, 1e-9))) if win_imp_ceiling < 1.0 else 1.0
        win_cap_multiplier = min(win_cap_max, cov_cap_ratio, imp_cap_ratio)
        weights["Win"] *= max(0.0, min(1.0, win_cap_multiplier))
        total_weight = sum(weights.values())
        normalized_weights = {
            model: (weights[model] / total_weight) if total_weight else 0.0
            for model in weights
        }
        weighted_sum = (
            normalized_weights["Win"] * model_pct["Win"].get(team, 0.0)
            + normalized_weights["Pyth"] * model_pct["Pyth"].get(team, 0.0) * breadth_damping
            + normalized_weights["AdjPyth"] * model_pct["AdjPyth"].get(team, 0.0) * breadth_damping
            + normalized_weights["BCAR"] * model_pct["BCAR"].get(team, 0.0)
            + normalized_weights["Elo"] * model_pct["Elo"].get(team, 0.0)
        )
        calibrated_score = weighted_sum if total_weight else 0.0
        low_confidence_floor = 35.0
        max_low_conf_damping = 0.03
        if composite_confidence < low_confidence_floor:
            damping_strength = (low_confidence_floor - composite_confidence) / low_confidence_floor
            calibrated_score *= max(0.0, 1.0 - (max_low_conf_damping * damping_strength))
        ordinal_ranks = [
            rank_lookup["Win"].get(team, len(orders["Win"]) + 1),
            rank_lookup["Pyth"].get(team, len(orders["Pyth"]) + 1),
            rank_lookup["AdjPyth"].get(team, len(orders["AdjPyth"]) + 1),
            rank_lookup["BCAR"].get(team, len(orders["BCAR"]) + 1),
            rank_lookup["Elo"].get(team, len(orders["Elo"]) + 1),
        ]
        rows.append({
            "Team": team,
            "Calibrated Score": calibrated_score,
            "Games Confidence": games_confidence,
            "SOS Confidence": sos_confidence,
            "Composite Confidence": composite_confidence,
            "Confidence Tier": confidence_tier_label(composite_confidence),
            "SOS": sos.get(team, 0.0),
            "Direct H2H Tiebreak": compute_rank_tie_break_key(team, stats, sos, h2h)[0],
            "SOS Margin Tiebreak": compute_rank_tie_break_key(team, stats, sos, h2h)[1],
            "Stable Secondary": compute_rank_tie_break_key(team, stats, sos, h2h)[2],
            "Ordinal Avg (Debug)": round(sum(ordinal_ranks) / len(ordinal_ranks), 2),
            "Win Rank": ordinal_ranks[0],
            "Pyth Rank": ordinal_ranks[1],
            "AdjPyth Rank": ordinal_ranks[2],
            "BCAR Rank": ordinal_ranks[3],
            "Elo Rank": ordinal_ranks[4],
            "Win %tile": model_pct["Win"].get(team, 0.0),
            "Pyth %tile": model_pct["Pyth"].get(team, 0.0),
            "AdjPyth %tile": model_pct["AdjPyth"].get(team, 0.0),
            "BCAR %tile": model_pct["BCAR"].get(team, 0.0),
            "Elo %tile": model_pct["Elo"].get(team, 0.0),
            "Weight Win": weights["Win"],
            "Weight Pyth": weights["Pyth"],
            "Weight AdjPyth": weights["AdjPyth"],
            "Weight BCAR": weights["BCAR"],
            "Weight Elo": weights["Elo"],
            "Norm Weight Win": normalized_weights["Win"],
            "Norm Weight Pyth": normalized_weights["Pyth"],
            "Norm Weight AdjPyth": normalized_weights["AdjPyth"],
            "Norm Weight BCAR": normalized_weights["BCAR"],
            "Norm Weight Elo": normalized_weights["Elo"],
            "Games Ratio": games_ratio,
            "Coverage Ratio": coverage_ratio,
            "Imputation Rate": imp_ratio,
            "Resume Breadth Damping": breadth_damping,
            "Unique Opponents": unique_opp_count,
            "Unique Opponent Ratio": unique_opp_ratio,
            "Breadth Raw Score": breadth_raw_score,
        })
    df = pd.DataFrame(rows)
    enabled_nudge, _, max_rank_shift, nudge_lookup = _build_expert_nudge_lookup(teams, expert_nudge_cfg)
    df["Nudge Applied"] = df["Team"].map(lambda t: nudge_lookup.get(t, 0.0))
    if enabled_nudge:
        base_sorted = df.sort_values(
            by=["Calibrated Score", "Direct H2H Tiebreak", "SOS Margin Tiebreak", "Stable Secondary"],
            ascending=[False, False, False, False],
            kind="mergesort"
        ).reset_index(drop=True)
        base_scores = base_sorted["Calibrated Score"].tolist()
        min_score = min(base_scores) if base_scores else 0.0
        max_score = max(base_scores) if base_scores else 0.0
        rank_index = {team: idx for idx, team in enumerate(base_sorted["Team"])}
        adjusted_scores = {}
        for team, score, nudge in zip(df["Team"], df["Calibrated Score"], df["Nudge Applied"]):
            idx = rank_index.get(team, 0)
            desired = score + nudge
            if max_rank_shift > 0 and base_scores:
                upper_idx = max(0, idx - max_rank_shift)
                lower_idx = min(len(base_scores) - 1, idx + max_rank_shift)
                upper_bound = base_scores[upper_idx] + 1e-9
                lower_bound = base_scores[lower_idx] - 1e-9
                desired = max(lower_bound, min(upper_bound, desired))
            desired = max(min_score - 1e-9, min(max_score + 1e-9, desired))
            adjusted_scores[team] = desired
        df["Calibrated Score"] = df["Team"].map(adjusted_scores)
    else:
        df = df.drop(columns=["Nudge Applied"])
    df = df.sort_values(
        by=["Calibrated Score", "Direct H2H Tiebreak", "SOS Margin Tiebreak", "Stable Secondary"],
        ascending=[False, False, False, False],
        kind="mergesort"
    ).reset_index(drop=True)
    df["Rank"] = df.index + 1
    ordered = ["Rank"] + [c for c in df.columns if c != "Rank"]
    return df[ordered]

def build_primary_ranking_payload(teams, model_orders, stats, h2h, sos, team_imputation, ensemble_base_weights=None, win_model_cap=None, ensemble_breadth_cfg=None, expert_nudge_cfg=None):
    eligible_teams = [t for t in teams if t in model_orders["Win"] and t in model_orders["Pyth"] and t in model_orders["AdjPyth"] and t in model_orders["BCAR"] and t in model_orders["Elo"]]
    ensemble_df = build_calibrated_ensemble(
        eligible_teams, model_orders, stats, h2h, sos, team_imputation,
        ensemble_base_weights=ensemble_base_weights,
        win_model_cap=win_model_cap,
        ensemble_breadth_cfg=ensemble_breadth_cfg,
        expert_nudge_cfg=expert_nudge_cfg
    )
    ordered_teams = ensemble_df["Team"].tolist()
    return {
        "ordered_teams": ordered_teams,
        "rank_lookup": {team: i + 1 for i, team in enumerate(ordered_teams)},
        "score_lookup": dict(zip(ensemble_df["Team"], ensemble_df["Calibrated Score"])),
        "confidence_lookup": {t: build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in ordered_teams},
        "component_diagnostics": {
            row["Team"]: {
                "elo": row["Norm Weight Elo"] * row["Elo %tile"],
                "bcar": row["Norm Weight BCAR"] * row["BCAR %tile"],
                "adj_pyth": row["Norm Weight AdjPyth"] * row["AdjPyth %tile"],
                "pyth": row["Norm Weight Pyth"] * row["Pyth %tile"],
                "win": row["Norm Weight Win"] * row["Win %tile"],
            } for _, row in ensemble_df.iterrows()
        },
        "table": ensemble_df,
    }


def get_primary_nav_options(is_admin_user):
    nav_options = ["Rankings", "Team Profiles/Resume", "Sectionals"]
    if is_admin_user:
        nav_options.append("Admin / Internal")
    return nav_options


def resolve_legacy_public_target(query_params, fallback_team=None):
    legacy_value = ""
    for key in ("tab", "section", "content_section", "primary_nav", "nav"):
        raw = query_params.get(key)
        if raw is None:
            continue
        legacy_value = str(raw).strip()
        if legacy_value:
            break
    if not legacy_value:
        return None

    normalized = legacy_value.lower().replace("_", " ").strip()
    if normalized in {"profile", "team profile", "teams"}:
        return {"target_nav": "Team Profiles/Resume", "team": fallback_team}
    if normalized in {"matchup insights", "matchups", "matchup"}:
        return {"target_nav": "Sectionals", "team": fallback_team}
    if normalized in {"team profiles/resume", "team profile/resume"}:
        return {"target_nav": "Team Profiles/Resume", "team": fallback_team}
    if normalized in {"sectionals"}:
        return {"target_nav": "Sectionals", "team": fallback_team}
    if normalized.startswith("profile/") or normalized.startswith("team/"):
        team_slug = normalized.split("/", 1)[1].strip()
        return {"target_nav": "Team Profiles/Resume", "team_slug": team_slug or None}
    return None


def slugify_team_name(team_name):
    normalized = re.sub(r"[^a-z0-9]+", "-", str(team_name).strip().lower()).strip("-")
    return normalized or "team"


def build_team_slug_lookup(teams):
    slug_lookup = {}
    reverse_lookup = {}
    for team in sorted(teams):
        base_slug = slugify_team_name(team)
        slug = base_slug
        suffix = 2
        while slug in reverse_lookup and reverse_lookup[slug] != team:
            slug = f"{base_slug}-{suffix}"
            suffix += 1
        slug_lookup[team] = slug
        reverse_lookup[slug] = team
    return slug_lookup


def build_team_canonical_path(team_slug, section="profile", timeframe=None):
    section_slug = str(section or "profile").strip().lower().replace(" ", "-")
    path = f"/teams/{team_slug}/{section_slug}"
    if timeframe:
        path = f"{path}?timeframe={timeframe}"
    return path


def build_profile_share_metadata(team, rank, key_signal, summary, canonical_url):
    return {
        "title": f"{team} Polo Profile · Rank #{rank}",
        "summary": summary,
        "key_rank_signal": key_signal,
        "canonical_url": canonical_url,
        "share_text": f"{team} is ranked #{rank}. Key signal: {key_signal}. {canonical_url}",
    }


def main():
    st.set_page_config(page_title="Polo Dashboard", layout="wide")
    is_deep_dive = str(st.query_params.get("debug_sectionals", "0")).strip().lower() in {"1", "true", "yes", "y"}

    is_admin = _is_admin_user()
    nav_options = get_primary_nav_options(is_admin)
    nav_labels = {
        "Rankings": "Rankings",
        "Team Profiles/Resume": "Team Profiles/Resume",
        "Sectionals": "Sectionals",
        "Admin / Internal": "Admin",
    }
    if "primary_nav" not in st.session_state or st.session_state["primary_nav"] not in nav_options:
        st.session_state["primary_nav"] = "Rankings"

    nav_cols = st.columns(len(nav_options), gap="small")
    for col, nav_key in zip(nav_cols, nav_options):
        is_active = st.session_state["primary_nav"] == nav_key
        button_kind = "primary" if is_active else "secondary"
        if col.button(nav_labels[nav_key], key=f"primary_nav_btn_{nav_key}", use_container_width=True, type=button_kind):
            st.session_state["primary_nav"] = nav_key

    current_nav = st.session_state["primary_nav"]
    config = load_model_config()
    ensemble_weights_cfg = sanitize_ensemble_weights(config.get("ensemble_weights", {}))
    win_model_cap_cfg = config.get("win_model_cap", {
        "max_multiplier": 0.85,
        "coverage_floor": 0.65,
        "imputation_ceiling": 0.30,
    })
    ensemble_breadth_cfg = config.get("ensemble_breadth_damping", {
        "min_damping": 0.85,
        "max_damping": 1.00,
        "opponent_weight": 0.45,
        "games_weight": 0.40,
        "imputation_weight": 0.15,
        "breadth_threshold": 0.70,
        "threshold_steepness": 1.0
    })
    expert_nudge_cfg = config.get("expert_nudge", {
        "enabled": False,
        "max_abs": 0.02,
        "max_rank_shift": 2,
        "team_adjustments": {}
    })
    hybrid_cfg = config.get("hybrid", {})

    # Sidebar settings
    st.sidebar.header("Data & Model Settings")
    with st.sidebar.expander("Data Refresh", expanded=True):
        st.caption("Rankings are built directly from *_scores_illpolo.txt files in the repo.")
        if st.button("Refresh from score files now", use_container_width=True):
            clear_score_pipeline_cache()
            st.rerun()

    raw_games, qa_meta = load_games_pipeline(DATA_DIR)
    prior_games = raw_games.copy()
    rebuilt_ts = pd.to_datetime(qa_meta.get("rebuilt_at"), utc=True, errors="coerce")
    rebuilt_label = rebuilt_ts.strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(rebuilt_ts) else "unknown"
    st.sidebar.markdown(f"**Freshness:** last rebuilt from files: `{rebuilt_label}`")
    with st.sidebar.expander("Ingestion QA Summary", expanded=False):
            st.markdown(f"- Files loaded: `{qa_meta.get('files_loaded', 0)}`")
            st.markdown(f"- Games parsed: `{qa_meta.get('games_parsed', 0)}`")
            st.markdown(f"- Duplicates removed: `{qa_meta.get('duplicates_dropped', 0)}`")
            st.markdown(f"- Suspicious lines count: `{qa_meta.get('suspicious_unparsed', 0)}`")
            st.markdown(f"- Skipped lines: `{qa_meta.get('skipped', 0)}`")
            unresolved = qa_meta.get("unresolved_suspicious_lines", [])
            if qa_meta.get("suspicious_unparsed", 0) > 0:
                st.warning("Some lines could not be interpreted and were excluded.")
                with st.expander("Show suspicious raw lines and formatting examples", expanded=False):
                    if unresolved:
                        st.caption("Raw lines we could not parse (first 25):")
                        st.code("\n".join(unresolved), language="text")
                    st.caption("Formatting examples that are recognized:")
                    st.code(
                        "\n".join(
                            [
                                "Team A 12 Team B 9",
                                "Team A 8 Team B 8 (OT)",
                                "Team A d. Team B",
                            ]
                        ),
                        language="text",
                    )
            else:
                st.caption("No unresolved suspicious lines detected.")
            per_file_reports = qa_meta.get("per_file_reports", [])
            if per_file_reports:
                st.caption("Per-file parse report:")
                report_df = pd.DataFrame(per_file_reports)[
                    ["file_name", "lines_scanned", "games_parsed", "skipped", "suspicious_unparsed"]
                ]
                st.dataframe(report_df, use_container_width=True, hide_index=True)
                flagged = [r for r in per_file_reports if r.get("suspicious_unparsed", 0) > 0]
                if flagged:
                    with st.expander("Files with suspicious lines", expanded=False):
                        for row in flagged:
                            st.markdown(
                                f"**{row['file_name']}** — suspicious lines: `{row['suspicious_unparsed']}`"
                            )
                            if row.get("suspicious_examples"):
                                st.code("\n".join(row["suspicious_examples"]), language="text")

    logistic_cfg = config["logistic"]
    elo_cfg = config["elo"]
    pythag_cfg = config["pythag"]
    game_count_cfg = config["game_count"]
    sectional_cfg = {**SECTIONAL_SCORE_PARAMS, **config["sectional"]}
    is_admin = _is_admin_user()

    enable_overrides = False
    k = int(logistic_cfg["k"])
    x0 = float(logistic_cfg["x0"])
    elo_k = int(elo_cfg.get("k", 22))
    phase_k_enabled = bool(elo_cfg.get("phase_k_enabled", False))
    early_phase_games = int(elo_cfg.get("early_phase_games", 40))
    early_phase_multiplier = float(elo_cfg.get("early_phase_multiplier", 1.15))
    late_phase_multiplier = float(elo_cfg.get("late_phase_multiplier", 0.9))
    pythag_exp = float(pythag_cfg["exponent"])
    include_estimated_scores = True
    highlight_estimated_games = True
    down_weight_imputed = False
    imputed_weight = 0.5
    min_games_ratio = float(game_count_cfg["min_games_ratio"])
    ensemble_weight_elo = float(ensemble_weights_cfg["Elo"])
    ensemble_weight_bcar = float(ensemble_weights_cfg["BCAR"])
    ensemble_weight_adjpyth = float(ensemble_weights_cfg["AdjPyth"])
    ensemble_weight_pyth = float(ensemble_weights_cfg["Pyth"])
    ensemble_weight_win = float(ensemble_weights_cfg["Win"])
    h2h_weight = float(sectional_cfg["h2h_weight"])
    common_weight = float(sectional_cfg["common_weight"])
    win_pct_weight = float(sectional_cfg["win_pct_weight"])
    sos_center = float(sectional_cfg["sos_center"])
    sos_scale = float(sectional_cfg["sos_scale"])
    sectional_sos_boost = float(sectional_cfg["sectional_sos_boost"])
    game_penalty_threshold = float(sectional_cfg["game_penalty_threshold"])
    game_penalty_power = float(sectional_cfg["game_penalty_power"])
    sectional_penalty_threshold = float(sectional_cfg["sectional_penalty_threshold"])
    reliability_floor = float(sectional_cfg["reliability_floor"])
    reliability_ceiling = float(sectional_cfg["reliability_ceiling"])
    reliability_shrink_k = float(sectional_cfg["reliability_shrink_k"])
    global_prior_min_weight = float(sectional_cfg["global_prior_min_weight"])
    global_prior_max_weight = float(sectional_cfg["global_prior_max_weight"])
    global_prior_shrink_k = float(sectional_cfg["global_prior_shrink_k"])

    with st.sidebar.expander("Advanced Settings", expanded=False):
            st.caption(f"Role gate: {'Admin' if is_admin else 'Read-only'}")
            if not is_admin:
                st.markdown(
                    " ".join(
                        [
                            f"`Elo {ensemble_weight_elo:.2f}`",
                            f"`BCAR {ensemble_weight_bcar:.2f}`",
                            f"`AdjPyth {ensemble_weight_adjpyth:.2f}`",
                            f"`Pyth {ensemble_weight_pyth:.2f}`",
                            f"`Win% {ensemble_weight_win:.2f}`",
                        ]
                    )
                )
                st.markdown(
                    " ".join(
                        [
                            f"`H2H {h2h_weight:.2f}`",
                            f"`Common {common_weight:.2f}`",
                            f"`Win% {win_pct_weight:.2f}`",
                            f"`SOS boost {sectional_sos_boost:.2f}`",
                        ]
                    )
                )
                st.caption("Admin controls are hidden. Set query param `is_admin=1` or enable `ui_admin_enabled` in secrets.")
            if is_admin:
                enable_overrides = st.checkbox("Enable UI overrides", value=False)
                k = st.slider("Logistic Steepness (k)", min_value=1, max_value=20, value=int(logistic_cfg["k"]), disabled=not enable_overrides)
                x0 = st.slider("Logistic Midpoint (x0)", min_value=0.0, max_value=1.0, value=float(logistic_cfg["x0"]), step=0.05, disabled=not enable_overrides)
                elo_k = st.slider("Elo K", min_value=1, max_value=64, value=int(elo_cfg.get("k", 22)), disabled=not enable_overrides)
                phase_k_enabled = st.toggle("Enable phase-based Elo K", value=bool(elo_cfg.get("phase_k_enabled", False)), disabled=not enable_overrides)
                early_phase_games = st.slider("Early-phase game count", min_value=0, max_value=200, value=int(elo_cfg.get("early_phase_games", 40)), disabled=(not enable_overrides or not phase_k_enabled))
                early_phase_multiplier = st.slider("Early-phase K multiplier", min_value=0.5, max_value=2.0, value=float(elo_cfg.get("early_phase_multiplier", 1.15)), step=0.05, disabled=(not enable_overrides or not phase_k_enabled))
                late_phase_multiplier = st.slider("Late-phase K multiplier", min_value=0.5, max_value=2.0, value=float(elo_cfg.get("late_phase_multiplier", 0.9)), step=0.05, disabled=(not enable_overrides or not phase_k_enabled))
                pythag_exp = st.slider("Pythagorean Exponent", min_value=1.0, max_value=5.0, value=float(pythag_cfg["exponent"]), step=0.1, disabled=not enable_overrides)
                include_estimated_scores = st.toggle("Include estimated scores", value=True)
                highlight_estimated_games = st.toggle("Highlight estimated games", value=True)
                down_weight_imputed = st.toggle("Down-weight inferred games", value=False)
                imputed_weight = st.slider("Inferred game weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=(not down_weight_imputed))
                min_games_ratio = st.slider("Min Games Ratio", min_value=0.1, max_value=1.0, value=float(game_count_cfg["min_games_ratio"]), step=0.05, disabled=not enable_overrides)

                st.markdown("**Ensemble Weights**")
                ensemble_weight_elo = st.slider("Ensemble Elo Weight", min_value=0.0, max_value=1.0, value=float(ensemble_weights_cfg["Elo"]), step=0.01, disabled=not enable_overrides)
                ensemble_weight_bcar = st.slider("Ensemble BCAR Weight", min_value=0.0, max_value=1.0, value=float(ensemble_weights_cfg["BCAR"]), step=0.01, disabled=not enable_overrides)
                ensemble_weight_adjpyth = st.slider("Ensemble AdjPyth Weight", min_value=0.0, max_value=1.0, value=float(ensemble_weights_cfg["AdjPyth"]), step=0.01, disabled=not enable_overrides)
                ensemble_weight_pyth = st.slider("Ensemble Pyth Weight", min_value=0.0, max_value=1.0, value=float(ensemble_weights_cfg["Pyth"]), step=0.01, disabled=not enable_overrides)
                ensemble_weight_win = st.slider("Ensemble Win% Weight", min_value=0.0, max_value=1.0, value=float(ensemble_weights_cfg["Win"]), step=0.01, disabled=not enable_overrides)

                st.markdown("**Sectional Weights**")
                h2h_weight = st.slider("H2H Weight", min_value=0.0, max_value=1.0, value=float(sectional_cfg["h2h_weight"]), step=0.05, disabled=not enable_overrides)
                common_weight = st.slider("Common Opp Weight", min_value=0.0, max_value=1.0, value=float(sectional_cfg["common_weight"]), step=0.05, disabled=not enable_overrides)
                win_pct_weight = st.slider("Win% Weight", min_value=0.0, max_value=1.0, value=float(sectional_cfg["win_pct_weight"]), step=0.05, disabled=not enable_overrides)

                st.markdown("**SOS Multiplier Range**")
                sos_center = st.slider("SOS Center", min_value=0.0, max_value=1.0, value=float(sectional_cfg["sos_center"]), step=0.05, disabled=not enable_overrides)
                sos_scale = st.slider("SOS Scale", min_value=0.0, max_value=4.0, value=float(sectional_cfg["sos_scale"]), step=0.1, disabled=not enable_overrides)
                sectional_sos_boost = st.slider("Sectional SOS Boost", min_value=0.5, max_value=2.0, value=float(sectional_cfg["sectional_sos_boost"]), step=0.05, disabled=not enable_overrides)

                st.markdown("**Penalty Exponents & Thresholds**")
                game_penalty_threshold = st.slider("Game Penalty Threshold", min_value=0.1, max_value=1.0, value=float(sectional_cfg["game_penalty_threshold"]), step=0.05, disabled=not enable_overrides)
                game_penalty_power = st.slider("Game Penalty Exponent", min_value=0.5, max_value=4.0, value=float(sectional_cfg["game_penalty_power"]), step=0.1, disabled=not enable_overrides)
                sectional_penalty_threshold = st.slider("Sectional Penalty Threshold", min_value=0.1, max_value=1.0, value=float(sectional_cfg["sectional_penalty_threshold"]), step=0.05, disabled=not enable_overrides)
                reliability_floor = st.slider("Reliability Floor", min_value=0.0, max_value=1.0, value=float(sectional_cfg["reliability_floor"]), step=0.05, disabled=not enable_overrides)
                reliability_ceiling = st.slider("Reliability Ceiling", min_value=0.0, max_value=1.0, value=float(sectional_cfg["reliability_ceiling"]), step=0.05, disabled=not enable_overrides)
                reliability_shrink_k = st.slider("Reliability Shrinkage (k)", min_value=0.5, max_value=20.0, value=float(sectional_cfg["reliability_shrink_k"]), step=0.5, disabled=not enable_overrides)
                st.markdown("**Global Prior Blend**")
                global_prior_min_weight = st.slider("Global Prior Min Weight", min_value=0.0, max_value=0.5, value=float(sectional_cfg["global_prior_min_weight"]), step=0.01, disabled=not enable_overrides)
                global_prior_max_weight = st.slider("Global Prior Max Weight", min_value=0.0, max_value=0.5, value=float(sectional_cfg["global_prior_max_weight"]), step=0.01, disabled=not enable_overrides)
                global_prior_shrink_k = st.slider("Global Prior Shrinkage (k)", min_value=0.5, max_value=20.0, value=float(sectional_cfg["global_prior_shrink_k"]), step=0.5, disabled=not enable_overrides)
    active_sectional_params = {
        **sectional_cfg,
        "h2h_weight": h2h_weight,
        "common_weight": common_weight,
        "win_pct_weight": win_pct_weight,
        "sos_center": sos_center,
        "sos_scale": sos_scale,
        "sectional_sos_boost": sectional_sos_boost,
        "game_penalty_threshold": game_penalty_threshold,
        "game_penalty_power": game_penalty_power,
        "sectional_penalty_threshold": sectional_penalty_threshold,
        "reliability_floor": reliability_floor,
        "reliability_ceiling": reliability_ceiling,
        "reliability_shrink_k": reliability_shrink_k,
        "global_prior_min_weight": min(global_prior_min_weight, global_prior_max_weight),
        "global_prior_max_weight": max(global_prior_min_weight, global_prior_max_weight),
        "global_prior_shrink_k": global_prior_shrink_k,
    } if enable_overrides else sectional_cfg

    active_ensemble_weights = sanitize_ensemble_weights({
        "Elo": ensemble_weight_elo,
        "BCAR": ensemble_weight_bcar,
        "AdjPyth": ensemble_weight_adjpyth,
        "Pyth": ensemble_weight_pyth,
        "Win": ensemble_weight_win,
    }) if enable_overrides else ensemble_weights_cfg

    ensemble_weights_cfg = active_ensemble_weights

    active_config = {
        "source": "UI overrides" if enable_overrides else f"defaults from {CONFIG_JSON}",
        "logistic": {"k": k, "x0": x0},
        "elo": {
            "k": elo_k,
            "initial": elo_cfg["initial"],
            "phase_k_enabled": phase_k_enabled,
            "early_phase_games": early_phase_games,
            "early_phase_multiplier": early_phase_multiplier,
            "late_phase_multiplier": late_phase_multiplier,
        },
        "pythag": {"exponent": pythag_exp},
        "imputed_games": {
            "include_estimated_scores": include_estimated_scores,
            "highlight_estimated_games": highlight_estimated_games,
            "down_weight_imputed": down_weight_imputed,
            "imputed_weight": imputed_weight,
        },
        "game_count": {"min_games_ratio": min_games_ratio},
        "ensemble_weights": active_ensemble_weights,
        "sectional": active_sectional_params,
    }

    with st.sidebar.expander("Model Settings", expanded=False):
        st.caption("Active model configuration for reproducibility")
        st.caption("Win% is intentionally low-trust in composite scoring and receives an additional reliability cap in the ensemble.")
        st.json(active_config)
    hybrid_ui_model = st.sidebar.selectbox(
        "Hybrid model",
        options=["Legacy Hybrid", "Schedule-adjusted Hybrid"],
        index=0,
        help="Defaults to the legacy hybrid flow. Select schedule-adjusted to enable the newer fit/prediction UI.",
        key="hybrid_ui_model",
    )
    with st.sidebar.expander("Expert Orders", expanded=False):
        st.caption("Paste Top 25 lists (one team per line; numbering optional) to compare model fit.")
        illpolo_text = st.text_area("Illpolo order", height=180, key="illpolo_order_text")
        maxpreps_text = st.text_area("MaxPreps order", height=180, key="maxpreps_order_text")
    
    # Initial stats
    scored_games = raw_games.dropna(subset=['score1'])
    initial_stats, _ = compute_stats(scored_games)
    all_teams = set(raw_games['team1']).union(raw_games['team2'])
    for t in all_teams:
        if t not in initial_stats:
            initial_stats[t] = {'wins':0,'losses':0,'ties':0,'gf':0,'ga':0,'games':0,'opponents':[]}
    # Infer defaults
    games_inferred = infer_default_scores(raw_games, initial_stats)
    imputed_mode = "full"
    if down_weight_imputed:
        imputed_mode = "down_weight"
    team_imputation = defaultdict(lambda: {"imputed": 0, "games": 0})
    for r in games_inferred.itertuples():
        imp = bool(getattr(r, "is_imputed", False))
        for team in [r.team1, r.team2]:
            team_imputation[team]["games"] += 1
            if imp:
                team_imputation[team]["imputed"] += 1
    st.caption("Estimated scores are generated when results contain winner-only notation (e.g., ‘Team A d. Team B’).")

    # Final stats
    stats,h2h = compute_stats(games_inferred)
    sos = compute_sos(stats)
    py = compute_pythag(games_inferred, stats, exp=pythag_exp, imputed_mode=imputed_mode, imputed_weight=imputed_weight)
    adj_vals = compute_adjusted_pythag(games_inferred, stats, k=k, x0=x0, imputed_mode=imputed_mode, imputed_weight=imputed_weight)
    adj_ord, _ = rank_adj_pyth(stats, games_inferred, h2h, k=k, x0=x0, imputed_mode=imputed_mode, imputed_weight=imputed_weight)
    elo_phase_enabled = phase_k_enabled if enable_overrides else bool(elo_cfg.get("phase_k_enabled", False))
    elo_phase_games = early_phase_games if enable_overrides else int(elo_cfg.get("early_phase_games", 40))
    elo_early_mult = early_phase_multiplier if enable_overrides else float(elo_cfg.get("early_phase_multiplier", 1.15))
    elo_late_mult = late_phase_multiplier if enable_overrides else float(elo_cfg.get("late_phase_multiplier", 0.9))
    elo = compute_elo(
        games_inferred,
        initial=elo_cfg["initial"],
        k=elo_k,
        phase_k_enabled=elo_phase_enabled,
        early_phase_games=elo_phase_games,
        early_phase_multiplier=elo_early_mult,
        late_phase_multiplier=elo_late_mult,
    )
    matchup_agg = build_matchup_aggregate(games_inferred)
    
    # Orders & filters
    win_ord = rank_win_pct(stats,h2h)
    py_ord = rank_pythag(stats,py)
    elo_ord = rank_elo(stats,elo)
    maxg = max(st['games'] for st in stats.values()) if stats else 0
    thr = maxg * min_games_ratio
    win_ord = [t for t in win_ord if stats[t]['games']>=thr]
    py_ord  = [t for t in py_ord  if stats[t]['games']>=thr]
    adj_ord = [t for t in adj_ord if stats[t]['games']>=thr]
    elo_ord = [t for t in elo_ord if stats[t]['games']>=thr]
    teams    = sorted(stats.keys())
    bcar_bundle = compute_confidence_adjusted_bayesian_rank(games_inferred, stats)
    bcar_ord = [t for t in bcar_bundle["order"] if stats[t]["games"] >= thr]
    bcar_table = bcar_bundle["table"][bcar_bundle["table"]["Team"].isin(bcar_ord)].reset_index(drop=True)
    hybrid_bundle = compute_hybrid_rankings(games_inferred, stats, h2h, sos, hybrid_cfg)
    if hybrid_ui_model == "Schedule-adjusted Hybrid":
        hybrid_bundle = compute_schedule_adjusted_hybrid(games_inferred, hybrid_cfg)
    hybrid_ord = [t for t in hybrid_bundle["order"] if stats[t]["games"] >= thr]
    hybrid_table = hybrid_bundle["table"][hybrid_bundle["table"]["Team"].isin(hybrid_ord)].reset_index(drop=True)
    model_orders = {"Win": win_ord, "Pyth": py_ord, "AdjPyth": adj_ord, "BCAR": bcar_ord, "Elo": elo_ord}
    global_prior_teams = [t for t in teams if t in win_ord and t in py_ord and t in adj_ord and t in bcar_ord and t in elo_ord]
    global_prior_df = build_calibrated_ensemble(global_prior_teams, model_orders, stats, h2h, sos, team_imputation, ensemble_base_weights=ensemble_weights_cfg, win_model_cap=win_model_cap_cfg, ensemble_breadth_cfg=ensemble_breadth_cfg, expert_nudge_cfg=expert_nudge_cfg)
    global_prior_scores = dict(zip(global_prior_df["Team"], global_prior_df["Calibrated Score"]))
    primary_payload = build_primary_ranking_payload(
        teams, model_orders, stats, h2h, sos, team_imputation,
        ensemble_base_weights=ensemble_weights_cfg,
        win_model_cap=win_model_cap_cfg,
        ensemble_breadth_cfg=ensemble_breadth_cfg,
        expert_nudge_cfg=expert_nudge_cfg
    )

    # Compute sectional rankings
    sectional_rankings, sectional_order, sectional_breakdowns = compute_sectional_rankings(
        stats, h2h, games_inferred, sos, matchup_agg, global_prior_scores=global_prior_scores, sectional_params=active_sectional_params
    )
    
    # Prior snapshot orders (used in Casual + Changes views)
    previous_orders = None
    uploader = st.session_state.get("uploader")
    if uploader and not prior_games.empty:
        prior_scored_games = prior_games.dropna(subset=["score1"])
        prior_base_stats, _ = compute_stats(prior_scored_games)
        prev_stats, prev_h2h = compute_stats(infer_default_scores(prior_games, prior_base_stats))
        prev_py = compute_pythag(
            infer_default_scores(prior_games, prev_stats),
            prev_stats,
            exp=pythag_exp,
            imputed_mode=imputed_mode,
            imputed_weight=imputed_weight,
        )
        prev_adj, _ = rank_adj_pyth(
            prev_stats,
            infer_default_scores(prior_games, prev_stats),
            prev_h2h,
            k=k,
            x0=x0,
            imputed_mode=imputed_mode,
            imputed_weight=imputed_weight,
        )
        prev_elo = compute_elo(infer_default_scores(prior_games, prev_stats), initial=elo_cfg["initial"], k=elo_k)
        previous_orders = {"Win%": rank_win_pct(prev_stats, prev_h2h), "Pythag": rank_pythag(prev_stats, prev_py), "AdjPyth": prev_adj, "Elo": rank_elo(prev_stats, prev_elo)}

    sort_modes = ["Ensemble rank", "Elo", "BCAR", "AdjPyth", "Pyth", "Win%", "SOS"]
    if st.session_state.get("dashboard_metric_lens") in LEGACY_METRIC_DEFAULTS:
        st.session_state["dashboard_metric_lens"] = DEFAULT_DASHBOARD_METRIC_LENS
    if st.session_state.get("dashboard_weekly_metric") in LEGACY_METRIC_DEFAULTS:
        st.session_state["dashboard_weekly_metric"] = DEFAULT_WEEKLY_TREND_METRIC
    if st.session_state.get("rank_table_sort_mode") in LEGACY_METRIC_DEFAULTS or st.session_state.get("rank_table_sort_mode") == "Ensemble rank":
        st.session_state["rank_table_sort_mode"] = DEFAULT_SORT_MODE
    if st.session_state.get("team_selector_sort_mode") in LEGACY_METRIC_DEFAULTS or st.session_state.get("team_selector_sort_mode") == "Ensemble rank":
        st.session_state["team_selector_sort_mode"] = DEFAULT_SORT_MODE
    bcar_scores = dict(zip(bcar_table["Team"], bcar_table["BCAR Score"])) if not bcar_table.empty else {}

    # Team profile selection
    default_team = "Evanston"
    default_compare_team = "New Trier"
    st.sidebar.header("Team Profile")
    if st.session_state.get("team_selector_sort_mode") in LEGACY_METRIC_DEFAULTS or st.session_state.get("team_selector_sort_mode") == "Ensemble rank":
        st.session_state["team_selector_sort_mode"] = DEFAULT_SORT_MODE
    selector_sort_mode = st.sidebar.selectbox("Team selector order", sort_modes, key="team_selector_sort_mode")
    selector_order = sort_teams_by_mode(
        selector_sort_mode, teams, stats, sos, primary_payload["table"], elo=elo, bcar_table=bcar_table, adj_vals=adj_vals, pyth_vals=py
    )
    default_team_index = selector_order.index(default_team) if default_team in selector_order else 0
    if "selected_team_route_target" not in st.session_state:
        st.session_state["selected_team_route_target"] = None
    team_slug_lookup = build_team_slug_lookup(teams)
    team_slug_reverse = {slug: team for team, slug in team_slug_lookup.items()}

    # Lifecycle safety order (must remain deterministic):
    # 1) parse query params -> 2) resolve route target -> 3) initialize state defaults -> 4) render widgets.
    team_from_url = str(st.query_params.get("team", "")).strip()
    valid_team_from_url = None
    if team_from_url in teams:
        valid_team_from_url = team_from_url
    elif team_from_url in team_slug_reverse:
        valid_team_from_url = team_slug_reverse[team_from_url]

    default_route_team = selector_order[default_team_index]
    legacy_target = resolve_legacy_public_target(st.query_params, fallback_team=default_route_team)
    if legacy_target and legacy_target.get("team_slug") and legacy_target["team_slug"] in team_slug_reverse:
        legacy_target["team"] = team_slug_reverse[legacy_target["team_slug"]]

    resolved_route_target = valid_team_from_url
    if not resolved_route_target and legacy_target and legacy_target.get("team"):
        legacy_team = legacy_target["team"]
        if legacy_team in selector_order:
            resolved_route_target = legacy_team
        else:
            st.warning(f"Ignored invalid legacy team target: {legacy_team}. Falling back to default team.")

    if resolved_route_target:
        st.session_state["selected_team_route_target"] = resolved_route_target

    route_target_team = st.session_state.get("selected_team_route_target")
    widget_default_team = default_route_team
    if route_target_team in selector_order:
        widget_default_team = route_target_team
    elif st.session_state.get("selected_team_widget") in selector_order:
        widget_default_team = st.session_state["selected_team_widget"]
    st.session_state["selected_team_widget"] = widget_default_team

    st.sidebar.selectbox("Select Team", selector_order, key="selected_team_widget")
    te = st.session_state.get("selected_team_widget", widget_default_team)
    st.session_state["selected_team"] = te
    if route_target_team in selector_order:
        st.session_state["selected_team_route_target"] = None
    compare_teams = [t for t in selector_order if t != te]
    if compare_teams:
        default_compare_index = compare_teams.index(default_compare_team) if default_compare_team in compare_teams else 0
        if (
            "selected_compare_team" not in st.session_state
            or st.session_state["selected_compare_team"] not in compare_teams
        ):
            st.session_state["selected_compare_team"] = compare_teams[default_compare_index]
        st.sidebar.selectbox("Compare vs", compare_teams, key="selected_compare_team")
    opp = st.session_state.get("selected_compare_team")
    if opp == te or opp not in teams:
        opp = compare_teams[0] if compare_teams else None
        st.session_state["selected_compare_team"] = opp
    team_resume = build_team_resume(te, games_inferred, bcar_scores, stats, h2h, sos)
    # Compute individual ranks
    ranks = {}
    ranks['win']  = win_ord.index(te)+1 if te in win_ord else None
    ranks['py']   = py_ord.index(te)+1 if te in py_ord else None
    ranks['adj']  = adj_ord.index(te)+1 if te in adj_ord else None
    ranks['elo']  = elo_ord.index(te)+1 if te in elo_ord else None
    ranks_list   = [v for v in ranks.values() if v]
    r_avg = round(sum(ranks_list)/len(ranks_list),2) if ranks_list else None
    if legacy_target:
        st.session_state["primary_nav"] = legacy_target["target_nav"]
        if st.session_state.get("selected_team_route_target") in selector_order:
            st.query_params["team"] = st.session_state["selected_team_route_target"]
        st.query_params["primary_nav"] = legacy_target["target_nav"]
        st.query_params["redirected_from"] = "legacy_public_tab"
        st.rerun()

    # Tabs & content
    if current_nav == "Rankings":
        if "dashboard_top10_only" not in st.session_state:
            st.session_state["dashboard_top10_only"] = False
        _init_dashboard_timeframe_state()
        _init_dashboard_metric_state()
        # Backward-compatible cleanup: ignore legacy persona state from prior sessions/links.
        st.session_state.pop("dashboard_persona", None)

        # Safe fallback for persisted/linked state values outside the current control options.
        if not isinstance(st.session_state["dashboard_top10_only"], bool):
            st.session_state["dashboard_top10_only"] = False
        if st.session_state["dashboard_time_window"] not in {"Last 4 weeks", "All"}:
            st.session_state["dashboard_time_window"] = "Last 4 weeks"
        if st.session_state["dashboard_metric_lens"] not in METRIC_LENS_OPTIONS:
            st.session_state["dashboard_metric_lens"] = DEFAULT_DASHBOARD_METRIC_LENS

        trend_top_n, movement_top_n = render_dashboard_controls()
        st.query_params["section"] = "rankings"

        metric_lens = st.session_state["dashboard_metric_lens"]
        if metric_lens == "BCAR":
            dashboard_order = bcar_ord
            dashboard_metric_values = dict(zip(bcar_table["Team"], bcar_table["BCAR Score"])) if not bcar_table.empty else {}
            metric_label = "BCAR"
            metric_format = CHART_FORMATS["float3"]
        elif metric_lens == "Adj Pyth":
            dashboard_order = adj_ord
            dashboard_metric_values = adj_vals
            metric_label = "Adj Pyth"
            metric_format = CHART_FORMATS["float3"]
        elif metric_lens == "Elo":
            dashboard_order = elo_ord
            dashboard_metric_values = elo
            metric_label = "Elo"
            metric_format = CHART_FORMATS["float1"]
        elif metric_lens == "Ensemble":
            dashboard_order = primary_payload["ordered_teams"]
            dashboard_metric_values = primary_payload["score_lookup"]
            metric_label = "Ensemble (Primary)"
            metric_format = CHART_FORMATS["float3"]
        else:
            dashboard_order = win_ord
            dashboard_metric_values = {t: stats.get(t, {}).get("win_pct", 0.0) for t in stats}
            metric_label = "Win %"
            metric_format = CHART_FORMATS["pct3"]

        render_spacing("section")
        weekly_ranks = compute_weekly_rank_history(DATA_DIR)
        window_size = 4 if st.session_state["dashboard_time_window"] == "Last 4 weeks" else 999
        dashboard_vm = build_dashboard_view_model(
            stats,
            dashboard_order,
            dashboard_metric_values,
            metric_label,
            games_inferred,
            weekly_ranks,
            window_size=window_size,
            top_n_rank=10 if st.session_state["dashboard_top10_only"] else 25,
            trend_top_n=trend_top_n,
            movement_top_n=movement_top_n,
            primary_table=primary_payload["table"],
            sos=sos,
        )
        st.subheader("Public Rankings")
        st.caption("BCAR — Primary public ranking metric.")
        rank_table = dashboard_vm["current_rank_table"].copy()
        if "team" in rank_table.columns:
            rank_table = rank_table.rename(columns={"team": "Team"})
        primary_cols = [c for c in ["Rank", "Team", "BCAR"] if c in rank_table.columns]
        if not primary_cols:
            primary_cols = rank_table.columns.tolist()
        st.dataframe(rank_table[primary_cols], use_container_width=True, hide_index=True)
        with st.expander("See all metrics", expanded=False):
            track_ui_event("metrics_expand_opened", surface="rankings")
            secondary_cols = [c for c in SECONDARY_METRIC_ORDER if c in rank_table.columns]
            if secondary_cols:
                st.dataframe(rank_table[[*primary_cols, *secondary_cols]], use_container_width=True, hide_index=True)
                st.caption("Secondary metrics are context signals; BCAR remains the primary ranking metric.")
                for metric_name in secondary_cols:
                    st.caption(f"• {metric_name}: {SECONDARY_METRIC_TOOLTIPS.get(metric_name, 'Secondary context metric.')}")
        with st.expander(f"Why #{primary_payload['ordered_teams'].index(te)+1} {te}?", expanded=False):
            render_team_explainer_card(build_team_explainer_card(te, stats, sos, h2h, team_imputation))
            st.caption("Public explainer only: internal tuning parameters are intentionally hidden.")
        return

    if current_nav == "Team Profiles/Resume":
        selected_section = "Profile"
        st.query_params["section"] = selected_section
        st.query_params["team"] = team_slug_lookup.get(te, te)
    elif current_nav == "Sectionals":
        selected_section = "Sectionals"
        st.query_params["section"] = selected_section
        st.query_params["team"] = team_slug_lookup.get(te, te)
    else:
        selected_section = None

    section_defaults = {
        "Rankings": "BCAR",
        "Admin / Internal": "Sectionals",
    }
    all_sections = ["BCAR", "Profile", "Sectionals", "Win%", "Pythag", "AdjPyth", "Elo", "Hybrid", "Ensemble (Primary)"]
    available_sections = {
        "Rankings": ["BCAR"],
        "Admin / Internal": ["Sectionals", "Win%", "Pythag", "AdjPyth", "Elo", "Hybrid", "Ensemble (Primary)"],
    }.get(current_nav, all_sections)
    if selected_section is None:
        default_section = section_defaults.get(current_nav, "BCAR")
        if "content_section" not in st.session_state or st.session_state["content_section"] not in available_sections:
            st.session_state["content_section"] = default_section if default_section in available_sections else available_sections[0]
        st.markdown("### Content")
        st.radio("View", options=available_sections, horizontal=True, key="content_section")
        selected_section = st.session_state["content_section"]
        st.query_params["section"] = selected_section
        st.query_params["team"] = team_slug_lookup.get(te, te)
    table_sort_mode = st.session_state.get("rank_table_sort_mode", DEFAULT_SORT_MODE)
    if table_sort_mode in LEGACY_METRIC_DEFAULTS or table_sort_mode == "Ensemble rank":
        table_sort_mode = DEFAULT_SORT_MODE
        st.session_state["rank_table_sort_mode"] = DEFAULT_SORT_MODE
    if selected_section in ["Win%", "Pythag", "AdjPyth", "Elo", "BCAR", "Hybrid", "Ensemble (Primary)"]:
        table_sort_mode = st.radio(
            "Team order",
            options=sort_modes,
            horizontal=True,
            key="rank_table_sort_mode",
        )

    table_order = sort_teams_by_mode(
        table_sort_mode, teams, stats, sos, primary_payload["table"], elo=elo, bcar_table=bcar_table, adj_vals=adj_vals, pyth_vals=py
    )

    if selected_section == "Profile":
        st.subheader(f"Profile: {te}")
        team_rank = primary_payload["ordered_teams"].index(te) + 1 if te in primary_payload["ordered_teams"] else "-"
        canonical_path = build_team_canonical_path(
            team_slug_lookup.get(te, slugify_team_name(te)),
            section="profile",
            timeframe="all" if st.session_state.get("dashboard_time_window") == "All" else "last-4-weeks",
        )
        share_meta = build_profile_share_metadata(
            te,
            team_rank,
            f"BCAR {bcar_scores.get(te, 0.0):.3f}",
            f"{te} profile snapshot with BCAR, resume context, and confidence tiers.",
            canonical_path,
        )
        st.caption(f"Canonical URL: {share_meta['canonical_url']}")
        st.caption(f"Share preview: {share_meta['title']} · {share_meta['summary']} · {share_meta['key_rank_signal']}")
        st.code(share_meta["share_text"], language="text")
        st.download_button(
            "Copy share text (mobile-friendly)",
            data=share_meta["share_text"],
            file_name=f"{team_slug_lookup.get(te, 'team')}-share.txt",
            mime="text/plain",
        )
        primary_profile = {
            'GPG For':f"{stats[te]['gf']/stats[te]['games']:.2f}",
            'GPG Against':f"{stats[te]['ga']/stats[te]['games']:.2f}",
            'GD/Game':f"{(stats[te]['gf']-stats[te]['ga'])/stats[te]['games']:.2f}",
            'BCAR (Primary)': f"{bcar_scores.get(te, 0.0):.3f}",
            'Imputed Share': f"{(team_imputation[te]['imputed']/team_imputation[te]['games'] if team_imputation[te]['games'] else 0):.1%}"
        }
        st.table(pd.DataFrame.from_dict(primary_profile, orient='index', columns=['Value']))
        with st.expander("See all metrics", expanded=False):
            track_ui_event("metrics_expand_opened", surface="profile")
            secondary_profile = {
                'Win %':f"{stats[te]['win_pct']:.3f}",
                'SOS':f"{sos[te]:.3f}",
                'Ensemble Score':r_avg,
                'Rank Elo':ranks['elo'],
                'Rank Adj Pyth':ranks['adj'],
                'Rank Pythag':ranks['py'],
                'Rank Win %':ranks['win'],
            }
            st.table(pd.DataFrame.from_dict(secondary_profile, orient='index', columns=['Value']))
            st.caption("Secondary profile metrics are directional context and are not equally weighted with BCAR.")
            st.caption("• Elo: Elo rating-based strength estimate used as a secondary cross-check.")
            st.caption("• Adj Pyth: Adjusted Pythagorean expectation from scored results.")
            st.caption("• Win %: Raw win percentage; useful context but less schedule-aware.")
            st.caption("• Ensemble Score: Composite score that blends multiple secondary models.")
        team_conf_row = primary_payload["table"].set_index("Team").loc[te] if te in set(primary_payload["table"]["Team"]) else None
        if team_conf_row is not None:
            st.caption(
                f"Confidence context — Games: {team_conf_row['Games Confidence']:.1f}/100 · SOS: {team_conf_row['SOS Confidence']:.1f}/100 · Composite: {team_conf_row['Composite Confidence']:.1f}/100 ({team_conf_row['Confidence Tier']})"
            )
        profile_card = build_team_explainer_card(te, stats, sos, h2h, team_imputation)
        render_team_explainer_card(profile_card)
        st.caption(f"Shared context: {te} vs {opp}")
        st.markdown("**Team resume**")
        resume_summary = team_resume["summary"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Record", resume_summary["record"])
        r2.metric("Goal Diff", f"{resume_summary['goal_diff']:+d}")
        r3.metric("Streak", resume_summary["notable_streak"])
        r4.metric("Vs ranked", resume_summary["ranked_opponent_split"])

        top_wins_df = pd.DataFrame(team_resume["top_wins"])
        worst_losses_df = pd.DataFrame(team_resume["worst_losses"])
        recent_form_df = pd.DataFrame(team_resume["recent_form"])
        sos_context_df = pd.DataFrame(team_resume["strength_of_schedule"])

        table_cols = ["opponent", "team_score", "opp_score", "margin", "quality"]
        for df in [top_wins_df, worst_losses_df, recent_form_df, sos_context_df]:
            for col in table_cols:
                if col not in df.columns:
                    df[col] = pd.Series(dtype=float if col in {"margin", "quality"} else object)

        rw1, rw2 = st.columns(2)
        with rw1:
            st.caption("Top wins")
            if top_wins_df.empty:
                st.info(team_resume["empty_states"]["top_wins"])
            else:
                st.dataframe(top_wins_df[table_cols], use_container_width=True)
        with rw2:
            st.caption("Worst losses")
            if worst_losses_df.empty:
                st.info(team_resume["empty_states"]["worst_losses"])
            else:
                st.dataframe(worst_losses_df[table_cols], use_container_width=True)

        compact_cols = table_cols + ["location", "match_date", "opp_rank_at_match"]

        with st.expander("More resume context", expanded=False):
            rw3, rw4 = st.columns(2)
            with rw3:
                st.caption("Recent form")
                if recent_form_df.empty:
                    st.info(team_resume["empty_states"]["recent_form"])
                else:
                    st.dataframe(recent_form_df[compact_cols], use_container_width=True)
            with rw4:
                st.caption("Strength of schedule")
                if sos_context_df.empty:
                    st.info(team_resume["empty_states"]["strength_of_schedule"])
                else:
                    st.dataframe(sos_context_df[compact_cols], use_container_width=True)
        h = h2h.get((te,opp),{'wins':0,'games':0})
        st.markdown(f"**H2H**: {h['wins']}-{h['games']-h['wins']} in {h['games']} games")
        st.caption("Head-to-head explorer: watch margin trend and whether recent meetings differ from overall record.")

        filter_col, n_col = st.columns([2, 1])
        with filter_col:
            meeting_filter = st.radio("Meeting filter", ["All meetings", "Last N meetings"], horizontal=True)
        with n_col:
            last_n = st.number_input("N", min_value=2, max_value=20, value=5, step=1, disabled=meeting_filter != "Last N meetings")

        matchup_games = []
        for i, g in enumerate(games_inferred.itertuples(), start=1):
            if {g.team1, g.team2} == {te, opp}:
                if g.team1 == te:
                    my_score, opp_score = g.score1, g.score2
                else:
                    my_score, opp_score = g.score2, g.score1
                margin = my_score - opp_score
                outcome = "positive" if margin > 0 else ("negative" if margin < 0 else "neutral")
                matchup_games.append({
                    "Meeting": len(matchup_games) + 1,
                    "Week/Date": f"Game {i}",
                    "Scoreline": f"{te} {my_score} - {opp_score} {opp}",
                    "For": my_score,
                    "Against": opp_score,
                    "Margin": margin,
                    "Outcome": outcome,
                    "Inferred": bool(getattr(g, "is_imputed", False)),
                })

        if matchup_games:
            matchup_df = pd.DataFrame(matchup_games)
            if meeting_filter == "Last N meetings":
                matchup_df = matchup_df.tail(int(last_n)).reset_index(drop=True)
                matchup_df["Meeting"] = matchup_df.index + 1

            wins = int((matchup_df["Outcome"] == "positive").sum())
            losses = int((matchup_df["Outcome"] == "negative").sum())
            ties = int((matchup_df["Outcome"] == "neutral").sum())
            total_for = int(matchup_df["For"].sum())
            total_against = int(matchup_df["Against"].sum())
            avg_margin = matchup_df["Margin"].mean()
            recent_three = matchup_df.tail(3)
            recent_w = int((recent_three["Outcome"] == "positive").sum())
            recent_l = int((recent_three["Outcome"] == "negative").sum())
            recent_t = int((recent_three["Outcome"] == "neutral").sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Record", f"{wins}-{losses}-{ties}")
            k2.metric("Avg Margin", f"{avg_margin:+.2f}")
            k3.metric("Total Goals For/Against", f"{total_for}/{total_against}")
            k4.metric("Recent 3", f"{recent_w}-{recent_l}-{recent_t}")

            margin_chart = alt.Chart(matchup_df).mark_bar().encode(
                x=alt.X("Meeting:O", title="Meeting (chronological)"),
                y=alt.Y("Margin:Q", title=f"{te} margin", scale=alt.Scale(domainMid=0)),
                color=alt.Color(
                    "Outcome:N",
                    scale=alt.Scale(
                        domain=["positive", "neutral", "negative"],
                        range=[SEMANTIC_COLORS["positive"], SEMANTIC_COLORS["neutral"], SEMANTIC_COLORS["negative"]],
                    ),
                    legend=alt.Legend(title="Outcome"),
                ),
                tooltip=["Week/Date", "Scoreline", "Margin", "Outcome", "Inferred"],
            ).properties(height=220)

            outcome_counts = matchup_df.groupby("Outcome").size().reset_index(name="Count")
            outcome_counts["stack"] = "Outcomes"
            outcome_chart = alt.Chart(outcome_counts).mark_bar().encode(
                x=alt.X("stack:N", axis=alt.Axis(title=None, labels=False, ticks=False)),
                y=alt.Y("Count:Q", title="Outcome count"),
                color=alt.Color(
                    "Outcome:N",
                    scale=alt.Scale(
                        domain=["positive", "neutral", "negative"],
                        range=[SEMANTIC_COLORS["positive"], SEMANTIC_COLORS["neutral"], SEMANTIC_COLORS["negative"]],
                    ),
                ),
                order=alt.Order("Outcome:N", sort="ascending"),
                tooltip=["Outcome", "Count"],
            ).properties(height=220)

            st.altair_chart((margin_chart | outcome_chart).resolve_scale(color="shared"), use_container_width=True)
            matchup_view = matchup_df[["Week/Date", "Scoreline", "Margin", "Inferred"]].copy()
            if highlight_estimated_games:
                matchup_view["Badge"] = matchup_view["Inferred"].map(lambda x: "🧩" if x else "")
            st.dataframe(matchup_view, use_container_width=True)
        if te in hybrid_bundle["ratings"] and opp in hybrid_bundle["ratings"]:
            st.markdown("**Hybrid matchup outlook**")
            proj = build_hybrid_matchup_projection(te, opp, stats, hybrid_bundle["ratings"], hybrid_cfg)
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected goals", f"{te} {proj['lam_a']:.2f} - {proj['lam_b']:.2f} {opp}")
            c2.metric("W / D / L", f"{proj['p_win']:.1%} / {proj['p_draw']:.1%} / {proj['p_loss']:.1%}")
            c3.metric("Matchup confidence", f"{proj['confidence']:.1f}")
            st.caption(f"Expected GD interval: [{proj['gd_interval'][0]:+.2f}, {proj['gd_interval'][1]:+.2f}] · Total goals interval: [{proj['tg_interval'][0]:.2f}, {proj['tg_interval'][1]:.2f}]")
            top_scores_df = pd.DataFrame([{"Scoreline": f"{te} {a} - {b} {opp}", "Probability": p} for a, b, p in proj["top_scores"]])
            st.dataframe(top_scores_df, use_container_width=True, hide_index=True)
        st.write(f"{opp} default rank #{primary_payload['rank_lookup'].get(opp, '-')} (Ensemble {primary_payload['score_lookup'].get(opp, 0.0):.3f}) | "
                 f"legacy Win% #{win_ord.index(opp)+1 if opp in win_ord else '-'}, "
                 f"Pyth #{py_ord.index(opp)+1 if opp in py_ord else '-'}, AdjPyth #{adj_ord.index(opp)+1 if opp in adj_ord else '-'}, Elo #{elo_ord.index(opp)+1 if opp in elo_ord else '-'}")
        st.markdown("**Common Opponents**")
        shared_summary = summarize_common_opponents(te, opp, matchup_agg)
        if shared_summary["detail_rows"]:
            rows = []
            for row in shared_summary["detail_rows"]:
                te_wins, te_losses = row[f"{te} record"].split("-")
                opp_wins, opp_losses = row[f"{opp} record"].split("-")
                rows.append(
                    {"Opp": row["Opponent"], f"{te} W": int(te_wins), f"{te} L": int(te_losses), f"{opp} W": int(opp_wins), f"{opp} L": int(opp_losses)}
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Shared-opponent edge ({te}): {shared_summary['normalized_edge_score']:.3f} "
                f"({te} {shared_summary['team_a_record']['wins']}-{shared_summary['team_a_record']['losses']} vs "
                f"{opp} {shared_summary['team_b_record']['wins']}-{shared_summary['team_b_record']['losses']})."
            )
            if shared_summary["confidence_note"]:
                st.caption(shared_summary["confidence_note"])
        else:
            st.write("No common opponents.")
        st.markdown("**Full Schedule**")
        sch=[{'Opp':(r.team2 if r.team1==te else r.team1),
              'Scored':(r.score1 if r.team1==te else r.score2),
              'Allowed':(r.score2 if r.team1==te else r.score1)} for r in games_inferred.itertuples() if r.team1==te or r.team2==te]
        st.dataframe(pd.DataFrame(sch))
    
    if selected_section == "Win%":
        st.subheader(
            "Rankings by Win %",
            help="Tie-breaks: 1) Win%, 2) if exactly two teams are tied use head-to-head, 3) for ties of 3+ use mini-table (head-to-head points, then mini-table win%), 4) mini-table goal differential, 5) full-season goal differential."
        )
        win_view = [t for t in table_order if t in win_ord]
        df_win=pd.DataFrame({'Team':win_view,
                             'Win %':[f"{stats[t]['win_pct']:.3f}" for t in win_view],
                             'SOS':[f"{sos[t]:.3f}" for t in win_view],
                             'Imputed %':[f"{(team_imputation[t]['imputed']/team_imputation[t]['games'] if team_imputation[t]['games'] else 0):.1%}" for t in win_view]})
        df_win["Confidence"] = [build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in win_view]
        st.dataframe(add_imputation_markers(df_win, include_estimated_scores, highlight_estimated_games))
        st.caption(f"Context overlay: {te} rank #{win_ord.index(te)+1 if te in win_ord else '-'} vs {opp} rank #{win_ord.index(opp)+1 if opp in win_ord else '-'}")
        st.caption("Why this rank")
        st.dataframe(build_why_rank_rows(win_ord, {t: stats[t]["win_pct"] for t in win_ord}, stats, h2h, sos, team_imputation))
        chart_data=pd.DataFrame({'Team':win_ord,'Win %':[stats[t]['win_pct'] for t in win_ord]})
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(x='Team',y='Win %'), use_container_width=True)
    
    if selected_section == "Pythag":
        st.subheader("Rankings by Pythagorean")
        py_view = [t for t in table_order if t in py_ord]
        df_py=pd.DataFrame({'Team':py_view,
                            'Exp %':[f"{py[t]:.3f}" for t in py_view],
                            'SOS':[f"{sos[t]:.3f}" for t in py_view],
                            'Imputed %':[f"{(team_imputation[t]['imputed']/team_imputation[t]['games'] if team_imputation[t]['games'] else 0):.1%}" for t in py_view]})
        df_py["Confidence"] = [build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in py_view]
        st.dataframe(add_imputation_markers(df_py, include_estimated_scores, highlight_estimated_games))
        st.caption(f"Context overlay: {te} rank #{py_ord.index(te)+1 if te in py_ord else '-'} vs {opp} rank #{py_ord.index(opp)+1 if opp in py_ord else '-'}")
        st.caption("Why this rank")
        st.dataframe(build_why_rank_rows(py_ord, py, stats, h2h, sos, team_imputation))
        chart_data=pd.DataFrame({'Team':py_ord,'Pythag':[py[t] for t in py_ord]})
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(x='Team',y='Pythag'),use_container_width=True)
    
    if selected_section == "AdjPyth":
        st.subheader("Rankings by Adjusted Pythagorean")
        adj_view = [t for t in table_order if t in adj_ord]
        df_adj=pd.DataFrame({'Team':adj_view,
                             'AdjPyth %':[f"{adj_vals[t]:.3f}" for t in adj_view],
                             'SOS':[f"{sos[t]:.3f}" for t in adj_view],
                             'Imputed %':[f"{(team_imputation[t]['imputed']/team_imputation[t]['games'] if team_imputation[t]['games'] else 0):.1%}" for t in adj_view]})
        df_adj["Confidence"] = [build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in adj_view]
        st.dataframe(add_imputation_markers(df_adj, include_estimated_scores, highlight_estimated_games))
        st.caption(f"Context overlay: {te} rank #{adj_ord.index(te)+1 if te in adj_ord else '-'} vs {opp} rank #{adj_ord.index(opp)+1 if opp in adj_ord else '-'}")
        st.caption("Why this rank")
        st.dataframe(build_why_rank_rows(adj_ord, adj_vals, stats, h2h, sos, team_imputation))
        # Bar chart of adjusted Pyth
        chart_data = pd.Series({t: adj_vals[t] for t in adj_ord}, name='AdjPyth %')
        st.bar_chart(chart_data)
        # Scatter of SOS vs AdjPyth
        df_sc = pd.DataFrame({'Team':list(adj_vals.keys()),
                              'SOS':[sos[t] for t in adj_vals.keys()],
                              'AdjPyth':[adj_vals[t] for t in adj_vals.keys()]})
        scatter = alt.Chart(df_sc).mark_circle(size=60).encode(
            x='SOS', y='AdjPyth', tooltip=['Team','SOS','AdjPyth']
        )
        st.altair_chart(scatter, use_container_width=True)
    
    if selected_section == "Elo":
        st.subheader("Rankings by Elo")
        st.caption("Elo volatility has been intentionally reduced to better match expert poll stability.")
        elo_view = [t for t in table_order if t in elo_ord]
        df_elo=pd.DataFrame({'Team':elo_view,
                             'Elo':[f"{elo[t]:.1f}" for t in elo_view],
                             'SOS':[f"{sos[t]:.3f}" for t in elo_view]})
        df_elo["Confidence"] = [build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in elo_view]
        st.dataframe(add_imputation_markers(df_elo, include_estimated_scores, highlight_estimated_games))
        st.caption(f"Context overlay: {te} rank #{elo_ord.index(te)+1 if te in elo_ord else '-'} vs {opp} rank #{elo_ord.index(opp)+1 if opp in elo_ord else '-'}")
        st.caption("Why this rank")
        st.dataframe(build_why_rank_rows(elo_ord, elo, stats, h2h, sos, team_imputation))
        chart_data=pd.DataFrame({'Team':elo_ord,'Elo':[elo[t] for t in elo_ord]})
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(x='Team',y='Elo'),use_container_width=True)

    if selected_section == "BCAR":
        st.subheader("Rankings by BCAR (Bayesian Confidence-Adjusted Ranking)")
        if bcar_table.empty:
            st.info("No BCAR rows available for the current minimum-games threshold.")
        else:
            show_cols = ["Rank", "Team", "BCAR Score", "Strength", "Strength 95% Low", "Strength 95% High", "SOS_BCAR", "SOV_BCAR", "BL_BCAR", "Resume_BCAR", "Confidence"]
            bcar_view = [t for t in table_order if t in set(bcar_table["Team"].tolist())]
            bcar_view_df = bcar_table.set_index("Team").loc[bcar_view].reset_index()
            st.dataframe(bcar_view_df[show_cols], use_container_width=True, hide_index=True)
            st.caption(f"Context overlay: {te} rank #{primary_payload['rank_lookup'].get(te, '-')} vs {opp} rank #{primary_payload['rank_lookup'].get(opp, '-')} (primary)")

    if selected_section == "Hybrid":
        st.subheader("Rankings by Hybrid")
        st.caption(f"Active hybrid model: {hybrid_ui_model}")
        if hybrid_table.empty:
            st.info("No hybrid rows available for the current minimum-games threshold.")
        else:
            hybrid_view = [t for t in table_order if t in set(hybrid_table["Team"].tolist())]
            hybrid_view_df = hybrid_table.set_index("Team").loc[hybrid_view].reset_index()
            st.dataframe(
                hybrid_view_df[["Rank", "Team", "rating", "rating_se", "ci_low", "ci_high", "confidence", "strength_of_schedule", "strength_of_victory", "volatility_penalty"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Context overlay: {te} rank #{(hybrid_ord.index(te)+1) if te in hybrid_ord else '-'} vs {opp} rank #{(hybrid_ord.index(opp)+1) if opp in hybrid_ord else '-'}")
            if hybrid_ui_model == "Schedule-adjusted Hybrid":
                st.markdown("#### Matchup projection")
                venue = st.selectbox("Venue", options=["neutral", "home", "away"], index=0, key="hybrid_matchup_venue")
                max_goals = st.slider("Max goals per team", min_value=6, max_value=14, value=10, key="hybrid_matchup_max_goals")
                model = hybrid_bundle.get("model")
                if model is not None and opp is not None:
                    pred = model.predict_matchup(te, opp, venue=venue, max_goals=max_goals)
                    st.write(
                        f"Expected goals — {te}: {pred['expected_goals_a']:.2f}, {opp}: {pred['expected_goals_b']:.2f}"
                    )
                    st.write(
                        f"W/D/L — {te}: {pred['p_win']:.1%}, Draw: {pred['p_draw']:.1%}, {opp}: {pred['p_loss']:.1%}"
                    )
                    st.write(
                        f"Goal diff intervals (50/80): {pred['goal_diff_interval_50']} / {pred['goal_diff_interval_80']}"
                    )
                    st.write(
                        f"Total goals intervals (50/80): {pred['total_goals_interval_50']} / {pred['total_goals_interval_80']}"
                    )
            st.write(f"Matchup confidence: {pred['matchup_confidence']:.1f}/100")

    if selected_section == "Matchup Insights":
        st.subheader("Matchup Insights")
        st.caption(f"Head-to-head history: {te} vs {opp}")
        if h2h_record["games"] > 0:
            st.write(f"Record: {h2h_record['wins']}-{h2h_record['losses']} in {h2h_record['games']} games")
        else:
            st.write("No direct head-to-head games logged yet.")

        st.markdown("**Common opponents**")
        shared_summary = summarize_common_opponents(te, opp, matchup_agg)
        if shared_summary["detail_rows"]:
            st.dataframe(pd.DataFrame(shared_summary["detail_rows"]), use_container_width=True, hide_index=True)
            st.caption(
                f"Shared-opponent edge ({te}): {shared_summary['normalized_edge_score']:.3f} "
                f"({te} {shared_summary['team_a_record']['wins']}-{shared_summary['team_a_record']['losses']} vs "
                f"{opp} {shared_summary['team_b_record']['wins']}-{shared_summary['team_b_record']['losses']})."
            )
            if shared_summary["confidence_note"]:
                st.caption(shared_summary["confidence_note"])
        else:
            st.caption("No common-opponent overlap available yet.")

        st.markdown("**Forecast**")
        proj = project_matchup(te, opp, stats, sos, elo, adj_vals, primary_payload["table"])
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{te} win prob", f"{proj['team_a_win_prob']:.1%}")
        c2.metric(f"{opp} win prob", f"{proj['team_b_win_prob']:.1%}")
        c3.metric("Matchup confidence", f"{proj['confidence']:.1f}")

    if selected_section == "Ensemble (Primary)":
        st.subheader("Rank Table Summary")
        df_avg = primary_payload["table"]
        df_avg_view = df_avg.set_index("Team").loc[[t for t in table_order if t in set(df_avg["Team"].tolist())]].reset_index()
        team_list = df_avg_view["Team"].tolist()
        df_avg_lookup = df_avg_view.set_index("Team")
        games_conf_series = df_avg_lookup.get("Games Confidence", pd.Series(dtype=float))
        sos_conf_series = df_avg_lookup.get("SOS Confidence", pd.Series(dtype=float))
        composite_conf_series = df_avg_lookup.get("Composite Confidence", pd.Series(dtype=float))
        tier_series = df_avg_lookup.get("Confidence Tier", pd.Series(dtype=object))
        summary_df = pd.DataFrame({
            "Team": team_list,
            "Ensemble Rank": [primary_payload["rank_lookup"].get(t) for t in team_list],
            "Win Rank": [win_ord.index(t) + 1 if t in win_ord else None for t in team_list],
            "Pyth Rank": [py_ord.index(t) + 1 if t in py_ord else None for t in team_list],
            "AdjPyth Rank": [adj_ord.index(t) + 1 if t in adj_ord else None for t in team_list],
            "Elo Rank": [elo_ord.index(t) + 1 if t in elo_ord else None for t in team_list],
            "BCAR Rank": [bcar_ord.index(t) + 1 if t in bcar_ord else None for t in team_list],
            "SOS": [sos.get(t, 0.0) for t in team_list],
            "Games": [stats.get(t, {}).get("games", 0) for t in team_list],
            "Record": [f"{stats[t]['wins']}-{stats[t]['losses']}-{stats[t]['ties']}" if t in stats else "0-0-0" for t in team_list],
            "Games Confidence": [games_conf_series.get(t, float("nan")) for t in team_list],
            "SOS Confidence": [sos_conf_series.get(t, float("nan")) for t in team_list],
            "Composite Confidence": [composite_conf_series.get(t, float("nan")) for t in team_list],
            "Confidence Tier": [tier_series.get(t, "") for t in team_list],
        })
        st.dataframe(format_confidence_columns(summary_df), use_container_width=True, hide_index=True)
        st.caption("Summary view includes all model ranks, SOS, and resume context.")

        st.markdown("#### Worst-to-best bar view (longer bars = worse rank)")
        bar_df = summary_df[["Team", "Ensemble Rank"]].dropna().copy()
        if not bar_df.empty:
            max_rank = int(bar_df["Ensemble Rank"].max())
            bar_df["Worstness"] = bar_df["Ensemble Rank"].astype(float)
            bar_df["Team Label"] = bar_df["Team"]
            worst_chart = alt.Chart(bar_df).mark_bar().encode(
                y=alt.Y("Team:N", sort=alt.EncodingSortField(field="Ensemble Rank", order="descending"), title=None),
                x=alt.X("Worstness:Q", title="Rank distance from best (0 starts at left)", scale=alt.Scale(domain=[0, max_rank])),
                tooltip=["Team", "Ensemble Rank"]
            )
            team_labels = alt.Chart(bar_df).mark_text(align="left", baseline="middle", dx=4).encode(
                y=alt.Y("Team:N", sort=alt.EncodingSortField(field="Ensemble Rank", order="descending")),
                x=alt.X("Worstness:Q"),
                text="Team Label:N"
            )
            st.altair_chart((worst_chart + team_labels).properties(height=max(420, 22 * len(bar_df))), use_container_width=True)

        st.markdown("#### Visual extras")
        rank_scatter = alt.Chart(summary_df.dropna(subset=["SOS", "Ensemble Rank"])).mark_circle(size=85, opacity=0.8).encode(
            x=alt.X("SOS:Q", title="Strength of Schedule (SOS)"),
            y=alt.Y("Ensemble Rank:Q", title="Ensemble Rank", scale=alt.Scale(reverse=True)),
            color=alt.Color("Confidence:N", title="Confidence"),
            tooltip=["Team", "Ensemble Rank", "SOS", "Win Rank", "Pyth Rank", "AdjPyth Rank", "Elo Rank", "BCAR Rank"]
        ).properties(height=320)
        st.altair_chart(rank_scatter, use_container_width=True)
        if opp:
            st.caption(f"Context overlay (Ensemble Primary): {te} rank #{primary_payload['rank_lookup'].get(te, '-')} vs {opp} rank #{primary_payload['rank_lookup'].get(opp, '-')}")
        render_primary_rank_caption_block()
        st.caption("Per-team contribution diagnostics (normalized model outputs × reliability-weighted contributions).")
        st.dataframe(df_avg[[
            "Team", "Win %tile", "Pyth %tile", "AdjPyth %tile", "BCAR %tile", "Elo %tile",
            "Norm Weight Win", "Norm Weight Pyth", "Norm Weight AdjPyth", "Norm Weight BCAR", "Norm Weight Elo",
            "Weight Win", "Weight Pyth", "Weight AdjPyth", "Weight BCAR", "Weight Elo",
            "Games Ratio", "Coverage Ratio", "Imputation Rate", "Resume Breadth Damping",
            "Unique Opponents", "Unique Opponent Ratio", "Breadth Raw Score"
        ]])
        if is_deep_dive:
            st.markdown("### Rank outcome comparison (with vs without estimated scores)")
            non_imputed_games = games_inferred[~games_inferred["is_imputed"]].copy()
            if not non_imputed_games.empty:
                stats_no_imp, h2h_no_imp = compute_stats(non_imputed_games)
                win_no_imp = rank_win_pct(stats_no_imp, h2h_no_imp)
                rank_with = {t:i+1 for i,t in enumerate(win_ord)}
                rank_without = {t:i+1 for i,t in enumerate(win_no_imp)}
                compare_rows = []
                for t in sorted(set(rank_with).union(rank_without)):
                    rw = rank_with.get(t)
                    rn = rank_without.get(t)
                    compare_rows.append({"Team": t, "With estimated": rw, "Without estimated": rn, "Δ rank (without-with)": (None if rw is None or rn is None else rn-rw), "is_imputed": team_imputation[t]["imputed"]>0})
                compare_df = pd.DataFrame(compare_rows).sort_values(by=["Δ rank (without-with)", "With estimated"], ascending=[False, True], na_position="last")
                st.dataframe(add_imputation_markers(compare_df, include_estimated_scores, highlight_estimated_games), use_container_width=True, hide_index=True)
            else:
                st.caption("No estimated games found to compare.")

            st.markdown("### Model comparison")
            st.caption("Disagreement diagnostic: look for teams with large spread where model assumptions diverge.")
            sort_mode = st.radio(
                "Row sort",
                options=["Disagreement (high to low)", "Ensemble rank", "Elo", "BCAR", "AdjPyth", "Pyth", "Win%", "SOS"],
                horizontal=True,
                key="model_disagreement_sort_mode",
            )
            heatmap_df = df_avg[["Rank", "Team", "Win %tile", "Pyth %tile", "AdjPyth %tile", "Elo %tile"]].copy()
            percentile_cols = ["Win %tile", "Pyth %tile", "AdjPyth %tile", "Elo %tile"]
            heatmap_df["Team Mean"] = heatmap_df[percentile_cols].mean(axis=1)
            heatmap_df["Spread"] = heatmap_df[percentile_cols].max(axis=1) - heatmap_df[percentile_cols].min(axis=1)
            heatmap_df["Std Dev"] = heatmap_df[percentile_cols].std(axis=1, ddof=0)
            if sort_mode == "Disagreement (high to low)":
                ordered_teams = heatmap_df.sort_values(["Spread", "Std Dev", "Rank"], ascending=[False, False, True])["Team"].tolist()
            else:
                ordered_teams = sort_teams_by_mode(
                    sort_mode, heatmap_df["Team"].tolist(), stats, sos, df_avg, elo=elo, bcar_table=bcar_table, adj_vals=adj_vals, pyth_vals=py
                )
            heatmap_long = heatmap_df.melt(
                id_vars=["Rank", "Team", "Team Mean", "Spread", "Std Dev"],
                value_vars=percentile_cols,
                var_name="Model",
                value_name="Percentile",
            )
            heatmap_long["Deviation"] = heatmap_long["Percentile"] - heatmap_long["Team Mean"]

            disagreement_indicator = alt.Chart(heatmap_df).mark_bar(size=10).encode(
                y=alt.Y("Team:N", sort=ordered_teams, title=None),
                x=alt.X("Spread:Q", title="Spread"),
                color=alt.Color("Spread:Q", scale=alt.Scale(scheme="orangered"), legend=None),
                tooltip=[
                    "Team",
                    alt.Tooltip("Spread:Q", format=".3f"),
                    alt.Tooltip("Std Dev:Q", format=".3f"),
                    alt.Tooltip("Team Mean:Q", format=".3f"),
                ],
            ).properties(width=90)

            model_heatmap = alt.Chart(heatmap_long).mark_rect().encode(
                x=alt.X("Model:N", title=None),
                y=alt.Y("Team:N", sort=ordered_teams, title=None),
                color=alt.Color(
                    "Deviation:Q",
                    scale=alt.Scale(scheme="redblue", domainMid=0),
                    title="Deviation from team mean",
                ),
                tooltip=[
                    "Team",
                    "Model",
                    alt.Tooltip("Percentile:Q", format=".3f"),
                    alt.Tooltip("Team Mean:Q", format=".3f"),
                    alt.Tooltip("Deviation:Q", format="+.3f"),
                    alt.Tooltip("Spread:Q", format=".3f"),
                ],
            )
            disagreement_chart = alt.hconcat(disagreement_indicator, model_heatmap, spacing=8).resolve_scale(y="shared")
            st.altair_chart(disagreement_chart.properties(height=360), use_container_width=True)

    if selected_section == "Sectionals":
        st.subheader("Sectional Rankings")
        
        # Display sectional strength rankings
        st.markdown("### Sectional Strength Rankings")
        df_strength = pd.DataFrame({
            'Sectional': sectional_order,
            'Strength': [f"{sum(stats[t]['win_pct'] for t in sectional_rankings[s] if t in stats) / len([t for t in sectional_rankings[s] if t in stats]):.3f}" for s in sectional_order]
        })
        st.dataframe(df_strength)
        
        st.caption("Sectional ordering uses BCAR-weighted scoring with in-season head-to-head guardrails.")

        # Display individual sectional rankings
        for sectional in sectional_order:
            st.markdown(f"### {sectional} Sectional")
            teams = sectional_rankings[sectional]
            
            # Calculate average rank for each team
            avg_ranks = {}
            for team in teams:
                if team in stats:
                    ranks = []
                    if team in win_ord:
                        ranks.append(win_ord.index(team) + 1)
                    if team in py_ord:
                        ranks.append(py_ord.index(team) + 1)
                    if team in adj_ord:
                        ranks.append(adj_ord.index(team) + 1)
                    if team in elo_ord:
                        ranks.append(elo_ord.index(team) + 1)
                    avg_ranks[team] = round(sum(ranks) / len(ranks), 1) if ranks else None
            
            # Basic rankings table with combined score
            df_sectional = pd.DataFrame({
                'Seed': range(1, len(teams) + 1),
                'Team': teams,
                'Combined Score': [f"{stats[t]['sectional_score'][sectional]:.3f}" if t in stats and 'sectional_score' in stats[t] else "N/A" for t in teams],
                'Win %': [f"{stats[t]['win_pct']:.3f}" if t in stats else "N/A" for t in teams],
                'Games': [stats[t]['games'] if t in stats else 0 for t in teams],
                'Avg Rank': [f"{avg_ranks[t]}" if t in avg_ranks else "N/A" for t in teams]
            })
            st.dataframe(df_sectional)
            
            # Detailed breakdown for each team (internal debug mode only)
            if is_deep_dive:
                st.markdown("#### Detailed Seeding Analysis (Internal Debug)")
                for team in teams:
                    if team not in stats:
                        continue

                    with st.expander(f"{team} - Detailed Seeding Analysis"):
                        breakdown = sectional_breakdowns[sectional][team]
                        h2h_score = breakdown["h2h_score"]
                        common_win_pct = breakdown["common_opponent_score"]
                        common_wins_weighted = breakdown["common_wins_weighted"]
                        common_games = breakdown["common_games"]
                        win_pct = breakdown["win_pct"]
                        game_penalty = breakdown["penalties"]["game_penalty"]
                        sectional_penalty = breakdown["penalties"]["sectional_penalty"]

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("H2H (45%)", f"{h2h_score:.3f}")
                        with col2:
                            st.metric("Common Opp (Non-Sectional, 45%)", f"{common_wins_weighted:.1f}/{common_games}", f"{common_win_pct:.3f}")
                        with col3:
                            st.metric("Win % (10%)", f"{win_pct:.3f}")
                        with col4:
                            penalties = []
                            if game_penalty < 1.0:
                                penalties.append(f"Games: {game_penalty:.2f}x")
                            if sectional_penalty < 1.0:
                                penalties.append(f"Sectional: {sectional_penalty:.2f}x")
                            st.metric("Penalties", "None" if not penalties else ", ".join(penalties))

                        st.metric("Combined Score", f"{breakdown['combined_score']:.3f}")

                        st.markdown("##### Head-to-Head Details")
                        if breakdown["h2h_details"]:
                            st.dataframe(pd.DataFrame(breakdown["h2h_details"]))
                        else:
                            st.write("No head-to-head games played")

                        st.markdown("##### Non-Sectional Common Opponents (Used in Common Opp Score)")
                        if breakdown["non_sectional_common_details"]:
                            st.dataframe(pd.DataFrame(breakdown["non_sectional_common_details"]))
                        else:
                            st.write("No non-sectional common opponents")

                        st.markdown("##### Sectional Matchups (Reported for Transparency, Counted in H2H)")
                        if breakdown["sectional_common_details"]:
                            st.dataframe(pd.DataFrame(breakdown["sectional_common_details"]))
                        else:
                            st.write("No sectional common opponents")


if __name__ == "__main__":
    main()
