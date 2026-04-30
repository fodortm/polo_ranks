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
CHART_FORMATS = {"pct3": ".3f", "float2": ".2f", "int0": ".0f"}

def render_typography(level, text):
    if level in TYPOGRAPHY_SCALE:
        st.markdown(f"{TYPOGRAPHY_SCALE[level]} {text}")
    elif level == "caption":
        st.caption(text)
    else:
        st.write(text)

def render_spacing(level="panel"):
    st.markdown(SPACING_SCALE.get(level, SPACING_SCALE["panel"]), unsafe_allow_html=True)

def render_kpi_card(container, label, value, delta=None, caption=None):
    container.metric(label, value, delta=delta)
    if caption:
        container.caption(caption)

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



def build_dashboard_view_model(stats, win_ord, games_inferred, weekly_ranks, window_size=3, top_n_rank=12, trend_top_n=8, movement_top_n=8):
    """Build all dashboard-ready datasets from a single shared computation path."""
    stats = stats or {}
    win_ord = win_ord or []
    games_inferred = games_inferred if games_inferred is not None else pd.DataFrame()
    weekly_ranks = weekly_ranks if weekly_ranks is not None else pd.DataFrame()

    top_team = win_ord[0] if win_ord else None
    top_team_win = stats.get(top_team, {}).get("win_pct") if top_team else None
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
        {"Rank": i + 1, "Team": team, "Win %": stats.get(team, {}).get("win_pct", 0.0)}
        for i, team in enumerate(win_ord[: min(top_n_rank, len(win_ord))])
    ])
    top_teams = win_ord[:min(trend_top_n, len(win_ord))]
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
            "top_team_win": top_team_win,
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
        return json.load(f)

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
    base_weights = ensemble_base_weights or {
        "Elo": 0.45,
        "Pyth": 0.30,
        "AdjPyth": 0.20,
        "Win": 0.05,
    }
    model_pct = {name: rank_percentile_map(order) for name, order in orders.items()}
    rank_lookup = {name: {t: i + 1 for i, t in enumerate(order)} for name, order in orders.items()}
    rows = []
    for team in teams:
        _, confidence, games_ratio, coverage_ratio, imp_ratio = build_confidence_badge(team, stats, h2h, team_imputation, teams)
        breadth_damping, unique_opp_count, unique_opp_ratio, _, _, breadth_raw_score = compute_resume_breadth_factor(
            team, stats, h2h, teams, team_imputation, ensemble_breadth_cfg
        )
        reliability = max(0.0, confidence)
        reliability_modulators = {
            "Win": 0.50 + 0.50 * coverage_ratio,
            "Pyth": 0.50 + 0.50 * games_ratio,
            "AdjPyth": 0.65 + 0.35 * (1 - imp_ratio),
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
            + normalized_weights["Elo"] * model_pct["Elo"].get(team, 0.0)
        )
        calibrated_score = weighted_sum if total_weight else 0.0
        ordinal_ranks = [
            rank_lookup["Win"].get(team, len(orders["Win"]) + 1),
            rank_lookup["Pyth"].get(team, len(orders["Pyth"]) + 1),
            rank_lookup["AdjPyth"].get(team, len(orders["AdjPyth"]) + 1),
            rank_lookup["Elo"].get(team, len(orders["Elo"]) + 1),
        ]
        rows.append({
            "Team": team,
            "Calibrated Score": calibrated_score,
            "Direct H2H Tiebreak": compute_rank_tie_break_key(team, stats, sos, h2h)[0],
            "SOS Margin Tiebreak": compute_rank_tie_break_key(team, stats, sos, h2h)[1],
            "Stable Secondary": compute_rank_tie_break_key(team, stats, sos, h2h)[2],
            "Ordinal Avg (Debug)": round(sum(ordinal_ranks) / len(ordinal_ranks), 2),
            "Win Rank": ordinal_ranks[0],
            "Pyth Rank": ordinal_ranks[1],
            "AdjPyth Rank": ordinal_ranks[2],
            "Elo Rank": ordinal_ranks[3],
            "Win %tile": model_pct["Win"].get(team, 0.0),
            "Pyth %tile": model_pct["Pyth"].get(team, 0.0),
            "AdjPyth %tile": model_pct["AdjPyth"].get(team, 0.0),
            "Elo %tile": model_pct["Elo"].get(team, 0.0),
            "Weight Win": weights["Win"],
            "Weight Pyth": weights["Pyth"],
            "Weight AdjPyth": weights["AdjPyth"],
            "Weight Elo": weights["Elo"],
            "Norm Weight Win": normalized_weights["Win"],
            "Norm Weight Pyth": normalized_weights["Pyth"],
            "Norm Weight AdjPyth": normalized_weights["AdjPyth"],
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


def main():
    st.set_page_config(page_title="Polo Dashboard", layout="wide")
    is_deep_dive = False

    nav_options = ["Dashboard", "Team Profile", "Rank Tables", "Sectionals"]
    if "primary_nav" not in st.session_state or st.session_state["primary_nav"] not in nav_options:
        st.session_state["primary_nav"] = "Dashboard"
    st.markdown("### Primary Navigation")
    st.radio(
        "Go to",
        options=nav_options,
        horizontal=True,
        key="primary_nav",
    )
    current_nav = st.session_state["primary_nav"]
    config = load_model_config()
    ensemble_weights_cfg = {
        "Elo": float(config.get("ensemble_weights", {}).get("Elo", 0.45)),
        "Pyth": float(config.get("ensemble_weights", {}).get("Pyth", 0.30)),
        "AdjPyth": float(config.get("ensemble_weights", {}).get("AdjPyth", 0.20)),
        "Win": float(config.get("ensemble_weights", {}).get("Win", 0.05)),
    }
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

    with st.sidebar.expander("Advanced Settings", expanded=False):
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
        "sectional": active_sectional_params,
    }

    with st.sidebar.expander("Model Settings", expanded=False):
        st.caption("Active model configuration for reproducibility")
        st.caption("Win% is intentionally low-trust in composite scoring and receives an additional reliability cap in the ensemble.")
        st.json(active_config)
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
    model_orders = {"Win": win_ord, "Pyth": py_ord, "AdjPyth": adj_ord, "Elo": elo_ord}
    global_prior_teams = [t for t in teams if t in win_ord and t in py_ord and t in adj_ord and t in elo_ord]
    global_prior_df = build_calibrated_ensemble(global_prior_teams, model_orders, stats, h2h, sos, team_imputation, ensemble_base_weights=ensemble_weights_cfg, win_model_cap=win_model_cap_cfg, ensemble_breadth_cfg=ensemble_breadth_cfg, expert_nudge_cfg=expert_nudge_cfg)
    global_prior_scores = dict(zip(global_prior_df["Team"], global_prior_df["Calibrated Score"]))

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

    # Team profile selection
    default_team = "Evanston"
    default_compare_team = "New Trier"
    st.sidebar.header("Team Profile")
    default_team_index = teams.index(default_team) if default_team in teams else 0
    if "selected_team" not in st.session_state or st.session_state["selected_team"] not in teams:
        st.session_state["selected_team"] = teams[default_team_index]
    te = st.sidebar.selectbox("Select Team", teams, key="selected_team")
    compare_teams = [t for t in teams if t != te]
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
    # Compute individual ranks
    ranks = {}
    ranks['win']  = win_ord.index(te)+1 if te in win_ord else None
    ranks['py']   = py_ord.index(te)+1 if te in py_ord else None
    ranks['adj']  = adj_ord.index(te)+1 if te in adj_ord else None
    ranks['elo']  = elo_ord.index(te)+1 if te in elo_ord else None
    ranks_list   = [v for v in ranks.values() if v]
    r_avg = round(sum(ranks_list)/len(ranks_list),2) if ranks_list else None
    

    # Tabs & content
    if current_nav == "Dashboard":
        render_typography("title", "Dashboard")
        render_typography("subtitle", "League pulse in three visual bands")
        render_typography("caption", "Read top-to-bottom: hero summary, core story visuals, then context + trust signals.")

        # ---------------- Band A: Hero summary ---------------- #
        render_spacing("section")
        render_typography("subtitle", "Band A · Hero summary")
        weekly_ranks = compute_weekly_rank_history(DATA_DIR)
        dashboard_vm = build_dashboard_view_model(stats, win_ord, games_inferred, weekly_ranks)
        kpi = dashboard_vm["kpi_payload"]

        hero_cols = st.columns(6)
        render_kpi_card(hero_cols[0], "Current #1", kpi["top_team"], delta=(f"Win% {kpi['top_team_win']:{CHART_FORMATS['pct3']}}" if kpi["top_team_win"] is not None else None), caption="Best current league position by Win% ranking.")
        render_kpi_card(hero_cols[1], "Biggest riser", kpi["biggest_riser_label"], caption="Largest upward movement in the recent 3-week window.")
        render_kpi_card(hero_cols[2], "Biggest faller", kpi["biggest_faller_label"], caption="Largest downward movement in the recent 3-week window.")
        render_kpi_card(hero_cols[3], "Total games", kpi["total_games"], caption="All parsed matchups currently included in this model run.")
        render_kpi_card(hero_cols[4], "Scored games", kpi["scored_results"], caption="Games with explicit scores recorded in source files.")
        render_kpi_card(hero_cols[5], "Estimated games", kpi["inferred_results"], caption="Games where scores were inferred from model defaults.")

        # ---------------- Band B: Core story visuals ---------------- #
        render_spacing("section")
        render_typography("subtitle", "Band B · Core story visuals")
        b_left, b_right = st.columns([1.2, 1.0])

        with b_left:
            top_n_df = dashboard_vm["current_rank_table"]
            if top_n_df.empty:
                st.info("No rank table available for the current filter window.")
            else:
                rank_chart = alt.Chart(top_n_df).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("Team:N", sort="-x", title=None),
                x=alt.X("Win %:Q", title="Win %", axis=alt.Axis(format=CHART_FORMATS["pct3"])),
                color=alt.condition(alt.datum.Rank == 1, alt.value(RANK_TIER_COLORS["elite"]), alt.value(RANK_TIER_COLORS["contender"])),
                tooltip=[alt.Tooltip("Rank:Q", format=CHART_FORMATS["int0"]), "Team:N", alt.Tooltip("Win %:Q", format=CHART_FORMATS["pct3"])],
                ).properties(title=f"Current rank bar chart (Top {len(top_n_df)})")
                st.altair_chart(apply_chart_theme(rank_chart), use_container_width=True)

        with b_right:
            trend_pool = dashboard_vm["windowed_rank_history"]
            if not trend_pool.empty:
                trend_chart = alt.Chart(trend_pool).mark_line(point=True).encode(
                    x=alt.X("week_num:Q", title="Week"),
                    y=alt.Y("rank:Q", title="Rank (1 is best)", scale=alt.Scale(reverse=True)),
                    color=alt.Color("team:N", legend=alt.Legend(title="Top teams")),
                    tooltip=["team:N", "week_label:N", alt.Tooltip("rank:Q", format=CHART_FORMATS["int0"])],
                ).properties(title="Weekly rank trajectory")
                st.altair_chart(apply_chart_theme(trend_chart), use_container_width=True)
            else:
                st.info("Weekly rank trajectory appears after multiple weekly files are available.")

        movement_rows = dashboard_vm["rank_movement_table"]
        if not movement_rows.empty:
            movement_chart = alt.Chart(movement_rows).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("team:N", sort=None, title=None),
                y=alt.Y("move:Q", title="Week-over-week rank change (+ is better)"),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["Riser", "Flat", "Faller"], range=[SEMANTIC_COLORS["positive"], SEMANTIC_COLORS["neutral"], SEMANTIC_COLORS["negative"]])),
                tooltip=["team:N", alt.Tooltip("latest_rank:Q", format=CHART_FORMATS["int0"]), alt.Tooltip("prior_rank:Q", format=CHART_FORMATS["int0"]), alt.Tooltip("move:Q", format=CHART_FORMATS["int0"]), "direction:N"],
            ).properties(title="Movement (risers/fallers)")
            st.altair_chart(apply_chart_theme(movement_chart), use_container_width=True)
            st.dataframe(
                movement_rows.rename(
                    columns={"team": "Team", "latest_rank": "Current Rank", "prior_rank": "Prior Rank", "move": "Δ Rank", "direction": "Direction"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Rank movement appears when at least two ranking periods are available.")

        # ---------------- Band C: Context and trust ---------------- #
        render_spacing("section")
        render_typography("subtitle", "Band C · Context and trust")
        c_left, c_mid, c_right = st.columns([1, 1, 1])

        with c_left:
            dist_df = dashboard_vm["distribution_dataset"]
            dist_chart = alt.Chart(dist_df).mark_bar().encode(
                x=alt.X("Win %:Q", bin=alt.Bin(maxbins=12), title="Win % bins"),
                y=alt.Y("count():Q", title="Teams"),
                tooltip=[alt.Tooltip("count():Q", title="Teams in bin", format=CHART_FORMATS["int0"])],
            ).properties(title="Metric distribution")
            st.altair_chart(apply_chart_theme(dist_chart), use_container_width=True)

        with c_mid:
            scatter_df = dashboard_vm["offense_defense_dataset"]
            scatter_chart = alt.Chart(scatter_df).mark_circle(size=90, opacity=0.8).encode(
                x=alt.X("Offense (GPG For):Q"),
                y=alt.Y("Defense (GPG Against):Q", scale=alt.Scale(reverse=True)),
                color=alt.condition(alt.datum.Team == kpi["top_team"], alt.value(RANK_TIER_COLORS["elite"]), alt.value(RANK_TIER_COLORS["support"])),
                tooltip=["Team:N", alt.Tooltip("Offense (GPG For):Q", format=CHART_FORMATS["float2"]), alt.Tooltip("Defense (GPG Against):Q", format=CHART_FORMATS["float2"])],
            ).properties(title="Offense vs defense scatter")
            st.altair_chart(apply_chart_theme(scatter_chart), use_container_width=True)

        with c_right:
            trust_metrics = dashboard_vm["trust_metrics"]
            impact_df = dashboard_vm["imputation_impact_dataset"]
            st.markdown("#### Data quality / trust capsule")
            st.caption("Quick confidence read before acting on rankings.")
            st.metric("Trust level", trust_metrics["trust_level"])
            st.progress(trust_metrics["confidence_progress"])
            st.caption(f"Inferred share: {trust_metrics['imputed_pct']:.1%} · Parsed games: {trust_metrics['parsed_games']} · Teams: {trust_metrics['team_count']}")
            st.caption("Lower inferred share generally means more stable ranking confidence.")
            if not impact_df.empty:
                st.dataframe(impact_df, hide_index=True, use_container_width=True)

        st.info("Primary questions answered above: top team, trend direction, and biggest mover without scrolling.")
        return

    section_defaults = {"Team Profile": "Profile", "Rank Tables": "Win%", "Sectionals": "Sectionals"}
    all_sections = ["Profile","Win%","Pythag","AdjPyth","Elo","Avg","3-Metric Plot","Sectionals"]
    available_sections = {"Team Profile": ["Profile"], "Rank Tables": ["Win%","Pythag","AdjPyth","Elo","Avg","3-Metric Plot"], "Sectionals": ["Sectionals"]}.get(current_nav, all_sections)
    default_section = section_defaults.get(current_nav, "Profile")
    if "content_section" not in st.session_state or st.session_state["content_section"] not in available_sections:
        st.session_state["content_section"] = default_section if default_section in available_sections else available_sections[0]
    st.markdown("### Content")
    st.radio("View", options=available_sections, horizontal=True, key="content_section")
    selected_section = st.session_state["content_section"]

    if selected_section == "Profile":
        st.subheader(f"Profile: {te}")
        st.table(pd.DataFrame.from_dict({
            'GPG For':f"{stats[te]['gf']/stats[te]['games']:.2f}",
            'GPG Against':f"{stats[te]['ga']/stats[te]['games']:.2f}",
            'GD/Game':f"{(stats[te]['gf']-stats[te]['ga'])/stats[te]['games']:.2f}",
            'Win %':f"{stats[te]['win_pct']:.3f}",
            'SOS':f"{sos[te]:.3f}",
            'Rank Win%':ranks['win'],'Rank Pythag':ranks['py'],
            'Rank Adj':ranks['adj'],'Rank Elo':ranks['elo'],'Avg':r_avg,
            'Imputed Share': f"{(team_imputation[te]['imputed']/team_imputation[te]['games'] if team_imputation[te]['games'] else 0):.1%}"
        },orient='index',columns=['Value']))
        st.caption(f"Shared context: {te} vs {opp}")
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
        st.write(f"{opp} Ranks: Win% #{win_ord.index(opp)+1 if opp in win_ord else '-'} (SOS {sos[opp]:.3f}), "
                 f"Pyth #{py_ord.index(opp)+1 if opp in py_ord else '-'} (SOS {sos[opp]:.3f}), "
                 f"AdjPyth #{adj_ord.index(opp)+1 if opp in adj_ord else '-'} (SOS {sos[opp]:.3f}), "
                 f"Elo #{elo_ord.index(opp)+1 if opp in elo_ord else '-'}")
        st.markdown("**Common Opponents**")
        com = set(stats[te]['opponents']) & set(stats[opp]['opponents'])
        if com:
            dfc = []
            for c in com:
                try:
                    rec_te = matchup_agg.get((te, c), {"wins": 0, "losses": 0})
                    rec_opp = matchup_agg.get((opp, c), {"wins": 0, "losses": 0})
                    wins_te = rec_te["wins"]
                    losses_te = rec_te["losses"]
                    wins_opp = rec_opp["wins"]
                    losses_opp = rec_opp["losses"]
                    
                    dfc.append({
                        'Opp': c,
                        f'{te} W': wins_te,
                        f'{te} L': losses_te,
                        f'{opp} W': wins_opp,
                        f'{opp} L': losses_opp
                    })
                except Exception as e:
                    st.error(f"Error processing opponent {c}: {str(e)}")
            
            if dfc:
                try:
                    df_common = pd.DataFrame(dfc)
                    st.dataframe(df_common)
                except Exception as e:
                    st.error(f"Error creating DataFrame: {str(e)}")
            else:
                st.write("No common opponent data available")
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
        df_win=pd.DataFrame({'Team':win_ord,
                             'Win %':[f"{stats[t]['win_pct']:.3f}" for t in win_ord],
                             'SOS':[f"{sos[t]:.3f}" for t in win_ord],
                             'Imputed %':[f"{(team_imputation[t]['imputed']/team_imputation[t]['games'] if team_imputation[t]['games'] else 0):.1%}" for t in win_ord]})
        df_win["Confidence"] = [build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in win_ord]
        st.dataframe(add_imputation_markers(df_win, include_estimated_scores, highlight_estimated_games))
        st.caption(f"Context overlay: {te} rank #{win_ord.index(te)+1 if te in win_ord else '-'} vs {opp} rank #{win_ord.index(opp)+1 if opp in win_ord else '-'}")
        st.caption("Why this rank")
        st.dataframe(build_why_rank_rows(win_ord, {t: stats[t]["win_pct"] for t in win_ord}, stats, h2h, sos, team_imputation))
        chart_data=pd.DataFrame({'Team':win_ord,'Win %':[stats[t]['win_pct'] for t in win_ord]})
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(x='Team',y='Win %'), use_container_width=True)
    
    if selected_section == "Pythag":
        st.subheader("Rankings by Pythagorean")
        df_py=pd.DataFrame({'Team':py_ord,
                            'Exp %':[f"{py[t]:.3f}" for t in py_ord],
                            'SOS':[f"{sos[t]:.3f}" for t in py_ord],
                            'Imputed %':[f"{(team_imputation[t]['imputed']/team_imputation[t]['games'] if team_imputation[t]['games'] else 0):.1%}" for t in py_ord]})
        df_py["Confidence"] = [build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in py_ord]
        st.dataframe(add_imputation_markers(df_py, include_estimated_scores, highlight_estimated_games))
        st.caption(f"Context overlay: {te} rank #{py_ord.index(te)+1 if te in py_ord else '-'} vs {opp} rank #{py_ord.index(opp)+1 if opp in py_ord else '-'}")
        st.caption("Why this rank")
        st.dataframe(build_why_rank_rows(py_ord, py, stats, h2h, sos, team_imputation))
        chart_data=pd.DataFrame({'Team':py_ord,'Pythag':[py[t] for t in py_ord]})
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(x='Team',y='Pythag'),use_container_width=True)
    
    if selected_section == "AdjPyth":
        st.subheader("Rankings by Adjusted Pythagorean")
        df_adj=pd.DataFrame({'Team':adj_ord,
                             'AdjPyth %':[f"{adj_vals[t]:.3f}" for t in adj_ord],
                             'SOS':[f"{sos[t]:.3f}" for t in adj_ord],
                             'Imputed %':[f"{(team_imputation[t]['imputed']/team_imputation[t]['games'] if team_imputation[t]['games'] else 0):.1%}" for t in adj_ord]})
        df_adj["Confidence"] = [build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in adj_ord]
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
        df_elo=pd.DataFrame({'Team':elo_ord,
                             'Elo':[f"{elo[t]:.1f}" for t in elo_ord],
                             'SOS':[f"{sos[t]:.3f}" for t in elo_ord]})
        df_elo["Confidence"] = [build_confidence_badge(t, stats, h2h, team_imputation, teams)[0] for t in elo_ord]
        st.dataframe(add_imputation_markers(df_elo, include_estimated_scores, highlight_estimated_games))
        st.caption(f"Context overlay: {te} rank #{elo_ord.index(te)+1 if te in elo_ord else '-'} vs {opp} rank #{elo_ord.index(opp)+1 if opp in elo_ord else '-'}")
        st.caption("Why this rank")
        st.dataframe(build_why_rank_rows(elo_ord, elo, stats, h2h, sos, team_imputation))
        chart_data=pd.DataFrame({'Team':elo_ord,'Elo':[elo[t] for t in elo_ord]})
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(x='Team',y='Elo'),use_container_width=True)
    
    if selected_section == "Avg":
        st.subheader("Rankings by Calibrated Ensemble")
        eligible_teams = [t for t in teams if stats[t]['games'] >= thr and t in win_ord and t in py_ord and t in adj_ord and t in elo_ord]
        df_avg = build_calibrated_ensemble(eligible_teams, model_orders, stats, h2h, sos, team_imputation, ensemble_base_weights=ensemble_weights_cfg, win_model_cap=win_model_cap_cfg, ensemble_breadth_cfg=ensemble_breadth_cfg, expert_nudge_cfg=expert_nudge_cfg)
        st.dataframe(df_avg[[
            "Rank", "Team", "Calibrated Score", "Ordinal Avg (Debug)", "Direct H2H Tiebreak", "SOS Margin Tiebreak", "Stable Secondary"
        ]])
        if opp:
            avg_rank_lookup = {row["Team"]: int(row["Rank"]) for _, row in df_avg.iterrows()}
            st.caption(f"Context overlay: {te} rank #{avg_rank_lookup.get(te, '-')} vs {opp} rank #{avg_rank_lookup.get(opp, '-')}")
        st.caption("Per-team contribution breakdown (normalized model outputs × reliability-weighted contributions).")
        st.dataframe(df_avg[[
            "Team", "Win %tile", "Pyth %tile", "AdjPyth %tile", "Elo %tile",
            "Norm Weight Win", "Norm Weight Pyth", "Norm Weight AdjPyth", "Norm Weight Elo",
            "Weight Win", "Weight Pyth", "Weight AdjPyth", "Weight Elo",
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
                options=["Disagreement (high to low)", "Ensemble rank"],
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
                ordered_teams = heatmap_df.sort_values("Rank")["Team"].tolist()
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

    if selected_section == "3-Metric Plot":
        st.subheader("Team Profile Space: Win% vs Adjusted Pythagorean (Elo size, SOS color)")
        st.caption("Elo is encoded by size (third-axis proxy) and Strength of Schedule (SOS) is encoded by color (red = weaker, yellow = mid, green = harder).")
        ctl1, ctl2 = st.columns([1, 1.5])
        with ctl1:
            st.caption("Encoding: Elo → size · SOS → color (red → yellow → green)")
        with ctl2:
            max_games = max(sts["games"] for sts in stats.values()) if stats else 1
            min_games_plot = st.slider("Minimum games", min_value=0, max_value=max_games, value=int(round(thr)), step=1, key="profile3d_min_games")
        team_search = st.text_input("Team search highlight", value="", key="profile3d_search").strip().lower()

        profile_df = build_team_profile_3metric_df(stats, adj_vals, elo, sos, min_games_plot)
        if profile_df.empty:
            st.info("No teams meet the current minimum games threshold.")
        else:
            focus_options = ["(None)"] + sorted(profile_df["Team"].tolist())
            focus_team = st.selectbox("Focus team", options=focus_options, index=0, key="profile3d_focus_team")
            profile_df["SearchHit"] = profile_df["Team"].str.lower().str.contains(team_search, regex=False) if team_search else False
            profile_df["IsFocus"] = profile_df["Team"] == focus_team
            profile_df["Highlight"] = profile_df["IsFocus"] | profile_df["SearchHit"]
            elo_min = float(profile_df["Elo"].min())
            elo_max = float(profile_df["Elo"].max())
            elo_span = max(elo_max - elo_min, 1e-6)
            profile_df["EloNorm"] = (profile_df["Elo"] - elo_min) / elo_span
            sos_min = float(profile_df["SOS"].min())
            sos_max = float(profile_df["SOS"].max())

            hover_sel = alt.selection_point(fields=["Team"], on="mouseover", empty=True, name="team_hover")
            base_opacity = alt.condition(hover_sel | alt.datum.Highlight, alt.value(1.0), alt.value(0.25))

            sos_color = alt.Color(
                "SOS:Q",
                scale=alt.Scale(
                    domain=[sos_min, (sos_min + sos_max) / 2.0, sos_max],
                    range=["#B91C1C", "#FACC15", "#15803D"],
                ),
                legend=alt.Legend(title=f"SOS ({sos_min:.3f}–{sos_max:.3f})"),
            )
            points = alt.Chart(profile_df).mark_circle().encode(
                x=alt.X("WinPct:Q", title="Win %"),
                y=alt.Y("AdjPyth:Q", title="Adjusted Pythagorean"),
                size=alt.Size(
                    "EloNorm:Q",
                    legend=alt.Legend(title=f"Elo ({elo_min:.0f}–{elo_max:.0f})"),
                    scale=alt.Scale(domain=[0, 1], range=[40, 900]),
                ),
                color=sos_color,
                opacity=base_opacity,
                tooltip=[
                    "Team:N",
                    alt.Tooltip("WinPct:Q", format=".3f"),
                    alt.Tooltip("AdjPyth:Q", format=".3f"),
                    alt.Tooltip("Elo:Q", format=".1f"),
                    "Games:Q",
                    alt.Tooltip("SOS:Q", format=".3f"),
                    "WinRank:Q",
                    "AdjRank:Q",
                    "EloRank:Q",
                ],
            )

            labels = alt.Chart(profile_df[profile_df["IsFocus"]]).mark_text(
                align="left", baseline="bottom", dx=6, dy=-4, color=SEMANTIC_COLORS["neutral"]
            ).encode(
                x="WinPct:Q",
                y="AdjPyth:Q",
                text="Team:N",
            )

            chart = (points.add_params(hover_sel) + labels).properties(height=460)
            st.altair_chart(chart, use_container_width=True)

            gpg_df = profile_df.copy()
            gpg_df["GoalsFor"] = gpg_df["Team"].map(lambda t: stats[t]["gf"] / max(stats[t]["games"], 1))
            gpg_df["GoalsAgainst"] = gpg_df["Team"].map(lambda t: stats[t]["ga"] / max(stats[t]["games"], 1))
            st.subheader("Team Profile Space: Goals For vs Goals Against (inverted Y)")
            st.caption("Elo is encoded by size and SOS by color; lower Goals Against appears higher due inverted y-axis.")
            gpg_points = alt.Chart(gpg_df).mark_circle().encode(
                x=alt.X("GoalsFor:Q", title="Goals For / Game"),
                y=alt.Y("GoalsAgainst:Q", title="Goals Against / Game", scale=alt.Scale(reverse=True)),
                size=alt.Size(
                    "EloNorm:Q",
                    legend=alt.Legend(title=f"Elo ({elo_min:.0f}–{elo_max:.0f})"),
                    scale=alt.Scale(domain=[0, 1], range=[40, 900]),
                ),
                color=sos_color,
                opacity=base_opacity,
                tooltip=[
                    "Team:N",
                    alt.Tooltip("GoalsFor:Q", format=".2f"),
                    alt.Tooltip("GoalsAgainst:Q", format=".2f"),
                    alt.Tooltip("Elo:Q", format=".1f"),
                    alt.Tooltip("SOS:Q", format=".3f"),
                    "Games:Q",
                ],
            )
            gpg_labels = alt.Chart(gpg_df[gpg_df["IsFocus"]]).mark_text(
                align="left", baseline="bottom", dx=6, dy=-4, color=SEMANTIC_COLORS["neutral"]
            ).encode(
                x="GoalsFor:Q",
                y="GoalsAgainst:Q",
                text="Team:N",
            )
            st.altair_chart((gpg_points.add_params(hover_sel) + gpg_labels).properties(height=460), use_container_width=True)
        st.markdown("### Expert Fit")
        illpolo_order = parse_expert_order_text(illpolo_text)
        maxpreps_order = parse_expert_order_text(maxpreps_text)
        if not illpolo_order and not maxpreps_order:
            st.info("Paste Illpolo and/or MaxPreps orders in the sidebar to compute expert-fit metrics after each parameter tweak.")
        else:
            method_orders = {
                "Win": win_ord,
                "Pyth": py_ord,
                "AdjPyth": adj_ord,
                "Elo": elo_ord,
                "Ensemble": df_avg["Team"].tolist(),
            }
            expert_orders = {
                "Illpolo": illpolo_order,
                "MaxPreps": maxpreps_order,
            }
            report_rows = []
            for method_name, method_order in method_orders.items():
                row = {"Method": method_name}
                for expert_name, expert_order in expert_orders.items():
                    if expert_order:
                        fit = compute_expert_fit(method_order, expert_order, top_n=25)
                        row[f"{expert_name} MAE (Top25 overlap)"] = round(fit["mean_abs_rank_error"], 2) if fit["mean_abs_rank_error"] is not None else None
                        row[f"{expert_name} Top10 overlap"] = fit["top10_overlap_count"]
                        row[f"{expert_name} Top25 overlap"] = fit["top25_overlap_count"]
                report_rows.append(row)
            st.dataframe(pd.DataFrame(report_rows))
            with st.expander("Detailed per-team rank deltas (Top 25 overlap only)", expanded=False):
                for expert_name, expert_order in expert_orders.items():
                    if not expert_order:
                        continue
                    st.markdown(f"#### {expert_name}")
                    for method_name, method_order in method_orders.items():
                        fit = compute_expert_fit(method_order, expert_order, top_n=25)
                        st.markdown(f"**{method_name}**")
                        if fit["deltas"]:
                            st.dataframe(pd.DataFrame(fit["deltas"]))
                        else:
                            st.caption("No Top 25 overlap with this expert list.")
    
    if selected_section == "Sectionals":
        st.subheader("Sectional Rankings")
        
        # Display sectional strength rankings
        st.markdown("### Sectional Strength Rankings")
        df_strength = pd.DataFrame({
            'Sectional': sectional_order,
            'Strength': [f"{sum(stats[t]['win_pct'] for t in sectional_rankings[s] if t in stats) / len([t for t in sectional_rankings[s] if t in stats]):.3f}" for s in sectional_order]
        })
        st.dataframe(df_strength)
        
        # Display individual sectional rankings with detailed breakdowns
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
            
            # Detailed breakdown for each team
            st.markdown("#### Detailed Seeding Analysis")
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
                    
                    # Display factors with updated weights
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Team Momentum" if not is_deep_dive else "H2H (45%)", f"{h2h_score:.3f}", help="How well this team has done in direct matchups that matter for seeding.")
                    with col2:
                        st.metric("Scoring Strength" if not is_deep_dive else "Common Opp (Non-Sectional, 45%)", f"{common_wins_weighted:.1f}/{common_games}", f"{common_win_pct:.3f}", help="Performance against shared opponents helps estimate overall strength.")
                    with col3:
                        st.metric("Schedule Difficulty" if not is_deep_dive else "Win % (10%)", f"{win_pct:.3f}", help="Teams are adjusted for the quality and difficulty of opponents faced.")
                    with col4:
                        penalties = []
                        if game_penalty < 1.0:
                            penalties.append(f"Games: {game_penalty:.2f}x")
                        if sectional_penalty < 1.0:
                            penalties.append(f"Sectional: {sectional_penalty:.2f}x")
                        st.metric("Penalties", "None" if not penalties else ", ".join(penalties))
                    
                    # Combined score
                    st.metric("Combined Score", f"{breakdown['combined_score']:.3f}")
                    
                    # Head-to-head details
                    st.markdown("##### Head-to-Head Details")
                    if breakdown["h2h_details"]:
                        st.dataframe(pd.DataFrame(breakdown["h2h_details"]))
                    else:
                        st.write("No head-to-head games played")
                    
                    # Common opponents details
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
