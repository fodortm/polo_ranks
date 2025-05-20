import pandas as pd
import os
import re
import math
import numpy as np # Added for vectorized operations
from collections import defaultdict
from functools import cmp_to_key
import streamlit as st
import altair as alt

# ---------------- Constants ---------------- #
SCORES_CSV = "scores.csv"
PATTERN = re.compile(r"^(.+?)\s+(\d+)(?:\s*\(OT\))?\s+(.+?)\s+(\d+)")

# ---------------- Parsing ---------------- #
def _parse_line(line):
    games = []
    raw = line.strip()
    if not raw or raw.lower().startswith("championship") or " vs " in raw.lower():
        return games
    # Default win notation "d." => placeholder None scores
    if re.search(r"\s+d\.\s+", raw, flags=re.IGNORECASE):
        a, b = re.split(r"\s+d\.\s+", raw, maxsplit=1, flags=re.IGNORECASE)
        games.append({"team1": a.strip(), "score1": None, "team2": b.strip(), "score2": None})
        return games
    m = PATTERN.match(raw)
    if m:
        t1, s1, t2, s2 = m.group(1).strip(), int(m.group(2)), m.group(3).strip(), int(m.group(4))
        games.append({"team1": t1, "score1": s1, "team2": t2, "score2": s2})
    return games

def parse_scores_text(text):
    records = []
    for line in text.splitlines():
        records.extend(_parse_line(line))
    return pd.DataFrame(records)

# ---------------- I/O ---------------- #
@st.cache_data
def load_scores():
    try:
        return pd.read_csv(SCORES_CSV)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["team1","score1","team2","score2"])

def save_scores(df):
    df.to_csv(SCORES_CSV, index=False)

def update_scores(existing, new_df):
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    save_scores(combined)
    return combined

# ---------------- Inference ---------------- #
def infer_default_scores(games_df, stats_dict): # Renamed stats to stats_dict for clarity
    df = games_df.copy()
    mask = df['score1'].isna()

    if mask.sum() == 0:
        return df

    def calculate_inferred_scores(row, all_stats):
        t1, t2 = row['team1'], row['team2']
        
        st1 = all_stats.get(t1)
        st2 = all_stats.get(t2)

        avg1, avg2 = 1, 1 # Default averages

        if st1 and st1.get('games', 0) > 0 and st2 and st2.get('games', 0) > 0:
            # Calculate averages only if both teams have stats and games played
            avg1 = (st1['gf'] / st1['games'] + st2['ga'] / st2['games']) / 2
            avg2 = (st2['gf'] / st2['games'] + st1['ga'] / st1['games']) / 2
        
        loser_avg, winner_avg = sorted([avg1, avg2])
        loser_score = int(math.floor(loser_avg + 0.5))
        winner_score = int(math.floor(winner_avg + 0.5)) + 1
        
        # team1 is always the winner by "d." notation, which leads to NaN scores
        return pd.Series({'score1': winner_score, 'score2': loser_score})

    inferred_scores_df = df[mask].apply(lambda row: calculate_inferred_scores(row, stats_dict), axis=1)
    
    # Update the DataFrame with the inferred scores
    # Ensure we only try to assign if inferred_scores_df is not empty
    if not inferred_scores_df.empty:
        df.loc[mask, ['score1', 'score2']] = inferred_scores_df[['score1', 'score2']].values
        # Ensure scores are integers after update
        df['score1'] = df['score1'].astype(int)
        df['score2'] = df['score2'].astype(int)

    return df

# ---------------- Stats ---------------- #
def compute_stats(games_df):
    # Ensure scores are integers
    games = games_df.copy()
    games['score1'] = games['score1'].astype(int)
    games['score2'] = games['score2'].astype(int)

    # Initialize stats and h2h
    stats = {}
    h2h = defaultdict(lambda: {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
    
    # Get unique teams
    unique_teams = pd.concat([games['team1'], games['team2']]).unique()
    for team in unique_teams:
        stats[team] = {'wins': 0, 'losses': 0, 'ties': 0, 'gf': 0, 'ga': 0, 'games': 0, 'opponents': []}

    # Calculate games played, goals for (gf), and goals against (ga)
    for team_col, score_col, opponent_score_col in [('team1', 'score1', 'score2'), ('team2', 'score2', 'score1')]:
        grouped = games.groupby(team_col)
        for team, group_data in grouped:
            if team not in stats: # Should not happen if unique_teams is comprehensive
                stats[team] = {'wins': 0, 'losses': 0, 'ties': 0, 'gf': 0, 'ga': 0, 'games': 0, 'opponents': []}
            stats[team]['games'] += len(group_data)
            stats[team]['gf'] += group_data[score_col].sum()
            stats[team]['ga'] += group_data[opponent_score_col].sum()
            
            # Collect opponents
            opponent_col_name = 'team2' if team_col == 'team1' else 'team1'
            stats[team]['opponents'].extend(group_data[opponent_col_name].tolist())

    # Calculate wins, losses, ties
    games['t1_wins'] = games['score1'] > games['score2']
    games['t2_wins'] = games['score2'] > games['score1']
    games['ties'] = games['score1'] == games['score2']

    wins_t1 = games[games['t1_wins']].groupby('team1').size()
    losses_t1 = games[games['t2_wins']].groupby('team1').size() # t1 loses if t2 wins
    ties_t1 = games[games['ties']].groupby('team1').size()

    wins_t2 = games[games['t2_wins']].groupby('team2').size()
    losses_t2 = games[games['t1_wins']].groupby('team2').size() # t2 loses if t1 wins
    ties_t2 = games[games['ties']].groupby('team2').size()

    for team in unique_teams:
        stats[team]['wins'] = wins_t1.get(team, 0) + wins_t2.get(team, 0)
        stats[team]['losses'] = losses_t1.get(team, 0) + losses_t2.get(team, 0)
        stats[team]['ties'] = ties_t1.get(team, 0) + ties_t2.get(team, 0)

    # Head-to-Head (h2h) stats using itertuples
    for row in games.itertuples(index=False):
        t1, s1, t2, s2 = row.team1, row.score1, row.team2, row.score2

        # Update stats for (t1, t2)
        h2h[(t1, t2)]['games'] += 1
        h2h[(t1, t2)]['gf'] += s1
        h2h[(t1, t2)]['ga'] += s2
        if s1 > s2:
            h2h[(t1, t2)]['wins'] += 1
        
        # Update stats for (t2, t1)
        h2h[(t2, t1)]['games'] += 1
        h2h[(t2, t1)]['gf'] += s2
        h2h[(t2, t1)]['ga'] += s1
        if s2 > s1:
            h2h[(t2, t1)]['wins'] += 1
            
    # Final calculations: win_pct and gd
    for team_name, team_data in stats.items():
        total_games = team_data['wins'] + team_data['losses'] + team_data['ties']
        # Ensure 'games' count is consistent, though it should be from prior calculation
        # This also implies that 'games' in stats[team] should be the sum of W+L+T
        # If stats[team]['games'] was calculated differently (e.g. counting rows where team appears),
        # it might not equal W+L+T if there are data inconsistencies (e.g. a game listed but no score)
        # For win_pct, using W+L+T is standard.
        if total_games > 0:
            team_data['win_pct'] = team_data['wins'] / total_games
        else:
            team_data['win_pct'] = 0
        team_data['gd'] = team_data['gf'] - team_data['ga']
        # Deduplicate opponents list
        if 'opponents' in team_data: # Should always be true
            team_data['opponents'] = list(set(team_data['opponents']))


    return stats, h2h

# ---------------- Metrics ---------------- #
def compute_sos(stats):
    sos={}
    for t,st in stats.items():
        opps=st['opponents']
        sos[t]=sum(stats[o]['win_pct'] for o in opps)/len(opps) if opps else 0
    return sos

def compute_pythag(stats,exp=2):
    p={}
    for t,st in stats.items():
        gf,ga=st['gf'],st['ga']
        p[t]=gf**exp/(gf**exp+ga**exp) if gf+ga>0 else 0
    return p

# Logistic scaling function
def logistic(x, k, x0):
    # Use np.exp for vectorized operations if x is a Series/array
    # If x is a scalar, math.exp would be fine, but np.exp handles both.
    return 1 / (1 + np.exp(-k * (x - x0)))

# Adjusted Pythagorean with logistic blend
def compute_adjusted_pythag(games_df, stats_dict, exp=2, k=10, x0=0.5): # Renamed for clarity
    # Ensure scores are numeric
    games = games_df.copy()
    games['score1'] = pd.to_numeric(games['score1'], errors='coerce').fillna(0).astype(int)
    games['score2'] = pd.to_numeric(games['score2'], errors='coerce').fillna(0).astype(int)

    # 1. Prepare Game Data with Opponent Stats
    win_pct_map = {t: st.get('win_pct', 0.0) for t, st in stats_dict.items()} # Default to 0.0 if 'win_pct' missing
    
    # Team1's perspective
    games_t1_perspective = games[['team1', 'score1', 'team2', 'score2']].copy()
    games_t1_perspective.rename(columns={'team1': 'me', 'score1': 'ms', 'team2': 'opp', 'score2': 'os_'}, inplace=True)
    games_t1_perspective['opp_win_pct'] = games_t1_perspective['opp'].map(win_pct_map).fillna(0.0) # FillNA for teams not in stats_dict

    # Team2's perspective
    games_t2_perspective = games[['team2', 'score2', 'team1', 'score1']].copy()
    games_t2_perspective.rename(columns={'team2': 'me', 'score2': 'ms', 'team1': 'opp', 'score1': 'os_'}, inplace=True)
    games_t2_perspective['opp_win_pct'] = games_t2_perspective['opp'].map(win_pct_map).fillna(0.0) # FillNA

    game_perspectives_df = pd.concat([games_t1_perspective, games_t2_perspective], ignore_index=True)

    # 2. Calculate Expected Goals (E_ms, E_os) Vectorized
    team_games_map = {t: st.get('games', 0) for t, st in stats_dict.items()}
    team_gf_map = {t: st.get('gf', 0) for t, st in stats_dict.items()}
    team_ga_map = {t: st.get('ga', 0) for t, st in stats_dict.items()}

    game_perspectives_df['me_games'] = game_perspectives_df['me'].map(team_games_map).fillna(0)
    game_perspectives_df['me_gf'] = game_perspectives_df['me'].map(team_gf_map).fillna(0)
    game_perspectives_df['me_ga'] = game_perspectives_df['me'].map(team_ga_map).fillna(0)
    
    game_perspectives_df['opp_games'] = game_perspectives_df['opp'].map(team_games_map).fillna(0)
    game_perspectives_df['opp_gf'] = game_perspectives_df['opp'].map(team_gf_map).fillna(0)
    game_perspectives_df['opp_ga'] = game_perspectives_df['opp'].map(team_ga_map).fillna(0)

    # Calculate E_ms and E_os, handling division by zero
    me_games = game_perspectives_df['me_games']
    me_gf = game_perspectives_df['me_gf']
    me_ga = game_perspectives_df['me_ga']
    opp_games = game_perspectives_df['opp_games']
    opp_gf = game_perspectives_df['opp_gf']
    opp_ga = game_perspectives_df['opp_ga']

    # Default E_ms and E_os to 1, then update where possible
    game_perspectives_df['E_ms'] = 1.0
    game_perspectives_df['E_os'] = 1.0

    valid_stats_mask = (me_games > 0) & (opp_games > 0)
    game_perspectives_df.loc[valid_stats_mask, 'E_ms'] = \
        (me_gf[valid_stats_mask] / me_games[valid_stats_mask] + opp_ga[valid_stats_mask] / opp_games[valid_stats_mask]) / 2
    game_perspectives_df.loc[valid_stats_mask, 'E_os'] = \
        (opp_gf[valid_stats_mask] / opp_games[valid_stats_mask] + me_ga[valid_stats_mask] / me_games[valid_stats_mask]) / 2

    # 3. Calculate Strength Factor (s) Vectorized
    game_perspectives_df['s'] = logistic(game_perspectives_df['opp_win_pct'], k, x0)

    # 4. Calculate Increments for adj_ms and adj_os
    game_perspectives_df['adj_ms_increment'] = game_perspectives_df['E_ms'] + \
        (game_perspectives_df['ms'] - game_perspectives_df['E_ms']) * game_perspectives_df['s']
    game_perspectives_df['adj_os_increment'] = game_perspectives_df['E_os'] + \
        (game_perspectives_df['os_'] - game_perspectives_df['E_os']) * game_perspectives_df['s']
        
    # 5. Sum Increments per Team
    summed_adj = game_perspectives_df.groupby('me')[['adj_ms_increment', 'adj_os_increment']].sum()

    # 6. Final Adjusted Pythagorean Calculation
    adj = {team: 0.0 for team in stats_dict.keys()} # Initialize for all teams in original stats

    for team_name, row in summed_adj.iterrows():
        if team_name in stats_dict: # Ensure team is in the original stats dictionary
            g = stats_dict[team_name].get('games', 0)
            agf = row['adj_ms_increment'] / g if g > 0 else 0
            aga = row['adj_os_increment'] / g if g > 0 else 0
            
            if agf + aga > 0:
                adj[team_name] = agf**exp / (agf**exp + aga**exp)
            else:
                adj[team_name] = 0.0
    
    return adj

# ---------------- Rankings ---------------- #
def rank_win_pct(stats,h2h):
    def cmp(a,b):
        if stats[a]['win_pct']!=stats[b]['win_pct']:
            return -1 if stats[a]['win_pct']>stats[b]['win_pct'] else 1
        h=h2h.get((a,b),{'wins':0,'games':0})
        if h['games']:
            p=h['wins']/h['games']
            if p!=0.5: return -1 if p>0.5 else 1
        return (stats[b]['gd'] - stats[a]['gd'])
    return sorted(stats.keys(),key=cmp_to_key(cmp))
def rank_pythag(stats,p):
    return sorted(stats.keys(),key=lambda t:p[t],reverse=True)

def rank_adj_pyth(stats,games,h2h,k=10,x0=0.5):
    vals = compute_adjusted_pythag(games,stats,k=k,x0=x0)
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
def compute_elo(games,initial=1500,k=32):
    teams=set(games['team1']).union(games['team2'])
    R={t:initial for t in teams}
    for _,r in games.iterrows():
        a,b,sa,sb=r.team1,r.team2,r.score1,r.score2
        ea=1/(1+10**((R[b]-R[a])/400)); eb=1-ea
        aa,ab = (1,0) if sa>sb else ((0,1) if sb>sa else (0.5,0.5))
        R[a]+=k*(aa-ea); R[b]+=k*(ab-eb)
    return R
def rank_elo(stats,elo):
    return sorted(stats.keys(),key=lambda t:elo[t],reverse=True)

# ---------------- Main Data Calculation ---------------- #
@st.cache_data
def get_main_data(raw_games_df, k_slider, x0_slider):
    scored_games = raw_games_df.dropna(subset=['score1'])
    initial_stats, _ = compute_stats(scored_games)
    all_teams = set(raw_games_df['team1']).union(raw_games_df['team2'])
    for t in all_teams:
        if t not in initial_stats:
            initial_stats[t] = {'wins':0,'losses':0,'ties':0,'gf':0,'ga':0,'games':0,'opponents':[]}
    games_inferred = infer_default_scores(raw_games_df, initial_stats)
    stats,h2h = compute_stats(games_inferred)
    sos = compute_sos(stats)
    py = compute_pythag(stats)
    adj_vals = compute_adjusted_pythag(games_inferred, stats, k=k_slider, x0=x0_slider)
    adj_ord_intermediate, _ = rank_adj_pyth(stats, games_inferred, h2h, k=k_slider, x0=x0_slider) # Renamed to avoid conflict
    elo = compute_elo(games_inferred)

    # Pre-calculate team_records_vs_all_opps for use in sectional rankings and profile tab
    team_records_vs_all_opps_dict = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0}))
    for r_game in games_inferred.itertuples(index=False): # Changed variable name to avoid conflict
        s1, s2 = int(r_game.score1), int(r_game.score2)
        
        # Team1 vs Team2
        record_t1_vs_t2 = team_records_vs_all_opps_dict[(r_game.team1, r_game.team2)]
        record_t1_vs_t2['games'] += 1
        record_t1_vs_t2['gf'] += s1
        record_t1_vs_t2['ga'] += s2
        if s1 > s2:
            record_t1_vs_t2['wins'] += 1
        
        # Team2 vs Team1
        record_t2_vs_t1 = team_records_vs_all_opps_dict[(r_game.team2, r_game.team1)]
        record_t2_vs_t1['games'] += 1
        record_t2_vs_t1['gf'] += s2
        record_t2_vs_t1['ga'] += s1
        if s2 > s1:
            record_t2_vs_t1['wins'] += 1
            
    return games_inferred, stats, h2h, sos, py, adj_vals, adj_ord_intermediate, elo, team_records_vs_all_opps_dict

# ---------------- Sectional Rankings ---------------- #

# Helper function to get detailed score components for a team in a sectional
def get_team_sectional_score_components(
    current_team_name,
    sectional_team_names_in_stats, # list of team names in this specific sectional that are in main_stats
    main_stats_dict,
    sos_data_dict,
    h2h_overall_dict, # Global H2H data
    team_records_vs_all_opps_dict, # Precalculated: (t1, t2) -> {wins, games, gf, ga}
    avg_games_played_by_all_teams_overall, # Precalculated: average games played by any team
    avg_win_pct_of_all_teams_overall, # Precalculated: average win_pct of any team
    avg_intra_sectional_games_for_this_sectional # Precalculated: avg games played by a team in this sectional vs other teams in this sectional
):
    # Initialize results dictionary
    result = {
        'h2h_score_val': 0.0, 'h2h_details_list': [],
        'common_wins_weighted_val': 0.0, 'common_games_val': 0, 'common_win_pct_val': 0.0,
        'non_sectional_common_opp_details_list': [], 'sectional_common_opp_details_list': [],
        'game_penalty_val': 1.0, 'sectional_penalty_val': 1.0,
        'current_team_overall_win_pct': main_stats_dict.get(current_team_name, {}).get('win_pct', 0.0),
        'final_combined_score': 0.0
    }

    if current_team_name not in main_stats_dict:
        return result # Should not happen if sectional_team_names_in_stats is filtered

    # --- Penalties ---
    # Game Penalty (based on overall games played by current_team vs avg games played by teams in this sectional)
    games_played_by_sectional_teams = [main_stats_dict[t]['games'] for t in sectional_team_names_in_stats if t in main_stats_dict]
    avg_games_overall_for_teams_in_sectional = sum(games_played_by_sectional_teams) / len(games_played_by_sectional_teams) if games_played_by_sectional_teams else 0
    
    current_team_games = main_stats_dict[current_team_name]['games']
    if avg_games_overall_for_teams_in_sectional > 0 and current_team_games < avg_games_overall_for_teams_in_sectional * 0.8: # Threshold: 80%
        result['game_penalty_val'] = (current_team_games / avg_games_overall_for_teams_in_sectional) ** 2
    elif avg_games_overall_for_teams_in_sectional == 0 and current_team_games > 0 : # Team played, but sectional avg is 0 (e.g. only team in sectional)
        result['game_penalty_val'] = 1.0


    # Sectional Game Play Penalty
    sectional_games_played_by_current_team = 0
    for opp_in_sectional in sectional_team_names_in_stats:
        if opp_in_sectional == current_team_name:
            continue
        sectional_games_played_by_current_team += team_records_vs_all_opps_dict.get((current_team_name, opp_in_sectional), {}).get('games', 0)

    if sectional_games_played_by_current_team == 0:
        result['sectional_penalty_val'] = 0.85  # 15% penalty if no intra-sectional games
    elif avg_intra_sectional_games_for_this_sectional > 0 and sectional_games_played_by_current_team < avg_intra_sectional_games_for_this_sectional * 0.7:
        result['sectional_penalty_val'] = 0.85 + (0.15 * (sectional_games_played_by_current_team / (avg_intra_sectional_games_for_this_sectional * 0.7)))
    
    # --- H2H Component (vs teams within the current sectional) ---
    h2h_scores_list_for_avg = []
    for opp_in_sectional in sectional_team_names_in_stats:
        if opp_in_sectional == current_team_name or opp_in_sectional not in main_stats_dict:
            continue

        h2h_direct_record = h2h_overall_dict.get((current_team_name, opp_in_sectional), {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
        if h2h_direct_record['games'] > 0:
            opp_strength = main_stats_dict[opp_in_sectional]['win_pct']
            opp_sos = sos_data_dict.get(opp_in_sectional, 0.0)
            
            sos_multiplier = 1.0 + ((opp_sos - 0.5) * 2.0) # typical 0.8 to 1.2
            sos_multiplier *= 1.1 # Sectional opponent boost

            adjusted_strength = opp_strength * sos_multiplier
            win_pct_vs_opp = h2h_direct_record['wins'] / h2h_direct_record['games']
            
            goal_diff_factor = 1.0
            if h2h_direct_record['wins'] > 0 and (h2h_direct_record['games'] - h2h_direct_record['wins']) > 0: # if series is split
                gd = h2h_direct_record['gf'] - h2h_direct_record['ga']
                goal_diff_factor = 1.0 + (gd * 0.05)

            current_h2h_score = win_pct_vs_opp * (adjusted_strength / (avg_win_pct_of_all_teams_overall if avg_win_pct_of_all_teams_overall > 0 else 1.0)) * goal_diff_factor
            h2h_scores_list_for_avg.append(current_h2h_score)
            result['h2h_details_list'].append({
                'Opponent': opp_in_sectional,
                'Record': f"{h2h_direct_record['wins']}-{h2h_direct_record['games']-h2h_direct_record['wins']}",
                'Win %': f"{win_pct_vs_opp:.3f}",
                'Opp Win %': f"{opp_strength:.3f}", 'Opp SOS': f"{opp_sos:.3f}",
                'SOS Mult': f"{sos_multiplier:.2f}x", 'Adj Strength': f"{adjusted_strength:.3f}",
                'GD Factor': f"{goal_diff_factor:.2f}x", 'Weighted Score': f"{current_h2h_score:.3f}"
            })
    if h2h_scores_list_for_avg:
        result['h2h_score_val'] = sum(h2h_scores_list_for_avg) / len(h2h_scores_list_for_avg)

    # --- Common Opponents Component ---
    my_opponents_overall = set(main_stats_dict[current_team_name].get('opponents', []))
    
    for common_opp_candidate in my_opponents_overall:
        if common_opp_candidate not in main_stats_dict: continue # common opp must have stats

        record_vs_common_opp = team_records_vs_all_opps_dict.get((current_team_name, common_opp_candidate))
        if not record_vs_common_opp or record_vs_common_opp['games'] == 0:
            continue

        wins_vs_co = record_vs_common_opp['wins']
        games_vs_co = record_vs_common_opp['games']
        
        co_strength = main_stats_dict[common_opp_candidate]['win_pct']
        co_sos = sos_data_dict.get(common_opp_candidate, 0.0)
        
        sos_multiplier = 1.0 + ((co_sos - 0.5) * 2.0)
        adj_co_strength = co_strength * sos_multiplier
        weight = 0.7 + (0.6 * (adj_co_strength / (avg_win_pct_of_all_teams_overall if avg_win_pct_of_all_teams_overall > 0 else 1.0)))
        
        result['common_wins_weighted_val'] += wins_vs_co * weight
        result['common_games_val'] += games_vs_co
        
        detail_item = {
            'Opponent': common_opp_candidate,
            'Record': f"{wins_vs_co}-{games_vs_co - wins_vs_co}",
            'Opp Win %': f"{co_strength:.3f}", 'Opp SOS': f"{co_sos:.3f}",
            'SOS Mult': f"{sos_multiplier:.2f}x", 'Adj Strength': f"{adj_co_strength:.3f}",
            'Weight': f"{weight:.1f}x", 'Weighted Wins': f"{wins_vs_co * weight:.1f}"
        }
        if common_opp_candidate in sectional_team_names_in_stats: # Already handled by H2H if it's an H2H opponent, this is for common opps in sectional not directly played
             # This logic might need refinement: common sectional opponents are typically those NOT directly played H2H in sectional
            if common_opp_candidate != current_team_name : # and not any(d['Opponent'] == common_opp_candidate for d in result['h2h_details_list']): # Avoid double counting H2H
                 result['sectional_common_opp_details_list'].append(detail_item)
        else:
            result['non_sectional_common_opp_details_list'].append(detail_item)

    if result['common_games_val'] > 0:
        result['common_win_pct_val'] = result['common_wins_weighted_val'] / result['common_games_val']

    # --- Final Score ---
    result['final_combined_score'] = (
        result['h2h_score_val'] * 0.45 +
        result['common_win_pct_val'] * 0.45 +
        result['current_team_overall_win_pct'] * 0.1
    ) * result['game_penalty_val'] * result['sectional_penalty_val']
    
    return result

@st.cache_data
def compute_sectional_rankings(main_stats_dict, h2h_overall_dict, games_inferred_df, sos_data_dict, team_records_vs_all_opps_dict): # Added team_records_vs_all_opps_dict
    sectionals = {
        "Barrington": ["Hersey", "Barrington", "Elk Grove", "Conant", "Hoffman Estates", "McHenry", "Fremd", "Palatine", "Meadows", "Schaumburg"],
        "Chicago (Lane)": ["Amundsen", "Jones-Payton", "Kenwood", "Lane", "Latin", "Senn", "St Ignatius", "Whitney Young"],
        "Elmhurst (York)": ["Morton", "Northside", "St Patrick", "Taft", "Westinghouse", "York", "Leyden", "Fenwick", "Oak Park", "STC"],
        "Glenview (GBS)": ["Maine West", "Evanston", "GBS", "Prospect", "GBN", "Maine East", "Maine South", "Niles West", "Loyola", "New Trier"],
        "LaGrange (Lyons)": ["Brother Rice", "Curie", "Kennedy", "Mt Carmel", "Solorio", "St Rita", "Lyons", "R-B", "Goode"],
        "Naperville (North)": ["Metea", "Waubonsie", "HC", "Lockport", "NC", "Neuqua", "NN", "Sandburg", "Shepard"],
        "New Lenox (LWW)": ["Bradley", "Chicago Ag", "Brooks", "H-F", "LWE", "Bremen", "LWC", "LWW", "Andrew"]
    }

    # team_records_vs_all_opps_dict is now passed as a parameter, so we don't calculate it here.

    # Pre-calculate average games played overall and average win_pct overall
    all_teams_with_stats = [t for t in main_stats_dict if main_stats_dict[t].get('games', 0) > 0]
    avg_games_played_by_all_teams_overall = sum(main_stats_dict[t]['games'] for t in all_teams_with_stats) / len(all_teams_with_stats) if all_teams_with_stats else 0
    avg_win_pct_of_all_teams_overall = sum(main_stats_dict[t]['win_pct'] for t in all_teams_with_stats) / len(all_teams_with_stats) if all_teams_with_stats else 0.5 # Default to 0.5 if no teams

    # Pre-calculate average intra-sectional games for each sectional
    avg_intra_sectional_games_map = {}
    for sec_name, sec_team_list_cfg in sectionals.items():
        sec_teams_in_stats = [t for t in sec_team_list_cfg if t in main_stats_dict]
        if not sec_teams_in_stats:
            avg_intra_sectional_games_map[sec_name] = 0
            continue
        
        total_intra_sec_games_sum = 0
        for t1_idx, t1_name in enumerate(sec_teams_in_stats):
            for t2_idx in range(t1_idx + 1, len(sec_teams_in_stats)):
                t2_name = sec_teams_in_stats[t2_idx]
                total_intra_sec_games_sum += team_records_vs_all_opps_dict.get((t1_name, t2_name), {}).get('games',0)
        # This is total unique games. Avg per team is more complex.
        # For now, let's use a simpler proxy: total intra-sectional games / number of teams in sectional.
        # Or, the average number of *such games a team in the sectional has played*.
        # Let's calculate sum of intra-sectional games for each team, then average that.
        games_played_by_each_team_in_sectional = []
        for team_in_sec in sec_teams_in_stats:
            current_team_intra_sec_games = 0
            for other_team_in_sec in sec_teams_in_stats:
                if team_in_sec == other_team_in_sec: continue
                current_team_intra_sec_games += team_records_vs_all_opps_dict.get((team_in_sec, other_team_in_sec),{}).get('games',0)
            games_played_by_each_team_in_sectional.append(current_team_intra_sec_games)
        
        if games_played_by_each_team_in_sectional:
            avg_intra_sectional_games_map[sec_name] = sum(games_played_by_each_team_in_sectional) / len(games_played_by_each_team_in_sectional)
        else:
            avg_intra_sectional_games_map[sec_name] = 0


    # Initialize 'sectional_details_by_name' for each team in main_stats_dict
    for team_name_in_stats in main_stats_dict:
        if 'sectional_details_by_name' not in main_stats_dict[team_name_in_stats]:
            main_stats_dict[team_name_in_stats]['sectional_details_by_name'] = {}

    sectional_rankings_map = {}

    def rank_teams_in_sectional_optimized(sectional_name_arg, teams_in_sectional_config, stats_arg, sos_arg, h2h_arg, team_records_arg, avg_intra_sec_games_arg):
        
        sectional_teams_actually_in_stats = [t for t in teams_in_sectional_config if t in stats_arg]
        if not sectional_teams_actually_in_stats:
            return []

        def get_score_for_sort(team_to_score):
            score_components = get_team_sectional_score_components(
                team_to_score,
                sectional_teams_actually_in_stats,
                stats_arg,
                sos_arg,
                h2h_arg,
                team_records_arg,
                avg_games_played_by_all_teams_overall, # from outer scope
                avg_win_pct_of_all_teams_overall, # from outer scope
                avg_intra_sec_games_arg
            )
            # Store the detailed components
            stats_arg[team_to_score]['sectional_details_by_name'][sectional_name_arg] = score_components
            return score_components['final_combined_score']

        ranked_teams = sorted(sectional_teams_actually_in_stats, key=get_score_for_sort, reverse=True)
        return ranked_teams

    for name, teams_list in sectionals.items():
        sectional_rankings_map[name] = rank_teams_in_sectional_optimized(
            name, teams_list, main_stats_dict, sos_data_dict, h2h_overall_dict, team_records_vs_all_opps_dict, avg_intra_sectional_games_map.get(name,0)
        )
    
    # Calculate sectional strength (remains the same, uses main_stats_dict)
    def get_sectional_strength(teams_in_ranking, stats_lookup):
        valid_teams = [t for t in teams_in_ranking if t in stats_lookup and 'win_pct' in stats_lookup[t]]
        if not valid_teams:
            return 0
        return sum(stats_lookup[t]['win_pct'] for t in valid_teams) / len(valid_teams)
    
    sectional_strengths = {name: get_sectional_strength(teams_ranked, main_stats_dict) for name, teams_ranked in sectional_rankings_map.items()}
    sectional_order = sorted(sectional_strengths.keys(), key=lambda x: sectional_strengths[x], reverse=True)
    
    return sectional_rankings_map, sectional_order
    # main_stats_dict is modified in-place, so no need to return it explicitly unless preferred

# ---------------- App ---------------- #
st.set_page_config(page_title="Polo Dashboard",layout="wide")

# Sidebar settings
st.sidebar.header("Data & Model Settings")
k = st.sidebar.slider("Logistic Steepness (k)", min_value=1, max_value=20, value=10)
x0 = st.sidebar.slider("Logistic Midpoint (x0)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Load & parse
raw_games = load_scores()
uploader = st.sidebar.file_uploader("Upload scores .txt", type="txt")
if uploader:
    # When new data is uploaded, clear all caches and re-parse.
    st.cache_data.clear()
    new_df = parse_scores_text(uploader.getvalue().decode())
    if not new_df.empty:
        raw_games = update_scores(raw_games, new_df)

# Get main data using the new cached function
games_inferred, stats, h2h, sos, py, adj_vals, adj_ord, elo, team_records_vs_all_opps = get_main_data(raw_games, k, x0) # Added team_records_vs_all_opps

# Compute sectional rankings
sectional_rankings, sectional_order = compute_sectional_rankings(stats, h2h, games_inferred, sos, team_records_vs_all_opps) # Pass team_records_vs_all_opps

# Orders & filters
win_ord = rank_win_pct(stats,h2h)
py_ord = rank_pythag(stats,py)
elo_ord = rank_elo(stats,elo)
maxg = max(st['games'] for st in stats.values()) if stats else 0
thr = maxg/2
win_ord = [t for t in win_ord if stats[t]['games']>=thr]
py_ord  = [t for t in py_ord  if stats[t]['games']>=thr]
adj_ord = [t for t in adj_ord if stats[t]['games']>=thr]
elo_ord = [t for t in elo_ord if stats[t]['games']>=thr]
teams    = sorted(stats.keys())

# Team profile selection
st.sidebar.header("Team Profile")
te=st.sidebar.selectbox("Select Team",teams,index=teams.index("Loyola") if "Loyola" in teams else 0)
# Compute individual ranks
ranks = {}
ranks['win']  = win_ord.index(te)+1 if te in win_ord else None
ranks['py']   = py_ord.index(te)+1 if te in py_ord else None
ranks['adj']  = adj_ord.index(te)+1 if te in adj_ord else None
ranks['elo']  = elo_ord.index(te)+1 if te in elo_ord else None
ranks_list   = [v for v in ranks.values() if v]
r_avg = round(sum(ranks_list)/len(ranks_list),2) if ranks_list else None

# Tabs & content
tabs = st.tabs(["Profile","Win%","Pythag","AdjPyth","Elo","Avg","Sectionals"])

# Profile tab
with tabs[0]:
    st.subheader(f"Profile: {te}")
    st.table(pd.DataFrame.from_dict({
        'GPG For':f"{stats[te]['gf']/stats[te]['games']:.2f}",
        'GPG Against':f"{stats[te]['ga']/stats[te]['games']:.2f}",
        'GD/Game':f"{(stats[te]['gf']-stats[te]['ga'])/stats[te]['games']:.2f}",
        'Win %':f"{stats[te]['win_pct']:.3f}",
        'SOS':f"{sos[te]:.3f}",
        'Rank Win%':ranks['win'],'Rank Pythag':ranks['py'],
        'Rank Adj':ranks['adj'],'Rank Elo':ranks['elo'],'Avg':r_avg
    },orient='index',columns=['Value']))
    opp=st.selectbox("Compare vs",[t for t in teams if t!=te])
    h = h2h.get((te,opp),{'wins':0,'games':0})
    st.markdown(f"**H2H**: {h['wins']}-{h['games']-h['wins']} in {h['games']} games")
    st.write(f"{opp} Ranks: Win% #{win_ord.index(opp)+1 if opp in win_ord else '-'} (SOS {sos[opp]:.3f}), "
             f"Pyth #{py_ord.index(opp)+1 if opp in py_ord else '-'} (SOS {sos[opp]:.3f}), "
             f"AdjPyth #{adj_ord.index(opp)+1 if opp in adj_ord else '-'} (SOS {sos[opp]:.3f}), "
             f"Elo #{elo_ord.index(opp)+1 if opp in elo_ord else '-'}")
    st.markdown("**Common Opponents**")
    com = set(stats.get(te, {}).get('opponents', [])) & set(stats.get(opp, {}).get('opponents', []))
    if com:
        dfc = []
        for c_opp in com: # Renamed to avoid conflict with 'c' in st.columns
            te_vs_c_record = team_records_vs_all_opps.get((te, c_opp), {'wins': 0, 'games': 0})
            opp_vs_c_record = team_records_vs_all_opps.get((opp, c_opp), {'wins': 0, 'games': 0})

            te_wins = te_vs_c_record['wins']
            te_games = te_vs_c_record['games']
            te_losses = te_games - te_wins

            opp_wins = opp_vs_c_record['wins']
            opp_games = opp_vs_c_record['games']
            opp_losses = opp_games - opp_wins
            
            dfc.append({
                'Opp': c_opp,
                f'{te} W': te_wins,
                f'{te} L': te_losses,
                f'{opp} W': opp_wins,
                f'{opp} L': opp_losses
            })
        
        if dfc:
            df_common = pd.DataFrame(dfc)
            st.dataframe(df_common)
        else:
            st.write("No common opponent data processed.") # Should be covered by outer 'if com:'
    else:
        st.write("No common opponents.")

    st.markdown("**Full Schedule**")
    mask_te_games = (games_inferred['team1'] == te) | (games_inferred['team2'] == te)
    te_games_df = games_inferred[mask_te_games].copy()

    if not te_games_df.empty:
        # Determine Opponent
        te_games_df['Opp'] = np.where(te_games_df['team1'] == te, te_games_df['team2'], te_games_df['team1'])
        
        # Determine Scored and Allowed
        te_games_df['Scored'] = np.where(te_games_df['team1'] == te, te_games_df['score1'], te_games_df['score2'])
        te_games_df['Allowed'] = np.where(te_games_df['team1'] == te, te_games_df['score2'], te_games_df['score1'])
        
        # Ensure scores are integers for display (they should be, but defensive)
        te_games_df['Scored'] = te_games_df['Scored'].astype(int)
        te_games_df['Allowed'] = te_games_df['Allowed'].astype(int)

        st.dataframe(te_games_df[['Opp', 'Scored', 'Allowed']])
    else:
        st.write("No games played by this team.")

# Win % tab
with tabs[1]:
    st.subheader("Rankings by Win %")
    df_win=pd.DataFrame({'Team':win_ord,
                         'Win %':[f"{stats[t]['win_pct']:.3f}" for t in win_ord],
                         'SOS':[f"{sos[t]:.3f}" for t in win_ord]})
    st.dataframe(df_win)

# Pythagorean tab
with tabs[2]:
    st.subheader("Rankings by Pythagorean")
    df_py=pd.DataFrame({'Team':py_ord,
                        'Exp %':[f"{py[t]:.3f}" for t in py_ord],
                        'SOS':[f"{sos[t]:.3f}" for t in py_ord]})
    st.dataframe(df_py)

# Adjusted Pythagorean tab
with tabs[3]:
    st.subheader("Rankings by Adjusted Pythagorean")
    df_adj=pd.DataFrame({'Team':adj_ord,
                         'AdjPyth %':[f"{adj_vals[t]:.3f}" for t in adj_ord],
                         'SOS':[f"{sos[t]:.3f}" for t in adj_ord]})
    st.dataframe(df_adj)
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

# Elo tab
with tabs[4]:
    st.subheader("Rankings by Elo")
    df_elo=pd.DataFrame({'Team':elo_ord,
                         'Elo':[f"{elo[t]:.1f}" for t in elo_ord],
                         'SOS':[f"{sos[t]:.3f}" for t in elo_ord]})
    st.dataframe(df_elo)

# Average composite tab
with tabs[5]:
    st.subheader("Rankings by Avg Composite")
    comp=[(t,
           win_ord.index(t)+1, py_ord.index(t)+1,
           adj_ord.index(t)+1, elo_ord.index(t)+1,
           round((win_ord.index(t)+1 + py_ord.index(t)+1 + adj_ord.index(t)+1 + elo_ord.index(t)+1)/4,2),
           sos[t])
          for t in teams if stats[t]['games']>=thr]
    df_avg=pd.DataFrame(comp,columns=['Team','Win','Pyth','AdjPyth','Elo','Avg','SOS']).sort_values('Avg')
    st.dataframe(df_avg)

# Sectionals tab
with tabs[6]:
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
            with st.expander(f"{team} - Detailed Seeding Analysis"):
                # Retrieve pre-calculated details
                sectional_details_dict = stats.get(team, {}).get('sectional_details_by_name', {}).get(sectional, {})

                if not sectional_details_dict:
                    st.write("Detailed analysis not available for this team in this sectional.")
                    continue

                # Display factors with updated weights
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("H2H (45%)", f"{sectional_details_dict.get('h2h_score_val', 0.0):.3f}")
                with col2:
                    st.metric("Common Opp (45%)", 
                              f"{sectional_details_dict.get('common_wins_weighted_val', 0.0):.1f}/{sectional_details_dict.get('common_games_val', 0)}", 
                              f"{sectional_details_dict.get('common_win_pct_val', 0.0):.3f}")
                with col3:
                    st.metric("Win % (10%)", f"{sectional_details_dict.get('current_team_overall_win_pct', 0.0):.3f}")
                
                penalties_display = []
                game_pen_val = sectional_details_dict.get('game_penalty_val', 1.0)
                sec_pen_val = sectional_details_dict.get('sectional_penalty_val', 1.0)
                if game_pen_val < 1.0: penalties_display.append(f"Games: {game_pen_val:.2f}x")
                if sec_pen_val < 1.0: penalties_display.append(f"Sectional: {sec_pen_val:.2f}x")
                
                with col4:
                    st.metric("Penalties", "None" if not penalties_display else ", ".join(penalties_display))
                
                st.metric("Combined Score", f"{sectional_details_dict.get('final_combined_score', 0.0):.3f}")
                
                st.markdown("##### Head-to-Head Details (vs Sectional Teams)")
                h2h_df = pd.DataFrame(sectional_details_dict.get('h2h_details_list', []))
                if not h2h_df.empty: st.dataframe(h2h_df)
                else: st.write("No head-to-head games played against teams in this sectional.")
                
                st.markdown("##### Non-Sectional Common Opponents")
                non_sec_df = pd.DataFrame(sectional_details_dict.get('non_sectional_common_opp_details_list', []))
                if not non_sec_df.empty: st.dataframe(non_sec_df)
                else: st.write("No non-sectional common opponents or no games against them.")

                st.markdown("##### Sectional Common Opponents (Not H2H)")
                sec_common_df = pd.DataFrame(sectional_details_dict.get('sectional_common_opp_details_list', []))
                if not sec_common_df.empty: st.dataframe(sec_common_df)
                else: st.write("No other common opponents within this sectional (that were not direct H2H).")
