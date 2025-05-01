import pandas as pd
import os
import re
import math
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
def infer_default_scores(games_df, stats):
    df = games_df.copy()
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
    return df

# ---------------- Stats ---------------- #
def compute_stats(games):
    stats = {}
    h2h = defaultdict(lambda: {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
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
            h2h[(me,opp)]['gf'] += ms
            h2h[(me,opp)]['ga'] += os
        if s1>s2:
            stats[t1]['wins']+=1; stats[t2]['losses']+=1; h2h[(t1,t2)]['wins']+=1
        elif s2>s1:
            stats[t2]['wins']+=1; stats[t1]['losses']+=1; h2h[(t2,t1)]['wins']+=1
        else:
            stats[t1]['ties']+=1; stats[t2]['ties']+=1
        h2h[(t1,t2)]['games']+=1; h2h[(t2,t1)]['games']+=1
    for t,st in stats.items():
        tot=st['wins']+st['losses']+st['ties']
        st['win_pct']=st['wins']/tot if tot else 0
        st['gd']=st['gf']-st['ga']
    return stats,h2h

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
    return 1/(1 + math.exp(-k * (x - x0)))

# Adjusted Pythagorean with logistic blend
def compute_adjusted_pythag(games, stats, exp=2, k=10, x0=0.5):
    adj_ms = defaultdict(float)
    adj_os = defaultdict(float)
    for _,r in games.iterrows():
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
            adj_ms[me] += E_ms + (ms - E_ms) * s
            adj_os[me] += E_os + (os_ - E_os) * s
    adj = {}
    for t,st in stats.items():
        g = st['games']
        agf = adj_ms[t]/g if g else 0
        aga = adj_os[t]/g if g else 0
        adj[t] = agf**exp/(agf**exp + aga**exp) if agf+aga>0 else 0
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

# ---------------- Sectional Rankings ---------------- #
def compute_sectional_rankings(stats, h2h, games_inferred):
    sectionals = {
        "Barrington": ["Hersey", "Barrington", "Elk Grove", "Conant", "Hoffman Estates", "McHenry", "Fremd", "Palatine", "Meadows", "Schaumburg"],
        "Chicago (Lane)": ["Amundsen", "Jones-Payton", "Kenwood", "Lane", "Latin", "Senn", "St Ignatius", "Whitney Young"],
        "Elmhurst (York)": ["Morton", "Northside", "St Patrick", "Taft", "Westinghouse", "York", "Leyden", "Fenwick", "Oak Park", "STC"],
        "Glenview (GBS)": ["Maine West", "Evanston", "GBS", "Prospect", "GBN", "Maine East", "Maine South", "Niles West", "Loyola", "New Trier"],
        "LaGrange (Lyons)": ["Brother Rice", "Curie", "Kennedy", "Mt Carmel", "Solorio", "St Rita", "Lyons", "R-B", "Goode"],
        "Naperville (North)": ["Metea", "Waubonsie", "HC", "Lockport", "NC", "Neuqua", "NN", "Sandburg", "Shepard"],
        "New Lenox (LWW)": ["Bradley", "Chicago Ag", "Brooks", "H-F", "LWE", "Bremen", "LWC", "LWW", "Andrew"]
    }
    
    def rank_teams_in_sectional(teams, sectional_name):
        # First, create a matrix of head-to-head results
        h2h_matrix = {}
        for team1 in teams:
            if team1 not in stats:
                continue
            h2h_matrix[team1] = {}
            for team2 in teams:
                if team2 not in stats or team1 == team2:
                    continue
                h2h_record = h2h.get((team1, team2), {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
                h2h_matrix[team1][team2] = {
                    'wins': h2h_record['wins'],
                    'games': h2h_record['games'],
                    'win_pct': h2h_record['wins'] / h2h_record['games'] if h2h_record['games'] > 0 else 0
                }
        
        def get_team_score(team):
            if team not in stats:
                return float('-inf')
            
            # Calculate average games played in sectional
            valid_teams = [t for t in teams if t in stats]
            if not valid_teams:
                return 0
            avg_games = sum(stats[t]['games'] for t in valid_teams) / len(valid_teams)
            team_games = stats[team]['games']
            
            # Apply game count penalty if significantly below average
            game_penalty = 1.0
            if team_games < avg_games * 0.8:  # Adjusted threshold to 80% of average
                game_penalty = (team_games / avg_games) ** 2  # More severe penalty (increased exponent)
            
            # Get all opponents' win percentages and SOS for strength calculation
            all_opp_win_pcts = [stats[opp]['win_pct'] for opp in stats if opp != team]
            avg_opp_win_pct = sum(all_opp_win_pcts) / len(all_opp_win_pcts) if all_opp_win_pcts else 0.5
            
            # Calculate sectional games adjustment
            sectional_games = sum(1 for opp in valid_teams if opp != team and h2h.get((team, opp), {'games': 0})['games'] > 0)
            avg_sectional_games = sum(1 for t in valid_teams for opp in valid_teams 
                                   if t != opp and h2h.get((t, opp), {'games': 0})['games'] > 0) / len(valid_teams)
            
            # Middle ground approach for sectional penalty
            sectional_penalty = 1.0
            if sectional_games == 0:
                # If no sectional games, use a moderate penalty
                sectional_penalty = 0.85  # 15% penalty
            elif sectional_games < avg_sectional_games * 0.7:
                # Moderate penalty for teams with some sectional games
                sectional_penalty = 0.85 + (0.15 * (sectional_games / (avg_sectional_games * 0.7)))
            
            # Calculate head-to-head matrix for this team
            h2h_scores = []
            for opp in valid_teams:
                if opp == team:
                    continue
                h2h_record = h2h.get((team, opp), {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
                if h2h_record['games'] > 0:
                    opp_strength = stats[opp]['win_pct']
                    opp_sos = sos[opp]
                    # Adjusted SOS multiplier formula to penalize weak SOS and reward strong SOS
                    sos_multiplier = 1.0 + ((opp_sos - 0.5) * 2.0)  # 0.8 to 1.2 range for typical SOS
                    if opp in valid_teams:  # If opponent is in same sectional
                        sos_multiplier *= 1.1  # Boost multiplier for sectional opponents
                    adjusted_strength = opp_strength * sos_multiplier
                    
                    # Calculate win percentage
                    win_pct = h2h_record['wins'] / h2h_record['games'] if h2h_record['games'] > 0 else 0
                    
                    # Apply GD factor to H2H score
                    goal_diff_factor = 1.0
                    if h2h_record['wins'] > 0 and h2h_record['games'] - h2h_record['wins'] > 0:
                        total_gf = h2h_record.get('gf', 0)
                        total_ga = h2h_record.get('ga', 0)
                        goal_diff_factor = 1.0 + ((total_gf - total_ga) * 0.05)  # Each goal difference adds 5% to the score
                    
                    h2h_scores.append(win_pct * (adjusted_strength / avg_opp_win_pct) * goal_diff_factor)
                else:
                    h2h_scores.append(0)  # No games played against this opponent
            
            h2h_score = sum(h2h_scores) / len(h2h_scores) if h2h_scores else 0
            
            # Common opponents record
            common_opps = set()
            for opp in valid_teams:
                common_opps.update(stats[opp]['opponents'])
            
            # Separate common opponents into sectional and non-sectional
            sectional_opps = set(valid_teams)
            non_sectional_opps = common_opps - sectional_opps
            
            # Initialize weighted wins and games counters
            common_wins_weighted = 0
            common_games = 0
            
            # Process non-sectional opponents
            non_sectional_details = []
            for opp in non_sectional_opps:
                if opp in stats:
                    wins = sum(1 for r in games_inferred.itertuples() 
                             if (r.team1==team and r.team2==opp and r.score1>r.score2) 
                             or (r.team2==team and r.team1==opp and r.score2>r.score1))
                    losses = sum(1 for r in games_inferred.itertuples() 
                               if (r.team1==team and r.team2==opp and r.score1<r.score2) 
                               or (r.team2==team and r.team1==opp and r.score2<r.score1))
                    if wins + losses > 0:
                        opp_strength = stats[opp]['win_pct']
                        opp_sos = sos[opp]
                        # Adjusted SOS multiplier formula
                        sos_multiplier = 1.0 + ((opp_sos - 0.5) * 2.0)  # 0.8 to 1.2 range
                        adjusted_strength = opp_strength * sos_multiplier
                        # New weight system based on adjusted strength relative to average
                        weight = 0.7 + (0.6 * (adjusted_strength / avg_opp_win_pct))
                        weighted_wins = wins * weight
                        common_wins_weighted += weighted_wins
                        common_games += wins + losses
                        non_sectional_details.append({
                            'Opponent': opp,
                            'Record': f"{wins}-{losses}",
                            'Opp Win %': f"{opp_strength:.3f}",
                            'Opp SOS': f"{opp_sos:.3f}",
                            'SOS Mult': f"{sos_multiplier:.2f}x",
                            'Adj Strength': f"{adjusted_strength:.3f}",
                            'Weight': f"{weight:.1f}x",
                            'Weighted Wins': f"{weighted_wins:.1f}"
                        })
            
            # Process sectional opponents
            sectional_details = []
            for opp in sectional_opps:
                if opp != team and opp in stats:
                    h2h_record = h2h.get((team, opp), {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
                    if h2h_record['games'] > 0:
                        opp_strength = stats[opp]['win_pct']
                        opp_sos = sos[opp]
                        # Adjusted SOS multiplier formula with sectional boost
                        sos_multiplier = 1.0 + ((opp_sos - 0.5) * 2.0)  # 0.8 to 1.2 range
                        sos_multiplier *= 1.1  # Boost multiplier for sectional opponents
                        adjusted_strength = opp_strength * sos_multiplier
                        
                        # Calculate win percentage
                        win_pct = h2h_record['wins'] / h2h_record['games'] if h2h_record['games'] > 0 else 0
                        
                        weight = 0.7 + (0.6 * (adjusted_strength / avg_opp_win_pct))
                        weighted_wins = h2h_record['wins'] * weight
                        common_wins_weighted += weighted_wins
                        common_games += h2h_record['games']
                        sectional_details.append({
                            'Opponent': opp,
                            'Record': f"{h2h_record['wins']}-{h2h_record['games']-h2h_record['wins']}",
                            'Win %': f"{h2h_record['wins']/h2h_record['games']:.3f}" if h2h_record['games'] > 0 else "0.000",
                            'Opp Win %': f"{opp_strength:.3f}",
                            'Opp SOS': f"{opp_sos:.3f}",
                            'SOS Mult': f"{sos_multiplier:.2f}x",
                            'Adj Strength': f"{adjusted_strength:.3f}",
                            'Weight': f"{weight:.1f}x",
                            'Weighted Score': f"{h2h_record['wins']/h2h_record['games'] * weight:.3f}" if h2h_record['games'] > 0 else "0.000"
                        })
            
            common_win_pct = common_wins_weighted / common_games if common_games > 0 else 0
            
            # Overall win percentage (reduced weight)
            win_pct = stats[team]['win_pct']
            
            # Calculate combined score with adjusted weights and sectional penalty
            # H2H: 45%, Common Opp: 45%, Win %: 10%
            combined_score = (h2h_score * 0.45 + 
                            common_win_pct * 0.45 + 
                            win_pct * 0.1) * game_penalty * sectional_penalty
            
            # Store the score in the team's stats for reference
            if 'sectional_score' not in stats[team]:
                stats[team]['sectional_score'] = {}
            stats[team]['sectional_score'][sectional_name] = combined_score
            
            return combined_score
        
        # First get initial rankings by score
        initial_rankings = sorted(teams, key=get_team_score, reverse=True)
        
        return initial_rankings
    
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
    
    return sectional_rankings, sectional_order

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
    new_df = parse_scores_text(uploader.getvalue().decode())
    if not new_df.empty:
        raw_games = update_scores(raw_games, new_df)

# Initial stats
scored_games = raw_games.dropna(subset=['score1'])
initial_stats, _ = compute_stats(scored_games)
all_teams = set(raw_games['team1']).union(raw_games['team2'])
for t in all_teams:
    if t not in initial_stats:
        initial_stats[t] = {'wins':0,'losses':0,'ties':0,'gf':0,'ga':0,'games':0,'opponents':[]}
# Infer defaults
games_inferred = infer_default_scores(raw_games, initial_stats)
# Final stats
stats,h2h = compute_stats(games_inferred)
sos = compute_sos(stats)
py = compute_pythag(stats)
adj_vals = compute_adjusted_pythag(games_inferred, stats, k=k, x0=x0)
adj_ord, _ = rank_adj_pyth(stats, games_inferred, h2h, k=k, x0=x0)
elo = compute_elo(games_inferred)

# Compute sectional rankings
sectional_rankings, sectional_order = compute_sectional_rankings(stats, h2h, games_inferred)

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
    com = set(stats[te]['opponents']) & set(stats[opp]['opponents'])
    if com:
        dfc = []
        for c in com:
            try:
                wins_te = sum(1 for r in games_inferred.itertuples() 
                            if (r.team1==te and r.team2==c and r.score1>r.score2) 
                            or (r.team2==te and r.team1==c and r.score2>r.score1))
                losses_te = sum(1 for r in games_inferred.itertuples() 
                              if (r.team1==te and r.team2==c and r.score1<r.score2) 
                              or (r.team2==te and r.team1==c and r.score2<r.score1))
                wins_opp = sum(1 for r in games_inferred.itertuples() 
                             if (r.team1==opp and r.team2==c and r.score1>r.score2) 
                             or (r.team2==opp and r.team1==c and r.score2>r.score1))
                losses_opp = sum(1 for r in games_inferred.itertuples() 
                               if (r.team1==opp and r.team2==c and r.score1<r.score2) 
                               or (r.team2==opp and r.team1==c and r.score2<r.score1))
                
                dfc.append({
                    'Opp': c,
                    f'{te} W': wins_te,
                    f'{te} L': losses_te,
                    f'{opp} W': wins_opp,
                    f'{opp} L': losses_opp
                })
            except Exception as e:
                st.error(f"Error processing opponent {c}: {str(e)}")
        
        if dfc:  # Only create DataFrame if we have data
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
                # Calculate average games played in sectional
                valid_teams = [t for t in teams if t in stats]
                if not valid_teams:
                    st.write("No valid teams in sectional")
                    st.stop()
                    
                avg_games = sum(stats[t]['games'] for t in valid_teams) / len(valid_teams)
                team_games = stats[team]['games']
                game_penalty = 1.0
                if team_games < avg_games * 0.6:  # Adjusted threshold to 60% of average
                    game_penalty = (team_games / avg_games) ** 2  # More severe penalty (increased exponent)
                
                # Get all opponents' win percentages and SOS for strength calculation
                all_opp_win_pcts = [stats[opp]['win_pct'] for opp in stats if opp != team]
                avg_opp_win_pct = sum(all_opp_win_pcts) / len(all_opp_win_pcts) if all_opp_win_pcts else 0.5
                
                # Calculate sectional games adjustment
                sectional_games = sum(1 for opp in valid_teams if opp != team and h2h.get((team, opp), {'games': 0})['games'] > 0)
                avg_sectional_games = sum(1 for t in valid_teams for opp in valid_teams 
                                       if t != opp and h2h.get((t, opp), {'games': 0})['games'] > 0) / len(valid_teams)
                
                # Middle ground approach for sectional penalty
                sectional_penalty = 1.0
                if sectional_games == 0:
                    # If no sectional games, use a moderate penalty
                    sectional_penalty = 0.85  # 15% penalty
                elif sectional_games < avg_sectional_games * 0.7:
                    # Moderate penalty for teams with some sectional games
                    sectional_penalty = 0.85 + (0.15 * (sectional_games / (avg_sectional_games * 0.7)))
                
                # Calculate head-to-head matrix for this team
                h2h_scores = []
                for opp in valid_teams:
                    if opp == team:
                        continue
                    h2h_record = h2h.get((team, opp), {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
                    if h2h_record['games'] > 0:
                        opp_strength = stats[opp]['win_pct']
                        opp_sos = sos[opp]
                        # Adjusted SOS multiplier formula to penalize weak SOS and reward strong SOS
                        sos_multiplier = 1.0 + ((opp_sos - 0.5) * 2.0)  # 0.8 to 1.2 range for typical SOS
                        if opp in valid_teams:  # If opponent is in same sectional
                            sos_multiplier *= 1.1  # Boost multiplier for sectional opponents
                        adjusted_strength = opp_strength * sos_multiplier
                        
                        # Calculate win percentage
                        win_pct = h2h_record['wins'] / h2h_record['games'] if h2h_record['games'] > 0 else 0
                        
                        # Apply GD factor to H2H score
                        goal_diff_factor = 1.0
                        if h2h_record['wins'] > 0 and h2h_record['games'] - h2h_record['wins'] > 0:
                            total_gf = h2h_record.get('gf', 0)
                            total_ga = h2h_record.get('ga', 0)
                            goal_diff_factor = 1.0 + ((total_gf - total_ga) * 0.05)  # Each goal difference adds 5% to the score
                        
                        h2h_scores.append(win_pct * (adjusted_strength / avg_opp_win_pct) * goal_diff_factor)
                    else:
                        h2h_scores.append(0)  # No games played against this opponent
                
                h2h_score = sum(h2h_scores) / len(h2h_scores) if h2h_scores else 0
                
                # Common opponents record
                common_opps = set()
                for opp in valid_teams:
                    common_opps.update(stats[opp]['opponents'])
                
                # Separate common opponents into sectional and non-sectional
                sectional_opps = set(valid_teams)
                non_sectional_opps = common_opps - sectional_opps
                
                # Initialize weighted wins and games counters
                common_wins_weighted = 0
                common_games = 0
                
                # Process non-sectional opponents
                non_sectional_details = []
                for opp in non_sectional_opps:
                    if opp in stats:
                        wins = sum(1 for r in games_inferred.itertuples() 
                                 if (r.team1==team and r.team2==opp and r.score1>r.score2) 
                                 or (r.team2==team and r.team1==opp and r.score2>r.score1))
                        losses = sum(1 for r in games_inferred.itertuples() 
                                   if (r.team1==team and r.team2==opp and r.score1<r.score2) 
                                   or (r.team2==team and r.team1==opp and r.score2<r.score1))
                        if wins + losses > 0:
                            opp_strength = stats[opp]['win_pct']
                            opp_sos = sos[opp]
                            # Adjusted SOS multiplier formula
                            sos_multiplier = 1.0 + ((opp_sos - 0.5) * 2.0)  # 0.8 to 1.2 range
                            adjusted_strength = opp_strength * sos_multiplier
                            # New weight system based on adjusted strength relative to average
                            weight = 0.7 + (0.6 * (adjusted_strength / avg_opp_win_pct))
                            weighted_wins = wins * weight
                            common_wins_weighted += weighted_wins
                            common_games += wins + losses
                            non_sectional_details.append({
                                'Opponent': opp,
                                'Record': f"{wins}-{losses}",
                                'Opp Win %': f"{opp_strength:.3f}",
                                'Opp SOS': f"{opp_sos:.3f}",
                                'SOS Mult': f"{sos_multiplier:.2f}x",
                                'Adj Strength': f"{adjusted_strength:.3f}",
                                'Weight': f"{weight:.1f}x",
                                'Weighted Wins': f"{weighted_wins:.1f}"
                            })
                
                # Process sectional opponents
                sectional_details = []
                for opp in sectional_opps:
                    if opp != team and opp in stats:
                        h2h_record = h2h.get((team, opp), {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
                        if h2h_record['games'] > 0:
                            opp_strength = stats[opp]['win_pct']
                            opp_sos = sos[opp]
                            # Adjusted SOS multiplier formula with sectional boost
                            sos_multiplier = 1.0 + ((opp_sos - 0.5) * 2.0)  # 0.8 to 1.2 range
                            sos_multiplier *= 1.1  # Boost multiplier for sectional opponents
                            adjusted_strength = opp_strength * sos_multiplier
                            
                            # Calculate win percentage
                            win_pct = h2h_record['wins'] / h2h_record['games'] if h2h_record['games'] > 0 else 0
                            
                            weight = 0.7 + (0.6 * (adjusted_strength / avg_opp_win_pct))
                            weighted_wins = h2h_record['wins'] * weight
                            common_wins_weighted += weighted_wins
                            common_games += h2h_record['games']
                            sectional_details.append({
                                'Opponent': opp,
                                'Record': f"{h2h_record['wins']}-{h2h_record['games']-h2h_record['wins']}",
                                'Win %': f"{h2h_record['wins']/h2h_record['games']:.3f}" if h2h_record['games'] > 0 else "0.000",
                                'Opp Win %': f"{opp_strength:.3f}",
                                'Opp SOS': f"{opp_sos:.3f}",
                                'SOS Mult': f"{sos_multiplier:.2f}x",
                                'Adj Strength': f"{adjusted_strength:.3f}",
                                'Weight': f"{weight:.1f}x",
                                'Weighted Score': f"{h2h_record['wins']/h2h_record['games'] * weight:.3f}" if h2h_record['games'] > 0 else "0.000"
                            })
                
                common_win_pct = common_wins_weighted / common_games if common_games > 0 else 0
                
                # Overall win percentage (reduced weight)
                win_pct = stats[team]['win_pct']
                
                # Display factors with updated weights
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("H2H (45%)", f"{h2h_score:.3f}")
                with col2:
                    st.metric("Common Opp (45%)", f"{common_wins_weighted:.1f}/{common_games}", f"{common_win_pct:.3f}")
                with col3:
                    st.metric("Win % (10%)", f"{win_pct:.3f}")
                with col4:
                    penalties = []
                    if game_penalty < 1.0:
                        penalties.append(f"Games: {game_penalty:.2f}x")
                    if sectional_penalty < 1.0:
                        penalties.append(f"Sectional: {sectional_penalty:.2f}x")
                    st.metric("Penalties", "None" if not penalties else ", ".join(penalties))
                
                # Combined score
                combined_score = (h2h_score * 0.45 + 
                                common_win_pct * 0.45 + 
                                win_pct * 0.1) * game_penalty * sectional_penalty
                st.metric("Combined Score", f"{combined_score:.3f}")
                
                # Head-to-head details
                st.markdown("##### Head-to-Head Details")
                h2h_details = []
                for opp in valid_teams:
                    if opp != team and opp in stats:
                        h2h_record = h2h.get((team, opp), {'wins': 0, 'games': 0, 'gf': 0, 'ga': 0})
                        if h2h_record['games'] > 0:
                            opp_strength = stats[opp]['win_pct']
                            opp_sos = sos[opp]
                            # More aggressive SOS adjustment
                            sos_multiplier = 0.3 + (opp_sos * 1.7)  # 0.3 to 2.0 range
                            adjusted_strength = opp_strength * sos_multiplier
                            weight = (adjusted_strength / avg_opp_win_pct) if adjusted_strength > avg_opp_win_pct else 0.3
                            
                            # Calculate goal differential factor only for split series
                            goal_diff_factor = 1.0
                            if h2h_record['wins'] > 0 and h2h_record['games'] - h2h_record['wins'] > 0:
                                # Calculate total goals for and against in all games
                                total_gf = h2h_record.get('gf', 0)
                                total_ga = h2h_record.get('ga', 0)
                                # Apply a stronger goal differential factor for split series
                                goal_diff_factor = 1.0 + ((total_gf - total_ga) * 0.05)  # Each goal difference adds 5% to the score
                            
                            h2h_details.append({
                                'Opponent': opp,
                                'Record': f"{h2h_record['wins']}-{h2h_record['games']-h2h_record['wins']}",
                                'Win %': f"{h2h_record['wins']/h2h_record['games']:.3f}",
                                'Opp Win %': f"{opp_strength:.3f}",
                                'Opp SOS': f"{opp_sos:.3f}",
                                'SOS Mult': f"{sos_multiplier:.2f}x",
                                'Adj Strength': f"{adjusted_strength:.3f}",
                                'Weight': f"{weight:.1f}x",
                                'Total GF': f"{h2h_record.get('gf', 0)}",
                                'Total GA': f"{h2h_record.get('ga', 0)}",
                                'GD Factor': f"{goal_diff_factor:.2f}x",
                                'Weighted Score': f"{h2h_record['wins']/h2h_record['games'] * weight * goal_diff_factor:.3f}"
                            })
                if h2h_details:
                    st.dataframe(pd.DataFrame(h2h_details))
                else:
                    st.write("No head-to-head games played")
                
                # Common opponents details
                st.markdown("##### Non-Sectional Common Opponents")
                if non_sectional_details:
                    st.dataframe(pd.DataFrame(non_sectional_details))
                else:
                    st.write("No non-sectional common opponents")
                
                st.markdown("##### Sectional Common Opponents")
                if sectional_details:
                    st.dataframe(pd.DataFrame(sectional_details))
                else:
                    st.write("No sectional common opponents")
