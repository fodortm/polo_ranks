import pandas as pd
import os
import re
from collections import defaultdict
from functools import cmp_to_key
import streamlit as st

# ---------------- Data Parsing ---------------- #
SCORES_CSV = "scores.csv"
PATTERN = re.compile(r"^(.+?)\s+(\d+)(?:\s*\(OT\))?\s+(.+?)\s+(\d+)")

def _parse_line(line):
    games = []
    raw = line.strip()
    if not raw or raw.lower().startswith("championship") or " vs " in raw.lower():
        return games
    if re.search(r"\s+d\.\s+", raw, flags=re.IGNORECASE):
        a, b = re.split(r"\s+d\.\s+", raw, maxsplit=1, flags=re.IGNORECASE)
        games.append({"team1": a.strip(), "score1": 1, "team2": b.strip(), "score2": 0})
        return games
    m = PATTERN.match(raw)
    if m:
        t1, s1, t2, s2 = m.group(1).strip(), int(m.group(2)), m.group(3).strip(), int(m.group(4))
        games.append({"team1": t1, "score1": s1, "team2": t2, "score2": s2})
    return games

def parse_scores_text(text):
    games = []
    for line in text.splitlines():
        games.extend(_parse_line(line))
    return pd.DataFrame(games)

# ---------------- Data I/O ---------------- #
def load_scores():
    return pd.read_csv(SCORES_CSV) if os.path.exists(SCORES_CSV) else pd.DataFrame(columns=["team1","score1","team2","score2"])

def save_scores(df):
    df.to_csv(SCORES_CSV, index=False)

def update_scores(existing, new_df):
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    save_scores(combined)
    return combined

# ---------------- Statistics ---------------- #
def compute_stats(games):
    stats = {}
    h2h = defaultdict(lambda: {'wins': 0, 'games': 0})
    teams = set(games['team1']).union(games['team2'])
    for t in teams:
        stats[t] = {'wins':0,'losses':0,'ties':0,'gf':0,'ga':0,'games':0,'opponents':[]}
    for _, r in games.iterrows():
        t1, t2, s1, s2 = r.team1, r.team2, r.score1, r.score2
        for me, opp, ms, os in [(t1,t2,s1,s2),(t2,t1,s2,s1)]:
            stats[me]['gf'] += ms
            stats[me]['ga'] += os
            stats[me]['games'] += 1
            stats[me]['opponents'].append(opp)
        if s1 > s2:
            stats[t1]['wins'] += 1; stats[t2]['losses'] += 1; h2h[(t1,t2)]['wins'] += 1
        elif s2 > s1:
            stats[t2]['wins'] += 1; stats[t1]['losses'] += 1; h2h[(t2,t1)]['wins'] += 1
        else:
            stats[t1]['ties'] += 1; stats[t2]['ties'] += 1
        h2h[(t1,t2)]['games'] += 1; h2h[(t2,t1)]['games'] += 1
    for t, st in stats.items():
        total = st['wins'] + st['losses'] + st['ties']
        st['win_pct'] = st['wins']/total if total else 0
        st['gd'] = st['gf'] - st['ga']
    return stats, h2h

def compute_sos(stats):
    sos = {}
    for t, st in stats.items():
        opps = st['opponents']
        sos[t] = sum(stats[o]['win_pct'] for o in opps)/len(opps) if opps else 0
    return sos


def compute_pythag(stats, exp=2):
    """
    Compute each team's Pythagorean expectation based on raw goals.
    """
    pythag = {}
    for t, st in stats.items():
        gf, ga = st['gf'], st['ga']
        pythag[t] = gf**exp / (gf**exp + ga**exp) if gf + ga > 0 else 0
    return pythag

# ---------------- Elo-Style Rankings ---------------- #
def compute_elo(games, initial=1500, k=32):
    """
    Compute Elo ratings treating each game as a match between two teams.
    Win=1, tie=0.5, loss=0. Updates in sequence.
    """
    teams = set(games['team1']).union(games['team2'])
    rating = {t: initial for t in teams}
    for _, r in games.iterrows():
        a, b, sa, sb = r.team1, r.team2, r.score1, r.score2
        # expected scores
        ea = 1/(1+10**((rating[b]-rating[a])/400))
        eb = 1 - ea
        # actual
        if sa > sb: aa, ab = 1, 0
        elif sa < sb: aa, ab = 0, 1
        else: aa, ab = 0.5, 0.5
        rating[a] += k*(aa - ea)
        rating[b] += k*(ab - eb)
    return rating

# ---------------- Adjusted Pythagorean ---------------- #
def compute_adjusted_pythag(games, stats, exp=2):
    adj_gf = defaultdict(float)
    adj_ga = defaultdict(float)
    for _, r in games.iterrows():
        for me, opp, ms, os in [(r.team1,r.team2,r.score1,r.score2),(r.team2,r.team1,r.score2,r.score1)]:
            weight = stats[opp]['win_pct'] ** 3
            adj_gf[me] += ms * weight
            adj_ga[me] += os * weight
    adjusted = {}
    for t, st in stats.items():
        g = st['games']
        avg_gf = adj_gf[t]/g if g else 0
        avg_ga = adj_ga[t]/g if g else 0
        adjusted[t] = avg_gf**exp/(avg_gf**exp + avg_ga**exp) if (avg_gf+avg_ga)>0 else 0
    return adjusted

def rank_adjusted_pythag(stats, games, h2h, exp=2, epsilon=1e-4):
    adjusted = compute_adjusted_pythag(games, stats, exp)
    order = sorted(stats.keys(), key=lambda t: adjusted[t], reverse=True)
    final = []
    for t in order:
        if final:
            prev = final[-1]
            if abs(adjusted[prev] - adjusted[t]) < epsilon:
                h = h2h.get((t, prev), {'wins':0,'games':0})
                if h['games'] and h['wins']/h['games'] > 0.5:
                    final[-1], t = t, prev
        final.append(t)
    return final, adjusted

# ---------------- Standard Rankings ---------------- #
def rank_win_pct(stats, h2h):
    def cmp(a, b):
        if stats[a]['win_pct'] != stats[b]['win_pct']:
            return -1 if stats[a]['win_pct'] > stats[b]['win_pct'] else 1
        h = h2h.get((a,b), {'wins':0,'games':0})
        if h['games']:
            p = h['wins']/h['games']
            if p != 0.5: return -1 if p>0.5 else 1
        if stats[a]['gd'] != stats[b]['gd']:
            return -1 if stats[a]['gd'] > stats[b]['gd'] else 1
        return 0
    return sorted(stats.keys(), key=cmp_to_key(cmp))

def rank_pythag(stats, pythag):
    return sorted(stats.keys(), key=lambda t: pythag[t], reverse=True)

def rank_elo(stats, elo_ratings):
    return sorted(stats.keys(), key=lambda t: elo_ratings[t], reverse=True)

# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Polo Rankings Dashboard", layout="wide")
st.title("Illinois HS Polo Dashboard")
# Sidebar upload
st.sidebar.header("Upload Scores")
f = st.sidebar.file_uploader(".txt file", type="txt")
games_df = load_scores()
if f:
    new = parse_scores_text(f.getvalue().decode())
    if not new.empty:
        games_df = update_scores(games_df, new)
        st.sidebar.success(f"Added {len(new)} games.")
    else:
        st.sidebar.error("No games parsed.")
# Compute stats and ranks
games_df = games_df if not games_df.empty else pd.DataFrame(columns=["team1","score1","team2","score2"])
stats, h2h = compute_stats(games_df)
sos = compute_sos(stats)
win_order = rank_win_pct(stats, h2h)
# Standard Pythagorean
py_vals = compute_pythag(stats)
py_order = rank_pythag(stats, py_vals)
# Adjusted Pythagorean
adj_vals = compute_adjusted_pythag(games_df, stats)
adj_order, _ = rank_adjusted_pythag(stats, games_df, h2h)
elo_vals = compute_elo(games_df)
elo_order = rank_elo(stats, elo_vals)
# filter teams
max_games = max(st['games'] for st in stats.values()) if stats else 0
thr = max_games/2
win_order = [t for t in win_order if stats[t]['games']>=thr]
py_order = [t for t in py_order if stats[t]['games']>=thr]
adj_order = [t for t in adj_order if stats[t]['games']>=thr]
elo_order = [t for t in elo_order if stats[t]['games']>=thr]
teams = sorted(stats.keys())
# Profile selection
st.sidebar.header("Team Profile")
team = st.sidebar.selectbox("Select Team", teams, index=teams.index("Loyola") if "Loyola" in teams else 0)
# Determine ranks
r_win = win_order.index(team)+1 if team in win_order else None
r_py = py_order.index(team)+1 if team in py_order else None
r_adj = adj_order.index(team)+1 if team in adj_order else None
r_elo = elo_order.index(team)+1 if team in elo_order else None
ranks = [r for r in [r_win, r_py, r_adj, r_elo] if r]
r_avg = sum(ranks)/len(ranks) if ranks else None
# Tabs
tabs = st.tabs(["Profile","Win %","Pythagorean","Adj Pythag","Elo","Avg Composite"])
with tabs[0]:
    st.subheader(f"Profile: {team}")
    gp = stats[team]['games']
    info = {}
    if gp:
        info['GPG For'] = f"{stats[team]['gf']/gp:.2f}"
        info['GPG Against'] = f"{stats[team]['ga']/gp:.2f}"
        info['GD/Game'] = f"{(stats[team]['gf']-stats[team]['ga'])/gp:.2f}"
        info['Win %'] = f"{stats[team]['win_pct']:.3f}"
        info['SOS'] = f"{sos[team]:.3f}"
        info['Rank Win%'] = r_win
        info['Rank Pythag'] = r_py
        info['Rank Adj Pyth'] = r_adj
        info['Rank Elo'] = r_elo
        info['Avg Rank'] = f"{r_avg:.2f}"
    st.table(pd.DataFrame.from_dict(info, orient='index', columns=['Value']))
    # Opponent comparison
    opp = st.selectbox("Compare vs", [t for t in teams if t!=team])
    opp_info = (
        f"Win% #{win_order.index(opp)+1 if opp in win_order else '-'} (SOS {sos[opp]:.3f}), "
        f"Pyth #{py_order.index(opp)+1 if opp in py_order else '-'} (SOS {sos[opp]:.3f}), "
        f"AdjPyth #{adj_order.index(opp)+1 if opp in adj_order else '-'} (SOS {sos[opp]:.3f}), "
        f"Elo #{elo_order.index(opp)+1 if opp in elo_order else '-'}"
    )
    st.write(f"**{opp} Ranks & SOS:** {opp_info}")
    h = h2h.get((team, opp), {'wins':0,'games':0}); st.write(f"H2H: {h['wins']}-{h['games']-h['wins']} in {h['games']} games")
    common = set(stats[team]['opponents']) & set(stats[opp]['opponents'])
    if common:
        rows = []
        for c in common:
            tW = sum(1 for r in games_df.itertuples() if (r.team1==team and r.team2==c and r.score1>r.score2) or (r.team2==team and r.team1==c and r.score2>r.score1))
            tL = sum(1 for r in games_df.itertuples() if (r.team1==team and r.team2==c and r.score1<r.score2) or (r.team2==team and r.team1==c and r.score2<r.score1))
            oW = sum(1 for r in games_df.itertuples() if (r.team1==opp and r.team2==c and r.score1>r.score2) or (r.team2==opp and r.team1==c and r.score2>r.score1))
            oL = sum(1 for r in games_df.itertuples() if (r.team1==opp and r.team2==c and r.score1<r.score2) or (r.team2==opp and r.team1==c and r.score2<r.score1))
            rows.append({'Opponent':c, f'{team} W':tW, f'{team} L':tL, f'{opp} W':oW, f'{opp} L':oL})
        st.dataframe(pd.DataFrame(rows))
    else:
        st.write("No common opponents.")
    st.markdown("**Full Schedule**")
    sched = []
    for r in games_df.itertuples():
        if r.team1==team or r.team2==team:
            o = r.team2 if r.team1==team else r.team1
            sc = r.score1 if r.team1==team else r.score2
            al = r.score2 if r.team1==team else r.score1
            sched.append({'Opponent':o,'Scored':sc,'Allowed':al})
    st.dataframe(pd.DataFrame(sched))
with tabs[1]:
    st.subheader("Rankings by Win Percentage")
    st.dataframe(pd.DataFrame({'Team':win_order,
                                 'Win %':[f"{stats[t]['win_pct']:.3f}" for t in win_order],
                                 'SOS':[f"{sos[t]:.3f}" for t in win_order]}))
with tabs[2]:
    st.subheader("Rankings by Pythagorean")
    st.dataframe(pd.DataFrame({'Team':py_order,
                                 'Adj Exp %':[f"{py_vals[t]:.3f}" for t in py_order],
                                 'SOS':[f"{sos[t]:.3f}" for t in py_order]}))
with tabs[3]:
    st.subheader("Rankings by Adjusted Pythagorean")
    st.dataframe(pd.DataFrame({'Team':adj_order,
                                 'AdjPyth %':[f"{adj_vals[t]:.3f}" for t in adj_order],
                                 'SOS':[f"{sos[t]:.3f}" for t in adj_order]}))
with tabs[4]:
    st.subheader("Rankings by Elo")
    df_elo = pd.DataFrame({'Team':elo_order,
                            'Elo Rating':[f"{elo_vals[t]:.1f}" for t in elo_order],
                            'SOS':[f"{sos[t]:.3f}" for t in elo_order]})
    st.dataframe(df_elo)
with tabs[5]:
    st.subheader("Rankings by Avg Composite")
    comp = []
    for t in teams:
        if stats[t]['games'] >= thr:
            w = win_order.index(t)+1
            p = py_order.index(t)+1
            a = adj_order.index(t)+1
            e = elo_order.index(t)+1
            avg = round((w+p+a+e)/4,2)
            comp.append((t, w, p, a, e, avg, sos[t]))
    df_avg = pd.DataFrame(comp, columns=['Team','Win Rank','Pyth Rank','AdjPyth Rank','Elo Rank','Avg Rank','SOS']).sort_values('Avg Rank')
    st.dataframe(df_avg)
