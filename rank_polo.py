import pandas as pd
import os
import re
import math
from collections import defaultdict
from functools import cmp_to_key
import streamlit as st

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
    return pd.read_csv(SCORES_CSV) if os.path.exists(SCORES_CSV) else pd.DataFrame(columns=["team1","score1","team2","score2"])

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
    h2h = defaultdict(lambda: {'wins':0,'games':0})
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

def compute_adjusted_pythag(games,stats,exp=2):
    adj_gf,adj_ga=defaultdict(float),defaultdict(float)
    for _,r in games.iterrows():
        for me,opp,ms,os in [(r.team1,r.team2,r.score1,r.score2),(r.team2,r.team1,r.score2,r.score1)]:
            weight=stats[opp]['win_pct']**3
            adj_gf[me]+=ms*weight; adj_ga[me]+=os*weight
    adj={}
    for t,st in stats.items():
        g=st['games']
        agf=adj_gf[t]/g if g else 0; aga=adj_ga[t]/g if g else 0
        adj[t]=agf**exp/(agf**exp+aga**exp) if agf+aga>0 else 0
    return adj

# ---------------- Rankings ---------------- #
def rank_win_pct(stats,h2h):
    def cmp(a,b):
        if stats[a]['win_pct']!=stats[b]['win_pct']: return -1 if stats[a]['win_pct']>stats[b]['win_pct'] else 1
        h=h2h.get((a,b),{'wins':0,'games':0});
        if h['games']:
            p=h['wins']/h['games']
            if p!=0.5: return -1 if p>0.5 else 1
        return (stats[b]['gd'] - stats[a]['gd'])
    return sorted(stats.keys(),key=cmp_to_key(cmp))
def rank_pythag(stats,p): return sorted(stats.keys(),key=lambda t:p[t],reverse=True)
def rank_adj_pyth(stats,games,h2h):
    vals=compute_adjusted_pythag(games,stats)
    order=sorted(stats.keys(),key=lambda t:vals[t],reverse=True)
    final=[]
    eps=1e-4
    for t in order:
        if final:
            prev=final[-1]
            if abs(vals[prev]-vals[t])<eps:
                h=h2h.get((t,prev),{'wins':0,'games':0})
                if h['games'] and h['wins']/h['games']>0.5: final[-1],t=t,prev
        final.append(t)
    return final,vals

def compute_elo(games,initial=1500,k=32):
    teams=set(games['team1']).union(games['team2'])
    R={t:initial for t in teams}
    for _,r in games.iterrows():
        a,b,sa,sb=r.team1,r.team2,r.score1,r.score2
        ea=1/(1+10**((R[b]-R[a])/400)); eb=1-ea
        aa,ab = (1,0) if sa>sb else ((0,1) if sb>sa else (0.5,0.5))
        R[a]+=k*(aa-ea); R[b]+=k*(ab-eb)
    return R
def rank_elo(stats,elo): return sorted(stats.keys(),key=lambda t:elo[t],reverse=True)

# ---------------- App ---------------- #
st.set_page_config(page_title="Polo Dashboard",layout="wide")
# Load & parse
games=load_scores()
uploader=st.sidebar.file_uploader("Upload scores .txt",type="txt")
if uploader:
    new=parse_scores_text(uploader.getvalue().decode())
    if not new.empty:
        games=update_scores(games,new)
    games_inferred=infer_default_scores(games,compute_stats(games)[0])
else:
    games_inferred=infer_default_scores(games,compute_stats(games)[0])
# Stats
stats,h2h=compute_stats(games_inferred)
sos=compute_sos(stats)
py=compute_pythag(stats)
adj=compute_adjusted_pythag(games_inferred,stats)
elo=compute_elo(games_inferred)
# Orders & filter
win_ord=rank_win_pct(stats,h2h)
py_ord=rank_pythag(stats,py)
adj_ord,adj_vals=rank_adj_pyth(stats,games_inferred,h2h)
elo_ord=rank_elo(stats,elo)
maxg=max(st['games'] for st in stats.values()) if stats else 0;thr=maxg/2
win_ord=[t for t in win_ord if stats[t]['games']>=thr]
py_ord=[t for t in py_ord if stats[t]['games']>=thr]
adj_ord=[t for t in adj_ord if stats[t]['games']>=thr]
elo_ord=[t for t in elo_ord if stats[t]['games']>=thr]
teams=sorted(stats.keys())
# Profile select
st.sidebar.header("Team Profile")
te=st.sidebar.selectbox("Select Team",teams,index=teams.index("Loyola") if "Loyola" in teams else 0)
# Rank numbers
r_win=win_ord.index(te)+1 if te in win_ord else None
r_py=py_ord.index(te)+1 if te in py_ord else None
r_adj=adj_ord.index(te)+1 if te in adj_ord else None
r_elo=elo_ord.index(te)+1 if te in elo_ord else None
ranks=[r for r in [r_win,r_py,r_adj,r_elo] if r]
r_avg=round(sum(ranks)/len(ranks),2) if ranks else None
# Tabs
tabs=st.tabs(["Profile","Win%","Pythag","AdjPyth","Elo","Avg"])
with tabs[0]:
    st.subheader(f"Profile: {te}")
    st.table(pd.DataFrame.from_dict({
        'GPG For':f"{stats[te]['gf']/stats[te]['games']:.2f}",
        'GPG Against':f"{stats[te]['ga']/stats[te]['games']:.2f}",
        'GD/Game':f"{(stats[te]['gf']-stats[te]['ga'])/stats[te]['games']:.2f}",
        'Win %':f"{stats[te]['win_pct']:.3f}",
        'SOS':f"{sos[te]:.3f}",
        'Rank Win%':r_win,'Rank Pythag':r_py,'Rank Adj':r_adj,'Rank Elo':r_elo,'Avg':r_avg
    },orient='index',columns=['Value']))
    opp=st.selectbox("Compare vs",[t for t in teams if t!=te])
    st.write(f"{opp} Ranks: Win% #{win_ord.index(opp)+1 if opp in win_ord else '-'} (SOS {sos[opp]:.3f}), "
             f"Pyth #{py_ord.index(opp)+1 if opp in py_ord else '-'} (SOS {sos[opp]:.3f}), "
             f"AdjPyth #{adj_ord.index(opp)+1 if opp in adj_ord else '-'} (SOS {sos[opp]:.3f}), "
             f"Elo #{elo_ord.index(opp)+1 if opp in elo_ord else '-'}")
    h=h2h.get((te,opp),{'wins':0,'games':0});st.write(f"H2H: {h['wins']}-{h['games']-h['wins']} in {h['games']} games")
    com=set(stats[te]['opponents'])&set(stats[opp]['opponents'])
    if com:
        dfc=[]
        for c in com:
            dfc.append({'Opp':c,
                        f'{te} W':sum(1 for r in games_inferred.itertuples() if (r.team1==te and r.team2==c and r.score1>r.score2) or (r.team2==te and r.team1==c and r.score2>r.score1)),
                        f'{te} L':sum(1 for r in games_inferred.itertuples() if (r.team1==te and r.team2==c and r.score1<r.score2) or (r.team2==te and r.team1==c and r.score2<r.score1)),
                        f'{opp} W':sum(1 for r in games_inferred.itertuples() if (r.team1==opp and r.team2==c and r.score1>r.score2) or (r.team2==opp and r.team1==c and r.score2>r.score1)),
                        f'{opp} L':sum(1 for r in games_inferred.itertuples() if (r.team1==opp and r.team2==c and r.score1<r.score2) or (r.team2==opp and r.team1==c and r.score2<r.score1))})
        st.dataframe(pd.DataFrame(dfc))
    else: st.write("No common opponents.")
    st.markdown("**Full Schedule**")
    sch=[]
    for r in games_inferred.itertuples():
        if r.team1==te or r.team2==te:
            o=r.team2 if r.team1==te else r.team1
            s=r.score1 if r.team1==te else r.score2
            a=r.score2 if r.team1==te else r.score1
            sch.append({'Opp':o,'Scored':s,'Allowed':a})
    st.dataframe(pd.DataFrame(sch))
with tabs[1]:
    st.subheader("Rankings by Win %")
    st.dataframe(pd.DataFrame({'Team':win_ord,'Win %':[f"{stats[t]['win_pct']:.3f}" for t in win_ord],'SOS':[f"{sos[t]:.3f}" for t in win_ord]}))
with tabs[2]:
    st.subheader("Rankings by Pythagorean")
    st.dataframe(pd.DataFrame({'Team':py_ord,'Exp %':[f"{py[t]:.3f}" for t in py_ord],'SOS':[f"{sos[t]:.3f}" for t in py_ord]}))
with tabs[3]:
    st.subheader("Rankings by Adjusted Pythagorean")
    st.dataframe(pd.DataFrame({'Team':adj_ord,'AdjPyth %':[f"{adj_vals[t]:.3f}" for t in adj_ord],'SOS':[f"{sos[t]:.3f}" for t in adj_ord]}))
with tabs[4]:
    st.subheader("Rankings by Elo")
    st.dataframe(pd.DataFrame({'Team':elo_ord,'Elo':[f"{elo[t]:.1f}" for t in elo_ord],'SOS':[f"{sos[t]:.3f}" for t in elo_ord]}))
with tabs[5]:
    st.subheader("Rankings by Avg Composite")
    comp=[]
    for t in teams:
        if stats[t]['games']>=thr:
            w=win_ord.index(t)+1; p=py_ord.index(t)+1; a=adj_ord.index(t)+1; e=elo_ord.index(t)+1
            comp.append((t,w,p,a,e,round((w+p+a+e)/4,2),sos[t]))
    df_avg=pd.DataFrame(comp,columns=['Team','Win','Pyth','AdjPyth','Elo','Avg','SOS']).sort_values('Avg')
    st.dataframe(df_avg)
