import pandas as pd
from domain.imputation import infer_default_scores

def test_impute_defaults():
    games = pd.DataFrame([{'team1':'A','score1':None,'team2':'B','score2':None}])
    stats = {'A': {'games':0,'gf':0,'ga':0}, 'B': {'games':0,'gf':0,'ga':0}}
    out = infer_default_scores(games, stats)
    assert out.loc[0, 'score1'] > out.loc[0, 'score2']
