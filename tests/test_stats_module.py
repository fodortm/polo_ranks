import pandas as pd
from domain.stats import compute_stats, compute_sos

def test_stats_and_sos():
    df = pd.DataFrame([{'team1':'A','score1':3,'team2':'B','score2':1}])
    stats, _ = compute_stats(df)
    sos = compute_sos(stats)
    assert stats['A']['wins'] == 1
    assert 'A' in sos
