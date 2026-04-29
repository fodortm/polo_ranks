from domain.parsing import normalize_team_name, parse_game_line_anchored

def test_parse_and_normalize():
    assert normalize_team_name('Chicago U') == 'U-Chicago'
    row = parse_game_line_anchored('Loyola 8 Fenwick 5')
    assert row['score1'] == 8
