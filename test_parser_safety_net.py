import textwrap
from pathlib import Path

from rank_polo import _parse_game_line_anchored, load_games_pipeline


def test_parser_representative_lines():
    standard = _parse_game_line_anchored("Loyola 8 Fenwick 5")
    assert standard is not None
    assert standard["team1"] == "Loyola"
    assert standard["score1"] == 8

    ot = _parse_game_line_anchored("New Trier 7 Sandburg 6 (OT)")
    assert ot is not None
    assert ot["team1"] == "New Trier"
    assert ot["team2"] == "Sandburg"

    placement = _parse_game_line_anchored("Brother Rice 9 Naperville North 4 (3rd Place)")
    assert placement is not None
    assert placement["team1"] == "Brother Rice"


def test_pipeline_dedup_aliases_and_qa_meta(tmp_path: Path):
    sample = textwrap.dedent(
        """
        Loyola 8 Fenwick 5
        Loyola 8 Fenwick 5
        Chicago U 11 Whitney Young 2
        New Trier 7 Sandburg 6 (OT)
        Brother Rice 9 Naperville North 4 (Final)
        Stevenson d. Latin
        Random unexpected token stream here
        """
    ).strip()
    score_file = tmp_path / "2026_week9_scores_illpolo.txt"
    score_file.write_text(sample + "\n", encoding="utf-8")

    games_df, qa_meta = load_games_pipeline(str(tmp_path))

    # Key parser coverage: score lines + OT/placement + d. are ingested
    assert len(games_df) == 5
    assert ((games_df["team1"] == "Stevenson") & games_df["score1"].isna()).any()

    # Duplicate score line is dropped
    assert qa_meta["duplicates_dropped"] == 1

    # Alias normalization collapses to canonical team name
    assert (games_df["team1"] == "U-Chicago").any()

    # UI QA panel data contract
    assert qa_meta["games_parsed"] == 6
    assert qa_meta["suspicious_unparsed"] == 1
    assert qa_meta["skipped"] == 0
    assert qa_meta["unresolved_suspicious_lines"] == ["Random unexpected token stream here"]
