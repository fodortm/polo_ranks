import json
from pathlib import Path

from domain.parsing import (
    apply_rule_c_cleanup_suffix,
    apply_rule_d_alias_normalization,
    load_parser_config,
    parse_game_line_anchored,
)

FIXTURES = json.loads(Path("tests/data/parser_line_fixtures.json").read_text(encoding="utf-8"))


def test_rule_a_strict_score_pattern():
    cfg = load_parser_config()
    for case in FIXTURES["rule_a"]:
        row = parse_game_line_anchored(case["line"], cfg)
        assert row is not None
        assert row["team1"] == case["team1"]
        assert row["score1"] == case["score1"]
        assert row["team2"] == case["team2"]
        assert row["score2"] == case["score2"]


def test_rule_b_winner_only_notation():
    cfg = load_parser_config()
    for case in FIXTURES["rule_b"]:
        row = parse_game_line_anchored(case["line"], cfg)
        assert row is not None
        assert row["team1"] == case["team1"]
        assert row["team2"] == case["team2"]
        assert row["score1"] is None
        assert row["score2"] is None


def test_rule_c_suffix_cleanup_only():
    cfg = load_parser_config()
    for case in FIXTURES["rule_c"]:
        assert apply_rule_c_cleanup_suffix(case["raw"], cfg) == case["normalized"]


def test_rule_d_alias_normalization_only():
    cfg = load_parser_config()
    for case in FIXTURES["rule_d"]:
        assert apply_rule_d_alias_normalization(case["raw"], cfg) == case["normalized"]
