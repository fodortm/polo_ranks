import json
import os
import re
from pathlib import Path

import pandas as pd

from config.constants import DATA_DIR, SCORES_GLOB_SUFFIX

_DEFAULT_SCORE_PATTERN = (
    r"^\s*(?P<team1>.+?)\s+(?P<score1>\d+)\s+"
    r"(?P<team2>.+?)\s+(?P<score2>\d+)\s*"
    r"(?:(?:\((?:OT|SO)\))|(?:OT)|(?:SO)|(?:\([^)]*OT[^)]*\))|(?:\((?:\d+(?:st|nd|rd|th)\s+Place|Final)\)))?\s*$"
)

DEFAULT_PARSER_CONFIG = {
    "aliases": {"chicago u": "U-Chicago", "chicago-u": "U-Chicago"},
    "cleanup_suffix_patterns": [r"\s*\((?:\d+(?:st|nd|rd|th)\s+Place|Final)\)\s*$"],
    "score_pattern": _DEFAULT_SCORE_PATTERN,
}



def load_parser_config(config_path="parser_config.json"):
    cfg = dict(DEFAULT_PARSER_CONFIG)
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            user_cfg = json.load(fh) or {}
        cfg.update({k: v for k, v in user_cfg.items() if v is not None})
    cfg["aliases"] = {k.lower(): v for k, v in cfg.get("aliases", {}).items()}
    cfg["cleanup_suffix_regexes"] = [re.compile(p, flags=re.IGNORECASE) for p in cfg.get("cleanup_suffix_patterns", [])]
    cfg["score_regex"] = re.compile(cfg.get("score_pattern", _DEFAULT_SCORE_PATTERN), flags=re.IGNORECASE)
    return cfg


def is_skippable_line(raw):
    lowered = raw.lower().strip()
    if not lowered:
        return True
    if re.fullmatch(r"[-=_.\s]+", raw):
        return True
    if " vs " in lowered:
        return True
    if lowered.startswith(("schedule", "championship", "tournament", "results", "bracket")):
        return True
    if "illinois high school polo association" in lowered or "ihspa" in lowered:
        return True
    if lowered.endswith(":"):
        return True
    return False


def apply_rule_c_cleanup_suffix(name, parser_config):
    out = name.strip()
    for pattern in parser_config["cleanup_suffix_regexes"]:
        out = pattern.sub("", out).strip()
    return out


def apply_rule_d_alias_normalization(name, parser_config):
    canonical = parser_config["aliases"].get(name.lower())
    return canonical if canonical else name


def normalize_team_name(name, parser_config=None):
    cfg = parser_config or load_parser_config()
    cleaned = apply_rule_c_cleanup_suffix(name, cfg)
    return apply_rule_d_alias_normalization(cleaned, cfg)


def parse_game_line_anchored(line, parser_config=None):
    cfg = parser_config or load_parser_config()
    raw = line.strip()
    if is_skippable_line(raw):
        return None
    # Rule A: strict score pattern
    m = cfg["score_regex"].match(raw)
    if m:
        return {
            "team1": normalize_team_name(m.group("team1"), cfg),
            "score1": int(m.group("score1")),
            "team2": normalize_team_name(m.group("team2"), cfg),
            "score2": int(m.group("score2")),
        }
    # Rule B: winner-only notation (d.)
    if re.search(r"\s+d\.\s+", raw, flags=re.IGNORECASE):
        a, b = re.split(r"\s+d\.\s+", raw, maxsplit=1, flags=re.IGNORECASE)
        return {"team1": normalize_team_name(a, cfg), "score1": None, "team2": normalize_team_name(b, cfg), "score2": None}
    return None


def parse_scores_text(text, parser_config=None):
    records = []
    cfg = parser_config or load_parser_config()
    for line in text.splitlines():
        parsed = parse_game_line_anchored(line, cfg)
        if parsed is not None:
            records.append(parsed)
    return pd.DataFrame(records)


def discover_score_files(data_dir=DATA_DIR):
    files = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir) if fn.endswith(SCORES_GLOB_SUFFIX)]
    return sorted(files)
