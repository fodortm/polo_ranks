import os
import re

import pandas as pd

from config.constants import DATA_DIR, RESULT_PATTERN, SCORES_GLOB_SUFFIX, TEAM_ALIASES, TRAILING_PLACEMENT_TAG


def normalize_team_name(name):
    cleaned = TRAILING_PLACEMENT_TAG.sub("", name).strip()
    canonical = TEAM_ALIASES.get(cleaned.lower())
    return canonical if canonical else cleaned


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


def parse_game_line_anchored(line):
    raw = line.strip()
    if is_skippable_line(raw):
        return None
    if re.search(r"\s+d\.\s+", raw, flags=re.IGNORECASE):
        a, b = re.split(r"\s+d\.\s+", raw, maxsplit=1, flags=re.IGNORECASE)
        return {"team1": normalize_team_name(a.strip()), "score1": None, "team2": normalize_team_name(b.strip()), "score2": None}
    m = RESULT_PATTERN.match(raw)
    if not m:
        return None
    return {
        "team1": normalize_team_name(m.group("team1")),
        "score1": int(m.group("score1")),
        "team2": normalize_team_name(m.group("team2")),
        "score2": int(m.group("score2")),
    }


def parse_scores_text(text):
    records = []
    for line in text.splitlines():
        parsed = parse_game_line_anchored(line)
        if parsed is not None:
            records.append(parsed)
    return pd.DataFrame(records)


def discover_score_files(data_dir=DATA_DIR):
    files = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir) if fn.endswith(SCORES_GLOB_SUFFIX)]
    return sorted(files)
