import re

SCORES_CSV = "scores.csv"
CONFIG_JSON = "model_config.json"
SCORES_GLOB_SUFFIX = "_scores_illpolo.txt"
DATA_DIR = "."
RESULT_PATTERN = re.compile(
    r"^\s*(?P<team1>.+?)\s+(?P<score1>\d+)\s+"
    r"(?P<team2>.+?)\s+(?P<score2>\d+)\s*"
    r"(?:(?:\((?:OT|SO)\))|(?:OT)|(?:SO)|(?:\([^)]*OT[^)]*\))|(?:\((?:\d+(?:st|nd|rd|th)\s+Place|Final)\)))?\s*$",
    flags=re.IGNORECASE,
)
TRAILING_PLACEMENT_TAG = re.compile(r"\s*\((?:\d+(?:st|nd|rd|th)\s+Place|Final)\)\s*$", flags=re.IGNORECASE)
TEAM_ALIASES = {
    "chicago u": "U-Chicago",
    "chicago-u": "U-Chicago",
}
