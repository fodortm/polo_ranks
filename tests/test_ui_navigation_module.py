from app.ui import (
    get_primary_nav_options,
    resolve_legacy_public_target,
    build_team_slug_lookup,
    build_team_canonical_path,
    build_profile_share_metadata,
)


def test_public_nav_has_no_section_tabs():
    nav = get_primary_nav_options(is_admin_user=False)
    assert nav == ["Rankings", "Team Profiles/Resume", "Sectionals"]
    assert "Matchup Insights" not in nav


def test_admin_nav_keeps_internal_access():
    nav = get_primary_nav_options(is_admin_user=True)
    assert nav == ["Rankings", "Team Profiles/Resume", "Sectionals", "Admin / Internal"]


def test_legacy_public_sections_redirect_to_canonical_destinations():
    profile_redirect = resolve_legacy_public_target({"section": "Profile"}, fallback_team="Evanston")
    matchup_redirect = resolve_legacy_public_target({"primary_nav": "Matchup Insights"}, fallback_team="Evanston")
    assert profile_redirect == {"target_nav": "Team Profiles/Resume", "team": "Evanston"}
    assert matchup_redirect == {"target_nav": "Sectionals", "team": "Evanston"}


def test_primary_nav_presence_regression_guard():
    nav = get_primary_nav_options(is_admin_user=False)
    assert set(nav) == {"Rankings", "Team Profiles/Resume", "Sectionals"}


def test_team_slug_lookup_is_stable_and_unique():
    lookup = build_team_slug_lookup(["New Trier", "New-Trier", "Evanston"])
    assert lookup["Evanston"] == "evanston"
    assert lookup["New Trier"] == "new-trier"
    assert lookup["New-Trier"] == "new-trier-2"


def test_profile_canonical_path_and_share_metadata():
    canonical = build_team_canonical_path("evanston", section="profile", timeframe="last-4-weeks")
    meta = build_profile_share_metadata("Evanston", 3, "BCAR 0.812", "Snapshot", canonical)
    assert canonical == "/teams/evanston/profile?timeframe=last-4-weeks"
    assert meta["canonical_url"] == canonical
    assert "Evanston" in meta["title"]
    assert "BCAR 0.812" in meta["share_text"]
