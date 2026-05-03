from app.ui import get_primary_nav_options, resolve_legacy_public_target


def test_public_nav_has_no_section_tabs():
    nav = get_primary_nav_options(is_admin_user=False)
    assert nav == ["Rankings"]
    assert "Team Profile" not in nav
    assert "Matchup Insights" not in nav


def test_admin_nav_keeps_internal_access():
    nav = get_primary_nav_options(is_admin_user=True)
    assert nav == ["Rankings", "Admin / Internal"]


def test_legacy_public_sections_redirect_to_rankings():
    profile_redirect = resolve_legacy_public_target({"section": "Profile"}, fallback_team="Evanston")
    matchup_redirect = resolve_legacy_public_target({"primary_nav": "Matchup Insights"}, fallback_team="Evanston")
    assert profile_redirect == {"target_nav": "Rankings", "team": "Evanston"}
    assert matchup_redirect == {"target_nav": "Rankings", "team": "Evanston"}

