"""The page files against the brief: identity, surfaces, copy, and what must not be there."""

from __future__ import annotations

import re
from importlib.resources import files

from dpo.console.copy import CARDS, STRINGS
from dpo.console.quantities import GRAINS


def _read(name: str) -> str:
    return files("dpo.console").joinpath(name).read_text(encoding="utf-8")


def _uncommented(css: str) -> str:
    """The stylesheet without its commentary, so prose about a rule is not a rule."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _block(css: str, selector: str) -> str:
    """The declarations of one rule, matched on the whole selector."""
    pattern = rf"(?:^|\}}|\*/)\s*{re.escape(selector)}\s*(?:,[^{{]*)?\{{([^}}]*)\}}"
    match = re.search(pattern, css, re.MULTILINE)
    assert match, f"no rule for {selector}"
    return match.group(1)


HTML = _read("console.html")
IDENTITY = _uncommented(_read("identity.css"))
LAYOUT = _uncommented(_read("console.css"))
SCRIPT = _read("console.js")
CSS = IDENTITY + LAYOUT


class TestServedFiles:
    """Each file is served as itself, with its own media type."""

    def test_the_page_links_exactly_the_files_the_app_serves(self) -> None:
        from dpo.console.app import STATIC_FILES

        linked = set(re.findall(r'(?:href|src)="/([^"]+)"', HTML))
        assert linked == set(STATIC_FILES)


class TestPalette:
    """§6 Table 4: seven tokens and no others."""

    TOKENS = {
        "--ground": "#DFE2DE",
        "--panel": "#F1F2EF",
        "--ink": "#191D1C",
        "--dim": "#6E7674",
        "--rule": "#BFC5C0",
        "--sodium": "#B96F14",
        "--teal": "#1F6663",
    }

    def test_every_token_is_declared_with_the_hex_the_brief_gives(self) -> None:
        for token, hex_value in self.TOKENS.items():
            assert re.search(rf"{token}:\s*{hex_value};", IDENTITY, re.IGNORECASE)

    def test_no_colour_is_used_that_is_not_one_of_the_seven(self) -> None:
        declared = {value.lower() for value in self.TOKENS.values()}
        used = {match.lower() for match in re.findall(r"#[0-9A-Fa-f]{3,8}", CSS)}
        assert used <= declared

    def test_sodium_is_only_on_the_primary_action_and_the_momentary_solo(self) -> None:
        """§6: sodium appears only on the primary action and the solo state."""
        rules = [block for block in CSS.split("}") if "--sodium" in block and ":root" not in block]
        for block in rules:
            assert ".primary" in block or ".solo" in block

    def test_teal_is_declared_for_the_bench_and_never_used_on_a_session_screen(self) -> None:
        assert "--teal" in IDENTITY
        assert "var(--teal)" not in CSS


class TestSurfaces:
    """§6: flat, zero radius, 1px hairlines in rule, no shadows, gradients, or icons."""

    def test_no_border_radius_is_ever_set_to_anything_but_zero(self) -> None:
        for value in re.findall(r"border-radius:\s*([^;]+);", CSS):
            assert value.strip() == "0"

    def test_there_are_no_shadows_or_gradients(self) -> None:
        assert "box-shadow" not in CSS and "text-shadow" not in CSS
        assert "gradient" not in CSS

    def test_every_border_is_a_one_pixel_hairline_in_rule_or_a_state_colour(self) -> None:
        for value in re.findall(r"border(?:-top|-right|-bottom|-left)?:\s*([^;]+);", CSS):
            stripped = value.strip()
            if stripped == "0":
                continue
            assert stripped.startswith("1px") or stripped.startswith("2px")

    def test_no_icon_font_or_svg_sneaks_in(self) -> None:
        assert "<svg" not in HTML and "svg" not in SCRIPT.lower().replace("viewport", "")
        assert "font-awesome" not in CSS and "material-icons" not in CSS


class TestType:
    """§6: Archivo for display, IBM Plex Mono for labels, IBM Plex Sans for body."""

    def test_the_three_faces_are_declared_with_fallbacks(self) -> None:
        assert "Archivo" in IDENTITY and "IBM Plex Mono" in IDENTITY and "IBM Plex Sans" in IDENTITY
        for family in ("--display", "--mono", "--body"):
            declaration = re.search(rf"{family}:\s*([^;]+);", IDENTITY)
            assert declaration and "," in declaration.group(1)

    def test_no_stylesheet_is_fetched_over_the_network(self) -> None:
        """The kiosk runs offline (§7); a blocked @import would swap the face mid-study."""
        assert "@import" not in CSS
        assert "fonts.googleapis" not in CSS + HTML

    def test_the_caption_band_is_the_display_face_at_reading_size(self) -> None:
        block = _block(IDENTITY, ".caption-prose")
        assert "var(--display)" in block
        assert "font-weight: 600" in block
        assert re.search(r"font-size:\s*2[0-2]px", block)

    def test_eyebrows_are_mono_ten_letterspaced_uppercase_and_dim(self) -> None:
        block = _block(IDENTITY, ".eyebrow")
        assert "var(--mono)" in block
        assert "font-size: 10px" in block
        assert "letter-spacing" in block
        assert "text-transform: uppercase" in block
        assert "var(--dim)" in block


class TestLayoutOrder:
    """§3.1: phase rail, stage, progress bar, caption band, console, skeleton, actions."""

    def test_the_shell_holds_them_in_the_briefed_order(self) -> None:
        order = ["rail", "stage", "progress", "band", "console", "skeleton", "actions"]
        positions = [HTML.index(f'id="{name}"') for name in order]
        assert positions == sorted(positions)

    def test_the_caption_band_reserves_its_height_so_the_layout_never_jumps(self) -> None:
        assert "--caption-band" in IDENTITY
        assert re.search(r"\.band\s*\{[^}]*min-height:\s*var\(--caption-band\)", LAYOUT, re.DOTALL)

    def test_the_stage_carries_only_the_footage_a_shot_label_and_the_caption(self) -> None:
        stage = HTML[HTML.index('id="stage"') : HTML.index('id="progress"')]
        for forbidden in ("mask", "overlay", "annotation", "canvas"):
            assert forbidden not in stage


class TestConsoleComponents:
    def test_the_three_controls_are_sources_balance_and_detail(self) -> None:
        for name in ("sources-control", "balance-control", "detail-control"):
            assert f'id="{name}"' in HTML

    def test_mute_is_shown_as_a_strike_through_so_the_source_stays_legible(self) -> None:
        block = _block(LAYOUT, ".strip .source.muted")
        assert "line-through" in block

    def test_solo_takes_the_sodium_accent_and_dims_the_other_rows(self) -> None:
        assert "var(--sodium)" in _block(LAYOUT, ".strip .source.solo")
        assert ".rows.soloing .row:not(.solo)" in LAYOUT

    def test_solo_releases_on_pointer_up_and_cannot_become_a_resting_state(self) -> None:
        assert "pointerup" in SCRIPT and "pointercancel" in SCRIPT
        assert "solo = null" in SCRIPT

    def test_the_crossfader_segments_take_their_width_from_the_regime_spans(self) -> None:
        assert "flex-grow:" in SCRIPT
        assert "span[1] - regime.span[0]" in SCRIPT

    def test_the_crossfader_carries_the_two_poles_and_no_numeral(self) -> None:
        assert 'id="pole-eye"' in HTML and 'id="pole-ear"' in HTML
        assert STRINGS["poles"] == {"eye": "EYE", "ear": "EAR"}

    def test_detail_is_four_equal_detents(self) -> None:
        block = _block(LAYOUT, ".detents .detent")
        assert "flex: 1 1 0" in block
        assert len(GRAINS) == 4

    def test_at_the_atmospheric_detent_the_other_two_controls_stop_responding(self) -> None:
        block = _block(LAYOUT, ".console .control.inert")
        assert "opacity" in block
        assert "pointer-events: none" in block
        assert 'className = unnamed ? "control sources inert"' in SCRIPT

    def test_the_skeleton_is_a_numbered_list(self) -> None:
        assert "counter-reset: row" in LAYOUT and "counter-increment: row" in LAYOUT
        assert "<ol" in HTML

    def test_the_skeleton_rows_carry_their_two_phrases_as_a_right_aligned_eyebrow(self) -> None:
        block = _block(LAYOUT, ".rows .row .phrases")
        assert "text-align: right" in block
        assert 'source.band + " · " + source.register' in SCRIPT


class TestActions:
    def test_keep_is_the_one_filled_action_and_show_is_outlined(self) -> None:
        assert 'class: "button primary"' in SCRIPT
        # Show caption is built with the plain button class.
        show = SCRIPT[SCRIPT.index("state.strings.actions.show") - 200 :][:400]
        assert "primary" not in show.split("state.strings.actions.keep")[0]

    def test_keep_is_disabled_until_the_shown_prose_matches_the_settings(self) -> None:
        assert 'disabled: isFresh(a) && !a.locked ? null : "disabled"' in SCRIPT

    def test_freshness_is_a_comparison_so_changing_back_needs_no_new_request(self) -> None:
        assert "a.band.key === keyOf(a)" in SCRIPT

    def test_the_criterion_sentence_sits_under_the_buttons(self) -> None:
        assert HTML.index('id="buttons"') < HTML.index('id="criterion"')

    def test_no_reroll_affordance_exists_anywhere(self) -> None:
        for text in (HTML, CSS, SCRIPT):
            assert "reroll" not in text.lower() and "regenerate" not in text.lower()

    def test_keep_is_locked_while_the_next_shot_opens_and_a_revision_survives_a_reload(self) -> None:
        # A second Keep while /api/shot is pending would commit the shot twice
        # and skip the one after it; a reload during a revision would reopen
        # the shot as a fresh commit.
        keep = SCRIPT[SCRIPT.index("function onKeep") :].split("\n}\n", 1)[0]
        assert "a.locked" in keep and "a.locked = true" in keep
        assert "state.snapshot.revising" in SCRIPT
        resume = SCRIPT[SCRIPT.index('state.snapshot.screen === "author"') :][:200]
        assert "state.snapshot.revising" in resume

    def test_the_writing_state_is_on_the_button_not_a_global_spinner(self) -> None:
        assert "a.writing ? state.strings.busy.writing" in SCRIPT
        assert "spinner" not in SCRIPT.lower()


class TestCopy:
    def test_the_page_holds_no_second_copy_of_the_strings(self) -> None:
        """Table 3 lives in copy.py; the page reads it from /api/session."""
        for string in (STRINGS["criterion"], STRINGS["probe"], STRINGS["error"]):
            assert string not in SCRIPT
        assert "state.strings" in SCRIPT

    def test_the_criterion_and_probe_are_the_table_three_strings(self) -> None:
        assert STRINGS["criterion"] == "Is this the caption you'd want for this moment?"
        assert STRINGS["probe"].startswith("Across the clip, what, if anything,")

    def test_the_phase_rail_reads_as_the_table_gives_it(self) -> None:
        assert STRINGS["phase_rail"] == ["1 watch", "2 author", "3 watch again", "4 check"]

    def test_the_busy_strings_are_both_present(self) -> None:
        assert STRINGS["busy"] == {"writing": "Writing…", "preparing": "Preparing comparison…"}

    def test_the_error_states_what_happened_and_the_way_forward_without_apology(self) -> None:
        assert STRINGS["error"] == "Caption request failed. Check the connection and try again."
        assert "sorry" not in STRINGS["error"].lower()

    def test_the_one_string_the_page_must_hold_is_the_table_string(self) -> None:
        # /api/session is gated on the participant, so the line for a missing
        # one is the one string the script carries; it must be copy.py's.
        match = re.search(r'const PARTICIPANT_MISSING = "([^"]+)";', SCRIPT)
        assert match and match.group(1) == STRINGS["participant_missing"]
        assert SCRIPT.count(STRINGS["participant_missing"]) == 1

    def test_every_instruction_card_the_screens_need_exists(self) -> None:
        assert set(CARDS) == {"intro", "author", "check"}
        for card in CARDS.values():
            assert card["heading"] and card["body"]


class TestAccessibility:
    def test_focus_is_visible_on_every_interactive_element(self) -> None:
        assert ":focus-visible" in IDENTITY and "outline:" in IDENTITY

    def test_touch_targets_meet_the_tablet_minimum(self) -> None:
        assert "44px" in _block(IDENTITY, ".button")
        for selector in (".strip .source", ".segments", ".detents .detent", ".rows .row"):
            assert "44px" in _block(LAYOUT, selector)

    def test_reduced_motion_is_respected(self) -> None:
        assert "prefers-reduced-motion" in IDENTITY

    def test_selection_is_announced_and_not_carried_by_fill_alone(self) -> None:
        assert "aria-pressed" in SCRIPT
        assert "aria-label" in SCRIPT


class TestScope:
    def test_nothing_from_the_bench_or_the_admin_view_appears(self) -> None:
        page = (HTML + CSS + SCRIPT).lower()
        for out_of_scope in ("residual", "lasso", "contact sheet", "saturation curve", "filmstrip"):
            assert out_of_scope not in page

    def test_no_measured_value_is_rendered_as_a_number(self) -> None:
        for measured in ("alpha", "r_g", "p_g", "share", "presence", "confidence", "energy"):
            assert f"textContent = {measured}" not in SCRIPT
        # alpha exists in the page, but only to carry a selection across a change.
        assert "a.alpha" in SCRIPT
        assert "alpha" not in HTML
