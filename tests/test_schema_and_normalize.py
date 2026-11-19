from hm import prompt_normalize, sb_schema


def _sb(*segs):
    return {"segments": list(segs)}


def test_blank_segment_has_no_fake_values():
    s = sb_schema.blank_segment("a", 1, "walk", "向前走")
    assert s["object"] is None and s["spatial_target"] is None and s["facing"] is None


def test_validate_ok():
    ok, errs, _ = sb_schema.validate(_sb(sb_schema.blank_segment("a", 1, "walk", "走")))
    assert ok and not errs


def test_validate_rejects_empty():
    ok, errs, _ = sb_schema.validate({"segments": []})
    assert not ok and errs


def test_validate_missing_required_and_duplicate_id():
    a = sb_schema.blank_segment("a", 1, "walk", "走")
    b = sb_schema.blank_segment("a", 2, "turn", "")
    ok, errs, _ = sb_schema.validate(_sb(a, b))
    assert not ok
    assert any("source_text" in e for e in errs)
    assert any("重复" in e for e in errs)


def test_validate_order_must_be_contiguous():
    a = sb_schema.blank_segment("a", 1, "walk", "走")
    b = sb_schema.blank_segment("b", 3, "turn", "转")
    ok, errs, _ = sb_schema.validate(_sb(a, b))
    assert not ok and any("order" in e for e in errs)


def test_validate_spatial_target_needs_xz():
    a = sb_schema.blank_segment("a", 1, "walk", "走")
    a["spatial_target"] = {"x": 1}
    ok, _, _ = sb_schema.validate(_sb(a))
    assert not ok


def test_validate_enum_violations_are_warnings_not_errors():
    a = sb_schema.blank_segment("a", 1, "walk", "走")
    a["speed"] = "warp"
    ok, _, warns = sb_schema.validate(_sb(a))
    assert ok and warns


# ---- prompt_normalize ----

def test_normalize_names_punching_hand():
    new, why = prompt_normalize.normalize("a person punches forward")
    assert "punch" in new and "hand" in new and why


def test_normalize_leaves_prompt_with_limb_alone():
    p = "a person punches with the right fist"
    assert prompt_normalize.normalize(p) == (p, None)


def test_normalize_strips_in_place_from_stop_prompt():
    new, why = prompt_normalize.normalize("a person slows down and stops in place")
    assert new == "a person slows down and stops" and why


def test_normalize_is_idempotent():
    once, _ = prompt_normalize.normalize("a person punches forward")
    twice, why = prompt_normalize.normalize(once)
    assert twice == once and why is None
