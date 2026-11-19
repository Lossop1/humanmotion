import pytest

from hm.jsonio import load_json


def test_missing_file_returns_default_silently(tmp_path, capsys):
    assert load_json(tmp_path / "nope.json", default={"x": 1}) == {"x": 1}
    assert capsys.readouterr().err == ""


def test_valid_file(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"a": [1, 2]}', encoding="utf-8")
    assert load_json(p) == {"a": [1, 2]}


def test_corrupt_file_returns_default_and_warns(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{oops", encoding="utf-8")
    assert load_json(p, default="D") == "D"
    assert "JSON 损坏" in capsys.readouterr().err


def test_wrong_top_level_type_counts_as_corrupt(tmp_path, capsys):
    p = tmp_path / "list.json"
    p.write_text("[1, 2]", encoding="utf-8")
    assert load_json(p, default=None, want=dict) is None
    assert "类型不符" in capsys.readouterr().err


def test_warning_can_be_silenced(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{oops", encoding="utf-8")
    load_json(p, warn_on_corrupt=False)
    assert capsys.readouterr().err == ""


def test_programming_errors_are_not_swallowed(tmp_path):
    with pytest.raises(TypeError):
        load_json(12345.6)
