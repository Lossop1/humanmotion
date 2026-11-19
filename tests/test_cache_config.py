import os
import re
from pathlib import Path

from hm import cache, config

REPO = Path(__file__).resolve().parent.parent


def test_cache_key_is_stable_and_sensitive():
    assert cache.key(("plan", "x")) == cache.key(("plan", "x"))
    assert cache.key(("plan", "x")) != cache.key(("plan", "y"))
    # 分隔符防止拼接歧义：("ab","c") 与 ("a","bc") 不能撞
    assert cache.key(("ab", "c")) != cache.key(("a", "bc"))


def test_cache_json_roundtrip_and_miss():
    k = cache.key(("t", 1))
    assert cache.get_json("t", k) is None
    cache.put_json("t", k, {"a": [1, 2], "中": "文"})
    assert cache.get_json("t", k) == {"a": [1, 2], "中": "文"}


def test_cache_corrupt_json_is_a_miss():
    k = cache.key(("t", 2))
    cache.path("t", k, "json").write_text("{not json", encoding="utf-8")
    assert cache.get_json("t", k) is None


def test_cache_can_be_disabled(monkeypatch):
    assert cache.enabled()
    monkeypatch.setenv("HM_NO_CACHE", "1")
    assert not cache.enabled()


def test_find_blender_env_override(monkeypatch, tmp_path):
    exe = tmp_path / "my-blender"
    monkeypatch.setenv("HM_BLENDER", str(exe))
    assert config.find_blender() == exe


def test_find_blender_scans_deps(monkeypatch, tmp_path):
    name = "blender.exe" if os.name == "nt" else "blender"
    d = tmp_path / "blender-4.5.1-x" / name
    d.parent.mkdir()
    d.write_text("", encoding="utf-8")
    monkeypatch.delenv("HM_BLENDER", raising=False)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    monkeypatch.setattr(config, "DEPS", tmp_path)
    assert config.find_blender() == d


def test_no_hardcoded_machine_paths_in_source():
    """所有路径经 config 或 __file__ 推导，不许写死某台机器。"""
    bad = []
    pat = re.compile(r"/home/chuan|100\.124\.")
    for p in list((REPO / "hm").glob("*.py")) + list(REPO.glob("*.py")):
        if pat.search(p.read_text(encoding="utf-8")):
            bad.append(p.name)
    assert not bad, bad


def test_secret_files_are_not_in_repo():
    assert not (REPO / ".deepseek_key").exists()
    gi = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "deps/" in gi
