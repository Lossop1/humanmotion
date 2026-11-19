"""规划器：兜底路径、密钥读取。这些是 LLM 不可用时整条管线的生命线。"""
import pytest

from hm import config, planner, planner_fallback


def _llm_down(*a, **k):
    raise RuntimeError("LLM 调用失败：402")


def test_plan_falls_back_when_llm_is_down(monkeypatch):
    """回归：_call 里的兜底曾引用未定义的 sentence，LLM 一挂就 NameError。"""
    monkeypatch.setattr(planner, "_call", _llm_down)
    segs, trans, terrain = planner.plan("一个人向前走，然后转身")
    assert [s["key"] for s in segs] == ["walk", "turn"]
    assert trans == []


def test_plan_raises_when_llm_down_and_fallback_cannot_parse(monkeypatch):
    monkeypatch.setattr(planner, "_call", _llm_down)
    with pytest.raises(RuntimeError):
        planner.plan("今天天气不错")


def test_fallback_result_is_not_cached(monkeypatch):
    from hm import cache
    monkeypatch.setattr(planner, "_call", _llm_down)
    planner.plan("一个人向前走")
    assert cache.get_json("plan", cache.key(("plan", "一个人向前走"))) is None


def test_forced_fallback_env(monkeypatch):
    monkeypatch.setenv("HM_PLAN_FALLBACK", "1")
    monkeypatch.setattr(planner, "_call", lambda *a, **k: pytest.fail("不应调用 LLM"))
    segs, _, _ = planner.plan("一个人先向前走，然后出拳，最后转身")
    assert [s["key"] for s in segs] == ["walk", "punch", "turn"]


def test_key_prefers_env_over_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DEPS", tmp_path)
    (tmp_path / ".deepseek_key").write_text("file-key", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    assert planner._key() == "env-key"


def test_key_file_fallback_warns(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DEPS", tmp_path)
    (tmp_path / ".deepseek_key").write_text("file-key\n", encoding="utf-8")
    with pytest.warns(UserWarning):
        assert planner._key() == "file-key"


def test_key_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DEPS", tmp_path)
    assert planner._key() == ""


# ---- 规则兜底本身 ----

@pytest.mark.parametrize("sentence,keys", [
    ("一个人向前走", ["walk"]),
    ("一个人蹲下，然后站起来", ["crouch", "stand_up"]),
    ("一个人快速冲刺几步然后停下", ["sprint", "stand"]),
    ("一个人向前跳两下再后退", ["jump", "walk_back"]),
])
def test_fallback_known_sentences(sentence, keys):
    segs, _, _ = planner_fallback.plan(sentence)
    assert [s["key"] for s in segs] == keys


def test_fallback_unknown_returns_none():
    assert planner_fallback.plan("今天天气不错") == (None, None, None)


def test_fallback_segments_carry_source_text():
    """可追溯：每段的 source 必须是原句里的片段。"""
    sentence = "一个人先向前走，然后出拳，最后转身"
    segs, _, _ = planner_fallback.plan(sentence)
    for s in segs:
        assert s["source"] and s["source"] in sentence
        assert s["frames"] > 0 and s["prompt"]


def test_fallback_on_fresh_set_covers_most():
    """新句集上规则兜底至少能认出 8/10（提交 31e1878 记录为 9/10）。"""
    text = (config.ROOT / "eval_fresh.txt").read_text(encoding="utf-8")
    lines = [x.strip() for x in text.splitlines() if x.strip() and not x.startswith("#")]
    assert len(lines) == 10
    ok = sum(1 for x in lines if planner_fallback.plan(x)[0])
    assert ok >= 8
