"""可控性执行层的单元测试：朝向、地形、速度。

为什么要有这些测试
    这三项以前只存字段、不执行（planner 给了 facing/terrain/speed，装配层没用），
    属于「文档说支持、实际不支持」。测试钉住的不只是函数，而是这条承诺。
"""
import math

import numpy as np
import pytest

from hm import control as AP


def _walk_J(n=40, step=(0.0, 0.0, 0.05), rise=0.0):
    """造一段直线走：根节点每帧移动 step，21 个关节跟随（够用即可）。"""
    J = np.zeros((n, 22, 3))
    for i in range(n):
        base = np.array([step[0] * i, rise * i, step[2] * i])
        J[i, 0] = base
        J[i, 1:] = base + np.array([0.0, 0.9, 0.0])
    return J


def test_disp_dir_reports_axis_and_distance():
    J = _walk_J(step=(0.05, 0.0, 0.0))   # 沿 +x
    phi, dist = AP._disp_dir(J)
    assert phi == pytest.approx(90.0, abs=1e-6)   # atan2(x, z)：+x 是 90°
    assert dist == pytest.approx(0.05 * 39, rel=1e-6)


def test_disp_dir_ignores_tiny_motion():
    J = _walk_J(step=(0.0, 0.0, 0.001))
    phi, _dist = AP._disp_dir(J)
    assert phi is None          # 位移过小不判定向，避免在原地动作上乱转


def test_enforce_facing_forward_leaves_segment_untouched():
    J = _walk_J()
    out, info = AP.enforce_facing(J, "forward", global_forward_deg=0.0)
    assert np.allclose(out, J)
    assert info["applied_deg"] == pytest.approx(0.0, abs=1e-6)


def test_enforce_facing_skips_left_and_right():
    """左右不整体旋转：旋转后身体仍朝行进方向，看起来还是直走。

    实测 "walks to the right" 朝向与行进夹角只有 4–12 度，必须换成 sidesteps 写法。
    """
    J = _walk_J()
    for name in ("left", "right"):
        out, info = AP.enforce_facing(J, name, 0.0)
        assert np.allclose(out, J)
        assert info and "skipped" in info


def test_enforce_facing_backward_rotates_to_180():
    J = _walk_J()
    out, _info = AP.enforce_facing(J, "backward", 0.0)
    phi, _d = AP._disp_dir(out)
    assert abs(((phi - 180.0 + 180.0) % 360.0) - 180.0) <= 2.0


def test_enforce_facing_respects_global_forward():
    J = _walk_J()
    out, _info = AP.enforce_facing(J, "forward", 90.0)   # 世界前方是 90°
    phi, _d = AP._disp_dir(out)
    assert ((phi - 90.0 + 180.0) % 360.0) - 180.0 == pytest.approx(0.0, abs=2.0)


def test_enforce_facing_skips_in_place():
    J = _walk_J()
    out, info = AP.enforce_facing(J, "in_place", 0.0)
    assert np.allclose(out, J) and info is None


def test_apply_terrain_lifts_to_requested_ground():
    J = _walk_J()
    lowest = float(np.min(J[:, :, 1]))
    out, dy = AP.apply_terrain(J, 1.2)
    assert float(np.min(out[:, :, 1])) == pytest.approx(1.2, abs=1e-6)
    assert dy == pytest.approx(1.2 - lowest, abs=1e-6)


def test_apply_terrain_none_means_noop():
    J = _walk_J()
    out, dy = AP.apply_terrain(J, None)
    assert np.allclose(out, J) and dy is None


def test_apply_speed_off_by_default(monkeypatch):
    """默认不伸缩帧数：使用者把「快跑」看成「快放」，那正是帧数伸缩的副作用。"""
    J = _walk_J(n=40)
    out, info = AP.apply_speed(J, "fast")
    assert len(out) == len(J) and info is None


def test_apply_speed_slow_adds_frames_and_slows_down(monkeypatch):
    monkeypatch.setattr(AP, "_USE_WARP", True)
    J = _walk_J(n=40)
    before = AP._mps(J)
    out, info = AP.apply_speed(J, "slow")
    assert len(out) > len(J)
    assert AP._mps(out) < before
    assert abs(info["err_pct"]) <= 5.0        # 期望 = 原速度 / 因子，误差应很小


def test_apply_speed_fast_removes_frames_and_speeds_up(monkeypatch):
    monkeypatch.setattr(AP, "_USE_WARP", True)
    J = _walk_J(n=60)
    before = AP._mps(J)
    out, info = AP.apply_speed(J, "fast")
    assert len(out) < len(J)
    assert AP._mps(out) > before
    assert abs(info["err_pct"]) <= 5.0


def test_apply_speed_normal_is_noop():
    J = _walk_J()
    out, info = AP.apply_speed(J, "normal")
    assert np.allclose(out, J) and info is None


def test_apply_speed_does_not_explode_on_short_segments():
    J = _walk_J(n=5)
    out, _info = AP.apply_speed(J, "fast")
    assert len(out) == len(J)                  # 太短不做伸缩，避免退化成几帧


def test_junction_frames_matches_the_s20_regression():
    # 轮25 S20：姿态差 120.6°、两侧最慢 0.6 度/帧、较快 1.0 度/帧
    n60 = AP.junction_frames(120.6, 1.0, cap=60)
    n120 = AP.junction_frames(120.6, 1.0, cap=120)
    assert 120.6 / n60 > 1.4          # 旧上限下速率比不合格
    assert 120.6 / n120 <= 1.4 + 0.02  # 新上限下合格


def test_junction_frames_floor_and_fast_motion():
    assert AP.junction_frames(1.0, 50.0) >= 2        # 姿态差很小也不少于 2 帧
    assert AP.junction_frames(90.0, 30.0) <= 120      # 动作很快时过渡不应无限长


def test_speed_word_added_once():
    from hm.prompt_normalize import apply_speed_word
    assert apply_speed_word("a person turns around in place", "slow") == \
        "a person turns around in place slowly"
    assert apply_speed_word("a person runs forward fast", "fast") == "a person runs forward fast"
    assert apply_speed_word("a person walks forward", "normal") == "a person walks forward"


def test_sidestep_not_judged_as_walk():
    """侧移不能被走路判据命中：\bsteps?\b 的边界是关键（sidesteps 里没有边界）。"""
    from hm import signature
    sig = {"disp_m": 0.3, "facing_travel_deg": 90.0, "knee_range": 15.0}
    ok, bad, _detail = signature.check("a person sidesteps to the left", sig)
    assert ok is not False, "侧移被判成「没走出去」：%s" % (bad,)


def test_describe_zh_march_and_sidestep():
    from hm.prompt_normalize import describe_zh
    assert describe_zh("a person marches in place with high knees") == ["踏步"]
    assert describe_zh("a person sidesteps to the left") == ["侧移"]
    assert describe_zh("a person walks forward") == ["走"]


def test_ambiguous_walk_back_declared_and_checked():
    """「走回来」必须先声明解释；没声明就判未验证，不能默认当后退。"""
    from hm import ambiguity
    hits = ambiguity.lookup("一个人快速跑过去，然后慢慢走回来")
    assert hits and hits[0][0] == "走回来"
    names = [n for _w, opts in hits for n, _h, _r in opts]
    assert "转身往回走" in names and "后退着回来" in names
    assert ambiguity.declared("走回来（理解为：转身往回走）") == "转身往回走"
    assert ambiguity.declared("走回来") is None
    name, bad = ambiguity.check_interpretation("a person walks backward", {"backward_m": 0.3})
    assert name == "后退着回来" and not bad


def test_usual_reading_of_walk_back():
    """「走回来」的通常用法是转身走回去；只有原句写「倒退着」才用后退。"""
    from hm import ambiguity
    assert ambiguity.usual("走回来") == "转身往回走"
    assert ambiguity.pick("一个人走回来")[0][1] == "转身往回走"
    assert ambiguity.pick("一个人倒退着走回来")[0][1] == "后退着回来"
