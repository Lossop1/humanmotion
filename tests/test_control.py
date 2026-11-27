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


def test_enforce_facing_left_rotates_to_plus_90():
    J = _walk_J()                                  # 自然朝 +z（0°）
    out, info = AP.enforce_facing(J, "left", 0.0)
    phi, _d = AP._disp_dir(out)
    assert ((phi - 0.0 + 180.0) % 360.0) - 180.0 == pytest.approx(90.0, abs=2.0)
    assert info["applied_deg"] == pytest.approx(90.0, abs=1e-6)


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


def test_apply_speed_slow_adds_frames_and_slows_down():
    J = _walk_J(n=40)
    before = AP._mps(J)
    out, info = AP.apply_speed(J, "slow")
    assert len(out) > len(J)
    assert AP._mps(out) < before
    assert abs(info["err_pct"]) <= 5.0        # 期望 = 原速度 / 因子，误差应很小


def test_apply_speed_fast_removes_frames_and_speeds_up():
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
