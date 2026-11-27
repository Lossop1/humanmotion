"""可控性执行：朝向、地形、速度的纯几何实现。

单独成模块的原因：装配层 assemble_params 会导入 fit/motius（依赖 GPU 环境），
而这三件事是纯 numpy 运算，应该能在任何机器上单测。
"""
import math

import numpy as np


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _disp_dir(J, min_m=0.05):
    """这段的水平位移方向（度，世界系：atan2(x, z)）与距离。位移太小返回 (None, 距离)。"""
    d = np.asarray(J[-1, 0] - J[0, 0], dtype=np.float64)
    h = np.array([d[0], 0.0, d[2]])
    n = float(np.linalg.norm(h))
    if n < min_m:
        return None, n
    return math.degrees(math.atan2(float(h[0]), float(h[2]))), n


def _rot_y(J, ang_deg):
    """绕竖直轴把整段旋转 ang_deg（绕该段起始骨盆位置）。"""
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    piv = np.asarray(J[0, 0], dtype=np.float64).copy()
    out = np.array(J, dtype=np.float64, copy=True)
    for i in range(len(out)):
        for k in range(22):
            v = out[i, k] - piv
            out[i, k] = piv + np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])
    return out


DIR_ANGLE = {"forward": 0.0, "left": 90.0, "right": -90.0, "backward": 180.0}


def junction_frames(gap_deg, rate_deg_per_frame, cap=120, ratio_max=1.4, floor=2):
    """过渡帧数：慢到不超两侧速率的 ratio_max 倍，快也别超过 cap 帧。

    反例（轮25 S20）：姿态差 120.6°、两侧 0.6–1.0 度/帧，cap=60 时过渡被迫
    2.01 度/帧（速率比 2.01 不合格）。cap=120 时 n=ceil(120.6/(1.4*1.0))=87，
    速率比降到 1.39。
    """
    rate = max(float(rate_deg_per_frame), 1e-6)
    n = max(floor, int(round(float(gap_deg) / rate)))
    need = int(__import__("math").ceil(float(gap_deg) / (ratio_max * rate)))
    return int(min(int(cap), max(n, need)))


def enforce_facing(J, direction, global_forward_deg):
    """把这一段的世界行进方向转到请求的方向上，返回 (J2, info)。

    为什么整体旋转有效：段内姿态与运动的关系不变（侧移还是侧移、倒退还是倒退），
    改的只是这个世界里的行进方向；同向对齐由 align_pair 负责，两者不冲突。
    """
    name = str(direction or "").strip().lower()
    if name not in DIR_ANGLE:
        return J, None
    phi, dist = _disp_dir(J)
    if phi is None:
        return J, {"requested": name, "skipped": "位移过小（%.2f 米）" % dist}
    target = _wrap_deg(global_forward_deg + DIR_ANGLE[name])
    delta = _wrap_deg(target - phi)
    J2 = _rot_y(J, delta) if abs(delta) > 1e-6 else J
    return J2, {"requested": name,
                "before_deg": round(_wrap_deg(phi - global_forward_deg), 1),
                "target_deg": round(DIR_ANGLE[name], 1),
                "applied_deg": round(delta, 1),
                "distance_m": round(dist, 2)}


def apply_terrain(J, ground_y):
    """按指定地面高度整体抬升/下降：以最低关节点对齐，返回 (J2, 修正量)。"""
    if ground_y is None:
        return J, None
    lowest = float(np.min(J[:, :, 1]))
    dy = float(ground_y) - lowest
    if abs(dy) < 1e-6:
        return J, 0.0
    J2 = np.array(J, dtype=np.float64, copy=True)
    J2[:, :, 1] += dy
    return J2, round(dy, 4)


SPEED_FACTOR = {"slow": 1.25, "normal": 1.0, "fast": 0.8}


def _mps(J, fps=20.0):
    """平均水平速度（米/秒）：看整段的位移，不看瞬时抖动。"""
    if len(J) < 2:
        return 0.0
    d = np.asarray(J[-1, 0] - J[0, 0], dtype=np.float64)
    return float(np.hypot(d[0], d[2])) / (len(J) / fps)


def apply_speed(J, speed):
    """按 slow/normal/fast 做时间伸缩，返回 (J2, info)。

    做法：帧数按因子重采样（slow 多帧、fast 少帧），关节位置线性插值；
    为什么不在生成时改：后端没有速度接口，改帧数是唯一可控且可测的手段。
    """
    name = str(speed or "").strip().lower()
    factor = SPEED_FACTOR.get(name)
    if factor in (None, 1.0) or len(J) < 8:
        return J, None
    before = _mps(J)
    n_new = max(8, int(round(len(J) * factor)))
    src = np.linspace(0.0, len(J) - 1.0, n_new)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(J) - 1)
    w = (src - i0)[:, None, None]
    J2 = J[i0] * (1.0 - w) + J[i1] * w
    after = _mps(J2)
    info = {"speed": name, "factor": factor, "frames_before": int(len(J)),
            "frames_after": int(n_new),
            "mps_before": round(before, 2), "mps_after": round(after, 2),
            "err_pct": round(100.0 * (after - before / factor) / max(before / factor, 1e-6), 1)}
    return J2, info


