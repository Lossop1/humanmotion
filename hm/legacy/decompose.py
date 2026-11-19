"""过渡分解：先把整体偏航单独拿出来，再处理残余姿态。

为什么要分解
------------
之前每根骨骼各自找最短弧旋转。当两帧的整体朝向差接近 180 度时有两个问题：
  1 姿态差被低估。22 根骨骼方向夹角的加权平均把整体偏航稀释掉了。
    实测过渡两端实际转了 181 度，pose_gap 只报 59.6 度，差了 3 倍。
  2 旋转轴不稳定。接近 180 度时最短弧的旋转轴是病态的，
    插到中间会突然翻面：实测出现过单帧 171.9 度的跳变。

分解之后：
  整体偏航   绕竖直轴一个标量角度，线性插值，永远不会翻面
  残余姿态   偏航对齐后剩下的差异，角度小、数值稳定
"""
import math

import numpy as np

from hm import config
from hm.legacy.assemble import BONES


def body_yaw(J_frame):
    """用左右髋连线算出这一帧的身体朝向（弧度）。"""
    J = np.asarray(J_frame, dtype=np.float64)
    across = J[2] - J[1]
    return math.atan2(float(across[2]), float(-across[0]))


def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def rotate_yaw(J, ang):
    """把整帧绕竖直轴旋转 ang 弧度，绕骨盆。"""
    J = np.asarray(J, dtype=np.float64).copy()
    c, s = math.cos(ang), math.sin(ang)
    pivot = J[0].copy()
    v = J - pivot
    out = v.copy()
    out[:, 0] = c * v[:, 0] + s * v[:, 2]
    out[:, 2] = -s * v[:, 0] + c * v[:, 2]
    return out + pivot


def yaw_gap(pa, pb):
    """两帧的整体偏航差（度，取值 -180..180）。"""
    return math.degrees(_wrap(body_yaw(pb) - body_yaw(pa)))


def decompose(pa, pb):
    """返回 (偏航差弧度, 对齐偏航后的 pb)。"""
    dy = _wrap(body_yaw(pb) - body_yaw(pa))
    return dy, rotate_yaw(pb, -dy)


def bone_gap(pa, pb):
    """偏航对齐后，按骨长加权的骨骼方向差 + 最差骨头（度）。"""
    ang, w = [], []
    for c, p in BONES:
        da, db = pa[c] - pa[p], pb[c] - pb[p]
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        if na < 1e-9 or nb < 1e-9:
            continue
        cos = float(np.clip(np.dot(da / na, db / nb), -1.0, 1.0))
        ang.append(math.degrees(math.acos(cos)))
        w.append(0.5 * (na + nb))
    if not ang:
        return 0.0, 0.0
    a = np.asarray(ang)
    wt = np.asarray(w)
    return float((a * wt).sum() / wt.sum()), float(a.max())


def total_gap(pa, pb):
    """过渡需要处理的总量。

    返回 dict：
        yaw_deg   整体偏航差（度，带符号）
        bone_deg  偏航对齐后的骨骼差（加权平均）
        bone_max  偏航对齐后的最差骨头
        span_deg  需要处理的最大角距 = max(|偏航|, 骨骼差)
    """
    dy, pb_aligned = decompose(pa, pb)
    wmean, wmax = bone_gap(pa, pb_aligned)
    yaw_deg = math.degrees(dy)
    span = max(abs(yaw_deg), wmean)
    return {
        "yaw_deg": yaw_deg,
        "bone_deg": wmean,
        "bone_max": wmax,
        "span_deg": span,
    }


def selftest():
    """自检：整体转了 180 度的两帧，必须报出 180 度量级的 span。"""
    import numpy as np
    T = np.zeros((22, 3))
    T[:, 1] = np.linspace(0.9, 1.0, 22)
    T[1] = [-0.1, 0.9, 0.0]
    T[2] = [0.1, 0.9, 0.0]
    T[4] = [-0.1, 0.5, 0.0]
    T[5] = [0.1, 0.5, 0.0]
    T[7] = [-0.1, 0.05, 0.0]
    T[8] = [0.1, 0.05, 0.0]
    T[12] = [0.0, 1.4, 0.0]
    R = rotate_yaw(T, math.pi)          # 整体转 180 度

    g = total_gap(T, R)
    ok = True
    if abs(abs(g["yaw_deg"]) - 180.0) > 5.0:
        ok = False
        print("  偏航差算错: %.1f 度" % g["yaw_deg"])
    if g["span_deg"] < 170.0:
        ok = False
        print("  span 被低估: %.1f 度" % g["span_deg"])
    if g["bone_deg"] > 10.0:
        ok = False
        print("  对齐偏航后骨骼差应接近 0，实际 %.1f 度" % g["bone_deg"])
    print("  偏航 %.1f 度  对齐后骨骼差 %.1f 度  span %.1f 度" % (
        g["yaw_deg"], g["bone_deg"], g["span_deg"]))
    print("  过渡分解自检: %s" % ("通过" if ok else "不通过"))
    return ok


if __name__ == "__main__":
    selftest()