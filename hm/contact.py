"""脚部接触约束：把支撑脚钉住，避免「脚下打滑」和「站在转盘上转」。

问题（使用者两次判定）
    「腿不是正常人会做的动作，而是诡异的平移」——支撑脚在地面上滑动
    「2–4s 双腿着地的完全光滑旋转」「4–6s 单脚着地完全光滑旋转」——
     换朝向时整段刚性旋转，脚不动、身子转，像站在转盘上

做法
    1 找接触：脚踝离地 < 3 厘米且移动慢 → 视为支撑
    2 钉住：支撑脚相邻帧的漂移，用「整体平移」抵消（人整体挪一点，脚就不滑）
    3 旋转在前、钉脚在后：支撑脚被钉住，身体转动就变成「绕着支撑脚转」，
      等价于迈步转身，而不是整体旋转
    4 限幅：单帧修正不超过 3 厘米，避免把大步幅硬掰成碎步

为什么用整体平移而不是 IK
    IK 逐帧解腿更"正确"，但会改变关节角、可能引入新的单帧跳变；
    整体平移只改根轨迹，对姿态零改动，可验证、可回退。
"""
import numpy as np

CONTACT_HEIGHT_M = 0.03      # 离地 3 厘米内算接触
CONTACT_VSPEED_MPS = 0.10    # 接触帧的「竖直」速度上限
# 只看竖直方向：支撑脚打滑时水平速度恰好很大，用水平速度筛会把要修的帧全筛掉
MAX_FIX_M = 0.03             # 单帧最大修正
FEET = (10, 11)              # 左右脚掌索引（22 关节）
# 为什么不用踝（7/8）：站着时踝离地 6–9 厘米，用「距地面 3 厘米」判接触永远判不到，
# 实测按踝算「连续 3 帧支撑」一个区间都没有；脚掌才是真正接触地面的点。


def _ankle_xz(J):
    """脚踝的水平坐标 (T, 2, 2)：第二维是左右脚，第三维是 x/z。"""
    J = np.asarray(J, dtype=np.float64)
    return J[:, list(FEET)][:, :, [0, 2]]


def contact_mask(J, fps=20):
    """(T, 2) 布尔：每帧每只脚是否在支撑。"""
    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    if T < 3:
        return np.zeros((T, 2), dtype=bool)
    ank = J[:, list(FEET), :]
    ground = float(np.min(ank[..., 1]))
    h = ank[..., 1] - ground
    vspeed = np.zeros((T, 2))
    vspeed[1:] = np.abs(np.diff(ank[..., 1], axis=0)) * fps
    return (h < CONTACT_HEIGHT_M) & (vspeed < CONTACT_VSPEED_MPS)


def foot_slide(J, mask=None, fps=20):
    """支撑脚相邻帧的水平漂移（毫米/帧，中位数）。"""
    J = np.asarray(J, dtype=np.float64)
    if mask is None:
        mask = contact_mask(J, fps=fps)
    if len(J) < 3:
        return 0.0
    d = np.linalg.norm(np.diff(_ankle_xz(J), axis=0), axis=2)   # (T-1, 2) 米/帧
    sel = mask[1:] & mask[:-1]
    if not sel.any():
        return 0.0
    return float(np.median(d[sel]) * 1000.0)


def pin_contacts(J, mask=None, max_fix_m=MAX_FIX_M, fps=20):
    """把支撑脚钉住：返回 (修正后的 J, 统计)。"""
    J = np.array(J, dtype=np.float64, copy=True)
    T = len(J)
    if mask is None:
        mask = contact_mask(J, fps=fps)
    if T < 3 or not mask.any():
        return J, {"fixed_m": 0.0, "frames": 0, "skipped": "无接触帧"}
    before = foot_slide(J, mask, fps=fps)
    total = np.zeros(2)
    fixed = 0
    for t in range(1, T):
        both = mask[t] & mask[t - 1]
        if not both.any():
            continue
        # 与上一帧都在支撑的脚，其水平漂移就是要抵消的量（多只脚取平均）
        drift = np.mean(_ankle_xz(J)[t][both] - _ankle_xz(J)[t - 1][both], axis=0)
        corr = -np.clip(drift, -max_fix_m, max_fix_m)
        total += corr
        J[t:, :, 0] += corr[0]
        J[t:, :, 2] += corr[1]
        fixed += 1
    after = foot_slide(J, mask, fps=fps)
    return J, {"fixed_m": round(float(np.linalg.norm(total)), 3),
               "frames": fixed,
               "slide_before_mm": round(before, 1),
               "slide_after_mm": round(after, 1)}


def _rot_y(points, angle_rad, pivot_xz):
    """把点集绕竖直轴旋转（pivot 在 xz 平面内固定）。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    out = points.copy()
    x = points[..., 0] - pivot_xz[0]
    z = points[..., 2] - pivot_xz[1]
    out[..., 0] = pivot_xz[0] + c * x + s * z
    out[..., 2] = pivot_xz[1] - s * x + c * z
    return out


def turn_with_feet(J, delta_deg, mask=None, fps=20):
    """按「绕支撑脚转」的方式完成转身，而不是绕根节点整体旋转。

    做法：把总转角平均分到每一帧，逐帧绕当时的支撑脚位置旋转；
    支撑脚因此在世界坐标里保持不动，身体绕着它摆过去 —— 视觉上是迈步转身，
    不是「站在转盘上」。支撑帧不足时退回绕根旋转。
    """
    J = np.array(J, dtype=np.float64, copy=True)
    T = len(J)
    if T < 3 or abs(delta_deg) < 1e-6:
        return J, {"turned_deg": 0.0, "pivot": "none"}
    if mask is None:
        mask = contact_mask(J, fps=fps)
    step = np.radians(delta_deg) / (T - 1)
    used_feet = 0
    for t in range(1, T):
        if mask[t - 1].any():
            idx = np.nonzero(mask[t - 1])[0][0]
            foot = FEET[idx]
            pivot = J[t - 1, foot][[0, 2]]
            used_feet += 1
        else:
            pivot = J[t - 1, 0][[0, 2]]
        J[t:] = _rot_y(J[t:], step, pivot)
    return J, {"turned_deg": round(float(delta_deg), 1),
               "pivot": "支撑脚 %d 帧 / 根 %d 帧" % (used_feet, T - 1 - used_feet)}


def mesh_foot_slide(verts, thr=0.02):
    """网格级脚部滑动（毫米/帧）：取每帧最低 thr 米内的顶点作为落地部分。

    为什么需要：实测同一段动作，踝级滑动 2–6 毫米/帧，网格级 8–13 毫米/帧，
    使用者看到的「腿在平移」是网格这一层，踝级度量把它低估了 3–4 倍。
    """
    V = np.asarray(verts, dtype=np.float64)
    gy = float(V[..., 1].min())
    prev = None
    vals = []
    for t in range(len(V)):
        low = V[t][V[t][..., 1] < gy + thr]
        if len(low) < 8:
            prev = None
            continue
        c = low.mean(axis=0)[[0, 2]]
        if prev is not None:
            vals.append(float(np.linalg.norm(c - prev)))
        prev = c
    return float(np.median(vals) * 1000.0) if vals else 0.0


def pin_mesh(verts, params, thr=0.02, max_fix_m=0.03):
    """在网格层钉脚：把支撑期脚部的水平漂移，从根平移里扣掉。

    与 pin_contacts 的区别：那个钉的是踝关节，实测留着 8–13 毫米/帧的网格滑动；
    这里直接量网格落地部分，改的是 SMPL 参数的根平移（82:85），不改关节角。
    """
    V = np.asarray(verts, dtype=np.float64)
    P = np.array(params, dtype=np.float64, copy=True)
    T = min(len(V), len(P))
    if T < 3:
        return P, {"slide_before_mm": 0.0, "slide_after_mm": 0.0, "fixed_frames": 0}
    before = mesh_foot_slide(V[:T], thr=thr)
    gy = float(V[..., 1].min())
    prev = None
    total = np.zeros(2)
    fixed = 0
    for t in range(T):
        low = V[t][V[t][..., 1] < gy + thr]
        if len(low) < 8:
            prev = None
            continue
        c = low.mean(axis=0)[[0, 2]]
        if prev is not None:
            corr = -np.clip(c - prev, -max_fix_m, max_fix_m)
            total += corr
            P[t:, 82] += corr[0]
            P[t:, 84] += corr[1]
            fixed += 1
        prev = c
    return P, {"slide_before_mm": round(before, 1), "fixed_frames": fixed,
               "fixed_m": round(float(np.linalg.norm(total)), 3)}
