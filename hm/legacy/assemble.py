"""把多段动作对齐、裁剪、拼接成一条序列。

三个操作：
    align   把一段旋转到目标朝向、平移到目标起点
    crop    在离目标距离最近的那一帧截断
    blend   在两段之间插入过渡帧

过渡帧在【旋转空间】插值，不在关节坐标上插值。
原因：在坐标上线性插值会把骨长压短，而且四肢会互相穿过。
"""
import math

import numpy as np

from hm import config

# HumanML3D 22 关节的骨架树：子 -> 父
PARENT = {
    1: 0, 4: 1, 7: 4, 10: 7,          # 左腿
    2: 0, 5: 2, 8: 5, 11: 8,          # 右腿
    3: 0, 6: 3, 9: 6,                 # 脊柱
    12: 9, 15: 12,                    # 颈、头
    13: 9, 16: 13, 18: 16, 20: 18,    # 左臂
    14: 9, 17: 14, 19: 17, 21: 19,    # 右臂
}


def _depth(j):
    d = 0
    while j in PARENT:
        j = PARENT[j]
        d += 1
    return d


BONES = sorted(PARENT.items(), key=lambda kv: _depth(kv[0]))


def heading_of(seg):
    """用骨盆首末帧的水平位移方向当作朝向。位移太小时返回 None。"""
    seg = np.asarray(seg)
    d = seg[-1, 0, [0, 2]] - seg[0, 0, [0, 2]]
    if np.linalg.norm(d) < 0.25:
        return None
    return math.atan2(float(d[0]), float(d[1]))


def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _rot_y(J, ang):
    c, s = math.cos(ang), math.sin(ang)
    out = J.copy()
    x, z = J[..., 0].copy(), J[..., 2].copy()
    out[..., 0] = c * x + s * z
    out[..., 2] = -s * x + c * z
    return out


def align(seg, start_xy, want_heading):
    """把一段平移到 start_xy，并让它的行进方向等于 want_heading。"""
    seg = np.asarray(seg, dtype=np.float64).copy()
    have = heading_of(seg)
    if have is not None:
        seg = _rot_y(seg, _wrap(want_heading - have))
    # 把首帧骨盆的水平位置搬到 start_xy
    seg[..., 0] -= seg[0, 0, 0] - start_xy[0]
    seg[..., 2] -= seg[0, 0, 2] - start_xy[1]
    return seg


def crop_to_distance(seg, target):
    """在离起点 target 米最近的那一帧截断。target <= 0 时不裁剪。"""
    if target is None or target <= 0.05:
        return seg, False
    seg = np.asarray(seg)
    root = seg[:, 0, :][:, [0, 2]]
    d = np.linalg.norm(root - root[0], axis=1)
    lo = max(2, int(0.4 * len(seg)))
    k = lo + int(np.argmin(np.abs(d[lo:] - target)))
    if k >= len(seg) - 1:
        return seg, False
    return seg[:k + 1], True


def _smooth(u):
    return u * u * (3.0 - 2.0 * u)


def _quat_from_to(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -1.0 + 1e-9:
        ax = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, np.array([0.0, 1.0, 0.0]))
        ax = ax / np.linalg.norm(ax)
        return np.array([0.0, ax[0], ax[1], ax[2]])
    ax = np.cross(a, b)
    q = np.array([1.0 + d, ax[0], ax[1], ax[2]])
    return q / np.linalg.norm(q)


def _slerp_pow(q, u):
    w = float(np.clip(q[0], -1.0, 1.0))
    ang = math.acos(w)
    if ang < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = q[1:] / math.sin(ang)
    half = ang * u
    return np.array([math.cos(half), *(axis * math.sin(half))])


def _rotate(v, q):
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def blend(pa, pb, n):
    """从姿势 pa 过渡到 pb，返回 n 帧 (22,3)。骨长保持不变。"""
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    bone = {}
    for c, p in BONES:
        da, db = pa[c] - pa[p], pb[c] - pb[p]
        la, lb = np.linalg.norm(da), np.linalg.norm(db)
        q = _quat_from_to(da / la, db / lb) if la > 1e-9 and lb > 1e-9 else None
        bone[c] = (da, db, la, lb, q)
    out = []
    for k in range(1, n + 1):
        s = _smooth(k / float(n + 1))
        pose = np.zeros((22, 3), dtype=np.float64)
        pose[0] = pa[0] * (1.0 - s) + pb[0] * s
        for c, p in BONES:
            da, db, la, lb, q = bone[c]
            if q is None:
                pose[c] = pose[p] + (da * (1.0 - s) + db * s)
                continue
            L = la * (1.0 - s) + lb * s
            d = _rotate(da / la, _slerp_pow(q, s))
            pose[c] = pose[p] + (d / (np.linalg.norm(d) + 1e-12)) * L
        out.append(pose)
    return np.stack(out, axis=0)


def pose_gap(pa, pb):
    """两段端点姿势的差异：每根骨骼方向的平均夹角（度）。"""
    ang = []
    for c, p in BONES:
        da, db = pa[c] - pa[p], pb[c] - pb[p]
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        if na < 1e-9 or nb < 1e-9:
            continue
        cos = float(np.clip(np.dot(da / na, db / nb), -1.0, 1.0))
        ang.append(np.degrees(np.arccos(cos)))
    return float(np.mean(ang)) if ang else 0.0


def adaptive_blend(pa, pb, base=18, lo=6, hi=24):
    """按两端姿势差异决定过渡帧数。

    两端姿势接近时，长过渡等于插出一段静止，看起来就是卡住。
    所以差异小就少插几帧，差异大才多插。
    """
    gap = pose_gap(pa, pb)
    if gap < 8.0:
        n = lo
    elif gap > 25.0:
        n = hi
    else:
        n = int(round(lo + (hi - lo) * (gap - 8.0) / 17.0))
    return max(lo, min(hi, n)), gap


def build(segments, plans, blend_frames=18, adaptive=True):
    """多段 → 一条序列。

    segments: [(T,22,3), ...] 每段的原始关节
    plans:    [{"start": (x,z), "end": (x,z), "heading": float}, ...]
    adaptive: 过渡帧数按两端姿势差异自适应
    返回 (整条序列, 每段信息列表)
    """
    parts, info = [], []
    prev = None
    for i, (seg, plan) in enumerate(zip(segments, plans)):
        sx, sz = plan["start"]
        ex, ez = plan["end"]
        dist = math.hypot(ex - sx, ez - sz)
        head = math.atan2(ex - sx, ez - sz) if dist > 0.05 else plan.get("heading", 0.0)
        a = align(seg, (sx, sz), head)
        a, cropped = crop_to_distance(a, dist)
        n_blend, gap = (0, 0.0)
        if prev is not None:
            if adaptive:
                n_blend, gap = adaptive_blend(prev[-1], a[0], base=blend_frames)
            else:
                n_blend = blend_frames
            if n_blend > 0:
                parts.append(blend(prev[-1], a[0], n_blend))
        parts.append(a)
        info.append({
            "index": i,
            "frames": len(a),
            "planned_m": round(dist, 3),
            "actual_m": round(float(np.linalg.norm(a[-1, 0, [0, 2]] - a[0, 0, [0, 2]])), 3),
            "cropped": cropped,
            "blend_in": n_blend,
            "pose_gap_deg": round(gap, 1),
        })
        prev = a
    return np.concatenate(parts, axis=0), info


def selftest():
    """自检：三段拼接后，每段行程接近规划，过渡帧骨长不塌。"""
    from hm import generate, to_joints
    ok = True
    F = generate.generate(["a person walks forward"] * 2, [56, 56])
    J = [to_joints.from_feats(f) for f in F]
    plans = [
        {"start": (0.0, 0.0), "end": (0.0, 2.4), "heading": 0.0},
        {"start": (0.0, 2.4), "end": (0.0, 4.8), "heading": 0.0},
    ]
    seq, info = build(J, plans, blend_frames=18)
    for r in info:
        if abs(r["actual_m"] - r["planned_m"]) > 0.6:
            ok = False
            print("  段%d 行程偏离: 规划%.2f 实际%.2f" % (
                r["index"], r["planned_m"], r["actual_m"]))
    # 骨长检查
    for c, p in BONES:
        if np.linalg.norm(seq[:, c] - seq[:, p], axis=-1).min() < 0.02:
            ok = False
            print("  关节%d 骨长塌陷" % c)
            break
    print("  装配自检: %s  总帧 %d  运动 %.2f m" % (
        "通过" if ok else "不通过", len(seq),
        float(np.linalg.norm(seq[-1, 0, [0, 2]] - seq[0, 0, [0, 2]]))))
    return ok


if __name__ == "__main__":
    selftest()
