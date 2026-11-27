"""补充度量：穿地、脚滑、加速度比、胶囊自相交。

为什么单独一个模块
    signature.py 管的是「动作做出来没有」（语义），这里管的是「动起来像不像话」（质量）。
    两者判据来源不同：这里多数只有相对比较，没有绝对阈值。

术语
    穿地   某帧最低关节低于地面 → 深度（米）与帧数
    脚滑   脚处于接触相位（脚踝高度低于阈值）时，水平方向的位移（毫米/帧）
    加速度比 过渡段平均加速度 / 两侧平均加速度；>1.5 表示过渡里猛甩了一下
    自相交 用骨骼胶囊近似，检查不该相交的骨头对（前臂互穿、手插躯干等）
"""
import math

import numpy as np

# 骨名 → (父关节, 子关节, 半径 m)；关节索引与 hm 里的 22 关节约定一致
CAPSULES = {
    "l_upper": (16, 18, 0.045),
    "l_fore": (18, 20, 0.040),
    "r_upper": (17, 19, 0.045),
    "r_fore": (19, 21, 0.040),
    "l_thigh": (1, 4, 0.080),
    "l_shin": (4, 7, 0.055),
    "r_thigh": (2, 5, 0.080),
    "r_shin": (5, 8, 0.055),
    "torso": (0, 9, 0.130),
    "head": (12, 15, 0.100),
}

PAIRS = [
    ("l_fore", "r_fore"), ("l_upper", "r_upper"), ("l_fore", "r_upper"),
    ("r_fore", "l_upper"), ("l_fore", "torso"), ("r_fore", "torso"),
    ("l_upper", "torso"), ("r_upper", "torso"),
    ("l_fore", "head"), ("r_fore", "head"),
    ("l_thigh", "r_thigh"), ("l_shin", "r_shin"),
    ("l_fore", "l_thigh"), ("r_fore", "r_thigh"),
]

ANKLES = (7, 8)


def _seg_dist(p1, q1, p2, q2):
    """两线段最短距离。"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-9
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t, s = 0.0, min(1.0, max(0.0, -c / a))
        else:
            b = float(np.dot(d1, d2))
            den = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / den)) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return float(np.linalg.norm(c1 - c2))


def ground_penetration(J, ground=0.0):
    """最低关节低于地面的深度与帧数。"""
    low = np.min(J[:, :, 1], axis=1) - float(ground)
    depth = np.maximum(0.0, -low)
    hit = int(np.sum(depth > 1e-3))
    return {"frames": int(len(J)), "frames_below": hit,
            "max_depth_m": round(float(depth.max()) if len(depth) else 0.0, 4),
            "mean_depth_m": round(float(depth[depth > 1e-3].mean()) if hit else 0.0, 4)}


def foot_slide(J, fps=20.0, contact_h=0.06):
    """接触相位内的脚部水平位移（毫米/帧）。

    判据思路：脚在地上时不该横向滑动；滑动越多说明生成/拟合在拖脚。
    只统计脚踝低于 contact_h 的帧对，避免把抬脚算成滑动。
    """
    slides = []
    for a in ANKLES:
        for i in range(len(J) - 1):
            if J[i, a, 1] - 0.0 > contact_h or J[i + 1, a, 1] > contact_h:
                continue
            d = (J[i + 1, a] - J[i, a])[[0, 2]]
            slides.append(float(np.linalg.norm(d)) * 1000.0)   # 每帧毫米
    if not slides:
        return {"contact_frames": 0, "median_mm_per_frame": None, "p90_mm_per_frame": None}
    arr = np.asarray(slides)
    return {"contact_frames": int(len(arr)),
            "median_mm_per_frame": round(float(np.median(arr)), 1),
            "p90_mm_per_frame": round(float(np.percentile(arr, 90)), 1)}


def accel_ratio(J, window=6):
    """过渡段的加速度与全段比较：返回 (过渡平均加速度/全段平均加速度)。"""
    P = np.asarray(J, dtype=np.float64)
    if len(P) < 5:
        return None
    v = np.linalg.norm(np.diff(P[:, :, :], axis=0), axis=2)[:, 1:]   # 每帧每关节位移
    a = np.abs(np.diff(v, axis=0))
    mean_a = float(np.mean(a)) + 1e-9
    mid = a[len(a) // 2 - window: len(a) // 2 + window]
    return round(float(np.mean(mid)) / mean_a, 2) if len(mid) else None


def self_intersection(J):
    """胶囊自相交：返回 (帧数占比, 最大插入深度 m, 涉及的骨头对)。

    注意：这是近似——胶囊半径固定，姿态极端时会把「擦边」判成插入。
    使用时必须配合正负对照（鼓掌应命中、走路不应命中）看灵敏度与误报。
    """
    bad_frames = 0
    worst = 0.0
    pairs_hit = {}
    for t in range(len(J)):
        hit_any = False
        for n1, n2 in PAIRS:
            p1, q1, r1 = J[t, CAPSULES[n1][0]], J[t, CAPSULES[n1][1]], CAPSULES[n1][2]
            p2, q2, r2 = J[t, CAPSULES[n2][0]], J[t, CAPSULES[n2][1]], CAPSULES[n2][2]
            d = _seg_dist(p1, q1, p2, q2)
            pen = (r1 + r2) - d
            if pen > 0:
                hit_any = True
                worst = max(worst, pen)
                pairs_hit["%s-%s" % (n1, n2)] = pairs_hit.get("%s-%s" % (n1, n2), 0) + 1
        bad_frames += 1 if hit_any else 0
    return {"frames": int(len(J)), "frames_hit": bad_frames,
            "hit_ratio": round(bad_frames / max(len(J), 1), 3),
            "max_depth_m": round(worst, 4),
            "pairs": dict(sorted(pairs_hit.items(), key=lambda kv: -kv[1])[:4])}
