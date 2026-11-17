"""动作签名：把「动作做没做出来」变成可比量，并按规则给判断。

规则表每条五项：(正则, 字段, 阈值, 说明, 方向)
    方向 "min" 该量必须 ≥ 阈值（走出去、举手、下蹲、转向幅度）
    方向 "max" 该量必须 ≤ 阈值（转身前倾、转身后退）
统一按 min 处理过一次，结果「前倾 2.1 度」被判不合格、「前倾 11.8 度」反而放行。
"""
import math
import re
import sys

import numpy as np

SHOULDER = 16
WRIST = 21
PELVIS = 0

RULES = [
    (r"(walks?|walking|runs?|running|sprints?|jogs?|steps?)\s+"
     r"(forward|backward|to the left|to the right)", "disp_m", 0.6, "应走出去", "min"),
    (r"turns? around|turning around", "yaw_deg", 45.0, "应转向", "min"),
    # 阈值取实测分离点：momask 前倾中位 2.0、最大 7.6、后退 ≤-0.01；
    #                     mdm   前倾中位 10.2、后退中位 0.39
    (r"turns? around|turning around", "lean_median", 9.0, "转身应站直", "max"),
    (r"turns? around|turning around", "backward_m", 0.1, "转身不该边退边走", "max"),
    (r"(raises?|raising|lifts?|lifting|waves?|waving)\s+(both\s+)?"
     r"(hands?|arms?|the (right|left) (hand|arm))", "wrist_above", 0.05, "手应举过肩", "min"),
    # 坐下：只认「坐下去」这类，不认「从坐着站起来」——后者结束时不该更低
    (r"sits? down|sitting down|sits? on|takes? a seat|is seated", "pelvis_end_drop", 0.18,
     "结束时重心应降低", "min"),
    (r"bends? forward|bending forward|bends? over|bowing", "lean_delta", 20.0, "躯干应前倾", "min"),
    (r"crouch|squat|kneel", "pelvis_drop", 0.18, "应降低重心", "min"),
    # 踢腿：校准于正对照「kicks forward」最小 0.552、负对照（走/站/原地踏步）最大 0.492
    (r"kicks? forward|kicking forward|does a (front|side) kick", "kick_reach", 0.52,
     "脚应向前踢出", "min"),
    # 出拳：校准后仍不可完全分离（正对照最小 -0.018、负对照最大 0.164），
    # 取 0.25 只作提示：宁可漏报，不做硬门禁
    (r"punches?|punching|throws? a punch", "punch_reach", 0.25, "手应向前伸出", "min"),
]


def _yaw(J):
    v = J[2] - J[1]
    return math.degrees(math.atan2(float(v[0]), float(v[2])))


def _unwrap(angles):
    out = [angles[0]]
    for a in angles[1:]:
        d = a - out[-1]
        while d > 180:
            d -= 360
        while d < -180:
            d += 360
        out.append(out[-1] + d)
    return np.array(out)


def _knee_angle(J, hip, knee, ankle):
    a, b = J[hip] - J[knee], J[ankle] - J[knee]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def signature(J):
    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    root = J[:, PELVIS, :]
    horizon = np.linalg.norm(root[:, [0, 2]] - root[0, [0, 2]], axis=1)
    speed = float(np.mean(np.linalg.norm(np.diff(root, axis=0), axis=1)) * 20) if T > 1 else 0.0
    wrist_above = float(J[:, WRIST, 1].max() - np.median(J[:, SHOULDER, 1]))
    pelvis_drop = float(J[0, PELVIS, 1] - J[:, PELVIS, 1].min())
    pelvis_up = float(J[:, PELVIS, 1].max() - J[0, PELVIS, 1])
    pelvis_end_drop = float(J[0, PELVIS, 1] - J[-1, PELVIS, 1])
    yaws = _unwrap([_yaw(f) for f in J])
    knee = [_knee_angle(f, 1, 4, 7) for f in J]
    tv = J[:, 9] - J[:, 0]
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    lean = np.degrees(np.arccos(np.clip(tv[:, 1], -1, 1)))
    fwd = J[0, 10] - J[0, 7]
    fwd[1] = 0.0
    nf = float(np.linalg.norm(fwd))
    fwd = fwd / nf if nf > 1e-9 else np.array([0.0, 0.0, 1.0])
    rw = J[:, WRIST, :] - J[:, SHOULDER, :]
    return {
        "disp_m": float(horizon.max()),
        "speed_mps": speed,
        "wrist_above": wrist_above,
        "pelvis_drop": pelvis_drop,
        "pelvis_up": pelvis_up,
        "pelvis_end_drop": pelvis_end_drop,
        "yaw_deg": float(abs(yaws[-1] - yaws[0])),
        "knee_range": float(max(knee) - min(knee)),
        "lean_delta": float(lean.max() - lean[0]),
        "lean_median": float(np.median(lean)),
        "lean_max": float(lean.max()),
        "backward_m": float(-(J[-1, 0] - J[0, 0]) @ fwd),
        "punch_reach": float((rw @ fwd).max()),
        # 踢腿：脚踝相对同侧髋，沿身体前方的最远伸展（取两腿较大者）
        "kick_reach": float(max(((J[:, 10, :] - J[:, 1, :]) @ fwd).max(),
                                ((J[:, 11, :] - J[:, 2, :]) @ fwd).max())),
        # 推：要求两手同时向前，所以取两手中较小的那个前伸
        "hands_forward": float(np.minimum((J[:, 20, :] - J[:, 17, :]) @ fwd,
                                          (J[:, 21, :] - J[:, 16, :]) @ fwd).max()),
        "wrist_speed": float((np.linalg.norm(np.diff(J[:, WRIST, :], axis=0), axis=1) * 20).max())
        if T > 1 else 0.0,
        "stance_width": float(np.mean(np.linalg.norm(J[:, 7, :] - J[:, 8, :], axis=1))),
    }


def check(prompt, sig, backend=None):
    """返回 (ok, 不合格项, 观察值)。没有规则命中时 ok 为 None。"""
    text = str(prompt).lower()
    detail, bad = {}, []
    for pattern, field, th, why, kind in RULES:
        if not re.search(pattern, text):
            continue
        v = sig.get(field)
        detail[field] = v
        if v is None:
            continue
        if kind == "min" and v < th:
            bad.append((field, th, why))
        elif kind == "max" and v > th:
            bad.append((field, th, why))
    if re.search(r"jump|hop", text):
        up = sig.get("pelvis_up") or 0.0
        detail["jump"] = up
        if up < 0.08:
            bad.append(("jump", 0.08, "应往上跳起"))
    if not detail:
        return None, [], {}
    return (len(bad) == 0), bad, detail


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    J = np.load(sys.argv[1])
    print("  " + "  ".join("%s=%.2f" % (k, v) for k, v in signature(J).items()))
