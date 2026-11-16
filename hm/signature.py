"""动作签名：用关节轨迹本身的粗指标，判断「这个动作到底做没做出来」。

为什么需要它
    几何判据（骨长、朝向、间距）只保证「像个人」，不保证「做了要求的动作」。
    44 段人工核查里几何全过、人判为错的占 20.5%。
    人要判「做没做」，看的是最粗的几件事：人有没有走出去、手有没有抬起来、
    有没有转身、有没有蹲下去。这些都能从关节轨迹直接量出来，不需要模型。

签名（每段一组数）
    disp_m        根节点的水平位移（米）
    speed_mps     根节点的平均水平速度
    wrist_above   手腕最高点相对肩高（米），正值表示手举过肩
    pelvis_drop   骨盆最低点相对起始高度（米，向下为正）
    yaw_deg       骨盆朝向的总变化（度）
    knee_range    膝关节角的活动范围（度）

规则（动作关键词 → 期望）
    walk/run/sprint → 应走出去          disp_m ≥ 0.6
    turn/转身       → 应转向            yaw_deg ≥ 45
    jump/hop        → 应上下起落        pelvis_drop 或上升 ≥ 0.08
    raise/wave/挥手/举手 → 手应过肩      wrist_above ≥ 0.05
    sit/坐下        → 应降低             pelvis_drop ≥ 0.20
"""
import math
import re
import sys

import numpy as np

from hm.job import BONES

SHOULDER = 16          # 右肩关节序号（HumanML3D 22 关节）
WRIST = 21             # 右腕
PELVIS = 0


def _yaw(J):
    """骨盆朝向：左右髋连线在水平面的角度。"""
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
    a = J[hip] - J[knee]
    b = J[ankle] - J[knee]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
    return math.degrees(math.acos(c))


def signature(J):
    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    root = J[:, PELVIS, :]
    horizon = np.linalg.norm(root[:, [0, 2]] - root[0, [0, 2]], axis=1)
    speed = float(np.mean(np.linalg.norm(np.diff(root, axis=0), axis=1)) * 20) if T > 1 else 0.0
    wrist = J[:, WRIST, 1]
    shoulder = J[:, SHOULDER, 1]
    wrist_above = float(wrist.max() - np.median(shoulder))
    pelvis_drop = float(J[0, PELVIS, 1] - J[:, PELVIS, 1].min())
    pelvis_up = float(J[:, PELVIS, 1].max() - J[0, PELVIS, 1])
    pelvis_end_drop = float(J[0, PELVIS, 1] - J[-1, PELVIS, 1])
    yaws = _unwrap([_yaw(f) for f in J])
    # 躯干前倾：骨盆→脊柱3 与竖直方向的夹角
    tv = J[:, 9] - J[:, 0]
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    lean = np.degrees(np.arccos(np.clip(tv[:, 1], -1, 1)))
    # 沿「朝向反方向」的净位移，正数表示在后退
    fwd = J[0, 10] - J[0, 7]
    fwd[1] = 0.0
    nf = float(np.linalg.norm(fwd))
    fwd = fwd / nf if nf > 1e-9 else np.array([0.0, 0.0, 1.0])
    back = float(-(J[-1, 0] - J[0, 0]) @ fwd)
    knee = [_knee_angle(f, 1, 4, 7) for f in J]
    return {
        "disp_m": float(horizon.max()),
        "speed_mps": speed,
        "wrist_above": wrist_above,
        "pelvis_drop": pelvis_drop,
        "pelvis_up": pelvis_up,
        "pelvis_end_drop": pelvis_end_drop,
        "yaw_deg": float(abs(yaws[-1] - yaws[0])),
        "knee_range": float(max(knee) - min(knee)),
        "lean_median": float(np.median(lean)),
        "lean_max": float(lean.max()),
        "backward_m": back,
    }


# 只认「动作词」，不认方向词。
# 之前把 forward 也写进位移规则，导致「bends over forward」被要求走出去 0.6 米，
# 这是把方向当成动作了。
RULES = [
    ("walk|walking|run|running|sprint|sprinting|jog|march|step forward| stroll",
     "disp_m", 0.6, "应走出去"),
    ("turn|turning|rotate|spin|pivot", "yaw_deg", 45.0, "应转向"),
    ("raise|raising|wave|waving|stretch|stretching|lift|lifting",
     "wrist_above", 0.05, "手应举过肩"),
    ("sit|sits|sitting|seated", "pelvis_end_drop", 0.18, "结束时重心应降低"),
    ("crouch|squat|bend|bending|kneel|kneeling", "pelvis_drop", 0.18, "应降低重心"),
    # 校准对：MDM 转身前倾中位 8.9 度、后退 0.97 米；MoMask 3.6 度、后退 -0.05 米
    ("turns around|turn around", "lean_median", 6.0, "转身应站直，不该弓着"),
    ("turns around|turn around", "backward_m", 0.3, "转身不该边退边走"),
]


def check(prompt, sig):
    """返回 (ok, 命中的规则, 明细)。没有任何规则命中时视为无法判断。"""
    text = str(prompt).lower()
    hits = []
    for pattern, field, th, why in RULES:
        if re.search(pattern, text):
            hits.append((field, th, why, sig.get(field)))
    detail = {f: v for f, _, _, v in hits}
    bad = [(f, th, w) for f, th, w, v in hits if (v or 0) < th]
    if re.search(r"jump|hop", text):
        up = sig.get("pelvis_up") or 0
        detail["jump"] = up
        if up < 0.08:
            bad.append(("jump", 0.08, "应往上跳起"))
    if not detail:
        return None, [], {}
    return (len(bad) == 0), bad, detail


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    J = np.load(sys.argv[1])
    s = signature(J)
    print("  " + "  ".join("%s=%.2f" % (k, v) for k, v in s.items()))
