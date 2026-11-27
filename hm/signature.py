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
    # 词边界必须是 \b：否则 "sidesteps to the left" 会被 \bsteps?\s+ 命中，
    # 于是侧移被套上「应该走出去 ≥0.6 米」的走路判据，被判可疑（实测 verify_02/03 都栽在这）。
    (r"\b(walks?|walking|runs?|running|sprints?|jogs?|steps?)\b\s+"
     r"(forward|backward|to the left|to the right)", "disp_m", 0.6, "应走出去", "min"),
    (r"turns? around|turning around", "yaw_deg", 45.0, "应转向", "min"),
    # 阈值取实测分离点：momask 前倾中位 2.0、最大 7.6、后退 ≤-0.01；
    #                     mdm   前倾中位 10.2、后退中位 0.39
    (r"turns? around|turning around", "lean_median", 9.0, "转身应站直", "max"),
    (r"turns? around|turning around", "backward_m", 0.1, "转身不该边退边走", "max"),
    # 踏步：膝角要明显高于走路（走路中位 52 度），且不能走远。
    # 实测 "marches in place" 只有 56 度（等于走路）；"with high knees" 是 93–111 度。
    (r"marches?|marching", "knee_range", 80.0, "踏步应有明显抬腿（膝角 >80 度）", "min"),
    (r"marches? in place|marching in place", "disp_m", 0.5, "原地踏步不该走远", "max"),

    # 侧移：身体朝向应与行进方向约 90 度；"walks to the right" 实测只有 4–12 度（直走）
    (r"sidesteps? to the (left|right)", "facing_travel_deg", 60.0,
     "侧移应侧着走（朝向与行进约 90 度）", "min"),
    (r"sidesteps? to the (left|right)", "facing_travel_deg", 120.0,
     "侧移不该转过身走", "max"),
    # 举手：校准后不可分（正对照最小 0.065 vs 负对照最大 0.139，弯腰时手腕也会相对抬升），
    # 因此不设规则，量保留在签名里供排序参考。

    # 坐下：只认「坐下去」这类，不认「从坐着站起来」——后者结束时不该更低
    (r"sits? down|sitting down|sits? on|takes? a seat|is seated", "pelvis_end_drop", 0.18,
     "结束时重心应降低", "min"),
    (r"bends? forward|bending forward|bends? over|bowing", "lean_delta", 20.0, "躯干应前倾", "min"),
    (r"crouch|squat|kneel", "pelvis_drop", 0.18, "应降低重心", "min"),
    # 跳：原来写在 check() 的特例里，--explain 的「能判什么」看不到，挪进规则表
    (r"jumps?|jumping|hops?", "pelvis_up", 0.08, "应往上跳起", "min"),
    # 停住规则已否证：speed_tail 随片段长度漂移（见 INSEPARABLE 与目标树 G39）。
    # 44 帧时「走然后停住」末速 1.19–1.64；80 帧时 0.01–0.05；119 帧时连纯走也只有 0.02。
    # 踢腿：校准于正对照「kicks forward」最小 0.552、负对照（走/站/原地踏步）最大 0.492
    (r"kicks? forward|kicking forward|does a (front|side) kick", "kick_reach", 0.52,
     "脚应向前踢出", "min"),
    # 出拳：措辞点名到哪只手时可分离（负对照最大 0.22，正对照最小 0.52）；
    # 不点手的写法仍有 1/6 打空，所以阈值取 0.35，介于两者之间
    (r"punches?|punching|throws? a (right |left )?punch", "punch_reach", 0.35,
     "手应向前伸出", "min"),
]


# 已证明「校准量分不开」的动作类别，以及分不开的依据（见 G28/G29）
INSEPARABLE = [
    (r"(raises?|lifting|lifts?)\s+(both\s+|one\s+|a\s+|the\s+(right|left)\s+)?(hands?|arms?)",
     "举手类：弯腰时手腕同样抬升，量分不开"),
    (r"pushes?|pushes? forward|pulls?", "推拉类：正负对照重叠（-0.29 vs 0.22）"),
    (r"picks? up|reaching|reaches", "捡取类：无校准量，幅度与弯腰重叠"),
    (r"performs? a series|a set of movements", "复合描述：不是一个具体动作"),
    # 停住曾被当作可判（末段速度 ≤0.10）。实测否证：同一句「走然后停住」在 44 帧时
    # 末速 1.19–1.64，80 帧时 0.01–0.05，119 帧时连纯走也只有 0.02——
    # speed_tail 随片段长度漂移：短片段判不过，长片段什么都判得过。
    (r"stands? still|stops? (walking|moving)|com(e|es) to a stop|slow(s)? down and stops?",
     "停住：末段速度随片段长度漂移（44/80/119 帧差异极大），分不开"),
    # 起身：试过 pelvis_up / pelvis_drop / pelvis_end_drop / lean_delta / knee_range 五个量，
    # 正负全部重叠（例如 pelvis_up 正 0.003–0.075 vs 负 0.000–0.056）→ 分不开
    (r"stands? (up|upright)|straightens? (up|his|her)|gets? to (his|her) feet|rises? from",
     "起身：骨盆升高与弯腰/走路重叠（正 0.003–0.075 vs 负 0.000–0.056），分不开"),
    # 喝水：需要手到嘴的距离与头部朝向，SMPL 22 关节没有嘴部点，手-头距离与「摸头/擦汗」重叠
    (r"drinks?|cup", "喝水：没有嘴部点，手到头的距离与摸头/举手重叠，分不开"),
    # 看/转头：22 关节里没有脖子/头部朝向自由度（yaw 只有根），头跟着身体走，量不出「头朝哪」
    (r"looks? (to the|at|left|right|up|down)|turns? (his|her) head", 
     "看/转头：22 关节没有头部朝向自由度，头跟身体走，量不出看的方向"),
    # 落地：落地瞬间必然屈膝，与蹲/跳用同一批量（膝角、骨盆高度），正负重叠
    (r"lands?|landing", "落地：与蹲/跳同量（膝角、骨盆高度）重叠，分不开"),
]


def why_unjudgeable(prompt):
    """该动作为什么无法判：返回一句原因，空串表示不属于已知不可判类别。"""
    text = str(prompt).lower()
    for pattern, why in INSEPARABLE:
        if re.search(pattern, text):
            return why
    return ""


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
    # 身体朝向：髋线（左髋 1 → 右髋 2）的垂线；行进方向：根的水平位移
    _hip = J[0, 2] - J[0, 1]
    _body = np.array([_hip[2], 0.0, -_hip[0]])
    _disp = (J[-1, 0] - J[0, 0])[[0, 2]]
    if np.linalg.norm(_body) < 1e-6 or np.linalg.norm(_disp) < 1e-6:
        facing_travel = 0.0
    else:
        _b = math.degrees(math.atan2(_body[0], _body[2]))
        _t = math.degrees(math.atan2(_disp[0], _disp[1]))
        facing_travel = float(abs((_t - _b + 180.0) % 360.0 - 180.0))
    rw = J[:, WRIST, :] - J[:, SHOULDER, :]
    return {
        "disp_m": float(horizon.max()),
        "speed_mps": speed,
        # 峰值速度：区分「站着/停住」与「原地踏步/慢跑」用得上（中位位移会重叠）
        "speed_max": float(np.max(np.linalg.norm(np.diff(root, axis=0), axis=1)) * 20)
        if T > 1 else 0.0,
        # 末段速度：判断「停住了」要看结尾，不能看整段峰值（先走再停的峰值很高）
        "speed_tail": float(np.mean(np.linalg.norm(np.diff(root[-9:], axis=0), axis=1)) * 20)
        if T > 9 else 0.0,
        "wrist_above": wrist_above,
        "pelvis_drop": pelvis_drop,
        "pelvis_up": pelvis_up,
        "pelvis_end_drop": pelvis_end_drop,
        "yaw_deg": float(abs(yaws[-1] - yaws[0])),
        # 身体朝向与行进方向的夹角：侧移应约 90 度，直走约 0 度。
        # 实测 "walks to the right" 夹角只有 4–12 度（其实是直走），
        # "sidesteps to the right" 是 58–93 度——这个量能把两者分开。
        "facing_travel_deg": facing_travel,
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


DEAD_END_STEMS = {
    "crawl": ("crawl",), "crouch": ("crouch",), "lie": ("lie", "lying", "lies down"),
    "dance": ("danc",), "kneel": ("kneel",), "get_up": ("get up", "gets up", "stood up"),
}


def dead_end_in(prompt):
    """提示词里是否含实测做不出来的动作（G2 的死路）。返回中文动作名或空串。

    必要性：像「躺下再走」这类句子，规划会给出一整段含 lie 的提示词，
    签名只查得出「走了」，查不出「没躺下」——不拦就会报「通过」。
    """
    text = str(prompt).lower()
    for key, stems in DEAD_END_STEMS.items():
        if any(st in text for st in stems):
            return key
    return ""


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
    if not detail:
        return None, [], {}
    return (len(bad) == 0), bad, detail


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    J = np.load(sys.argv[1])
    print("  " + "  ".join("%s=%.2f" % (k, v) for k, v in signature(J).items()))
