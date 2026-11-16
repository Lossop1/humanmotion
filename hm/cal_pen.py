"""穿模指标的校准：先在已知答案的对照上验证，再决定能不能用。

规则（写进目标树）
    任何新指标，必须先在校准集上跑：
      负对照（不该报警）必须全判空
      正对照（已知有问题）必须判中
    做不到分离的指标，判【否证】，不用；不能「先用着看看」。

校准集
    负对照：站立、走路、举手（这三条肉眼无穿模）
    正对照：鼓掌（已知前臂在胸前交叉、网格互相插入）

可调参数
    各骨半径、端点收缩比例（避免胶囊在关节处自然重叠）
"""
import itertools

import numpy as np

BONES = {
    "l_upper": (16, 18), "l_fore": (18, 20),
    "r_upper": (17, 19), "r_fore": (19, 21),
    "l_thigh": (1, 4), "l_shin": (4, 7),
    "r_thigh": (2, 5), "r_shin": (5, 8),
    "torso": (0, 9), "head": (12, 15),
}

PAIRS = [
    ("l_fore", "r_fore"), ("l_upper", "r_upper"), ("l_fore", "r_upper"),
    ("r_fore", "l_upper"), ("l_fore", "torso"), ("r_fore", "torso"),
    ("l_upper", "torso"), ("r_upper", "torso"),
    ("l_fore", "head"), ("r_fore", "head"),
    ("l_thigh", "r_thigh"), ("l_shin", "r_shin"),
]


def _seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1) if e > 1e-12 else 0.0
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def penetration(J, radii, trim=0.15):
    """返回 (穿模帧占比, 最深插入米, 骨对)。trim 为端点收缩比例。"""
    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    frames_bad = 0
    worst = (0.0, "")
    for t in range(T):
        bad = False
        for a, b in PAIRS:
            ia, ja = BONES[a]
            ib, jb = BONES[b]
            pa, qa = J[t, ia], J[t, ja]
            pb, qb = J[t, ib], J[t, jb]
            pa, qa = pa + (qa - pa) * trim, qa - (qa - pa) * trim
            pb, qb = pb + (qb - pb) * trim, qb - (qb - pb) * trim
            d = _seg_dist(pa, qa, pb, qb) - (radii[a] + radii[b])
            if d < 0:
                bad = True
                if d < worst[0]:
                    worst = (d, "%s×%s" % (a, b))
        frames_bad += 1 if bad else 0
    return frames_bad / max(T, 1), -worst[0], worst[1]


def sweep(controls, positive):
    print("  半径设定              负对照(应全 0)                正对照(鼓掌，应 >0)")
    base = {"l_upper": 0.04, "l_fore": 0.035, "r_upper": 0.04, "r_fore": 0.035,
            "l_thigh": 0.07, "l_shin": 0.05, "r_thigh": 0.07, "r_shin": 0.05,
            "torso": 0.09, "head": 0.09}
    best = None
    for torso, thigh, trim in itertools.product((0.07, 0.09, 0.11),
                                                (0.05, 0.07, 0.09),
                                                (0.10, 0.20, 0.30)):
        r = dict(base, torso=torso, l_thigh=thigh, r_thigh=thigh)
        ctrl = [penetration(J, r, trim)[0] for J in controls]
        pos = penetration(positive, r, trim)
        ctl_max = max(ctrl)
        sep = (ctl_max == 0.0) and (pos[0] > 0.2)
        line = "  torso %.2f thigh %.2f trim %.2f → 对照 max %.0f%%  鼓掌 %.0f%%  %s" % (
            torso, thigh, trim, 100 * ctl_max, 100 * pos[0], "分离 ✓" if sep else "")
        print(line)
        if sep and (best is None or pos[0] > best[0]):
            best = (pos[0], r, trim)
    return best
