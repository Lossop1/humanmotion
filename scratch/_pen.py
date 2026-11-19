"""穿模近似检测：把骨骼看成胶囊，检查不该相交的骨头对是否插入。

为什么用胶囊
    三角面求交需要网格对分割与 BVH，代价高且依赖额外库；
    胶囊近似只需要关节位置与一组半径，几十毫秒能算完整条序列，
    对「前臂交叉、手插进躯干、两腿互穿」这类问题足够灵敏。

校准
    鼓掌：前臂间距 7.8 厘米，两只前臂半径约 4 厘米 → 应判穿模
    走路、举手：应判不穿模
"""
import numpy as np

# 骨名 → (父关节, 子关节, 半径 m)
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

# 需要检查的骨头对（同一肢体上的相邻骨头不查）
PAIRS = [
    ("l_fore", "r_fore"), ("l_upper", "r_upper"), ("l_fore", "r_upper"),
    ("r_fore", "l_upper"), ("l_fore", "torso"), ("r_fore", "torso"),
    ("l_upper", "torso"), ("r_upper", "torso"),
    ("l_fore", "head"), ("r_fore", "head"),
    ("l_thigh", "r_thigh"), ("l_shin", "r_shin"),
    ("l_fore", "l_thigh"), ("r_fore", "r_thigh"),
]


def _seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1) if e > 1e-12 else 0.0
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def penetration(J):
    """返回 (穿模帧占比, 最深插入量 米, 最常出问题的骨对)。"""
    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    worst = (0.0, "")
    frames_bad = 0
    for t in range(T):
        bad_t = False
        for a, b in PAIRS:
            p1, q1, r1 = CAPSULES[a]
            p2, q2, r2 = CAPSULES[b]
            d = _seg_dist(J[t, p1], J[t, q1], J[t, p2], J[t, q2]) - (r1 + r2)
            if d < 0:
                bad_t = True
                if d < worst[0]:
                    worst = (d, "%s×%s" % (a, b))
        frames_bad += 1 if bad_t else 0
    return frames_bad / max(T, 1), -worst[0], worst[1]


if __name__ == "__main__":
    import sys
    J = np.load(sys.argv[1])
    frac, depth, pair = penetration(J)
    print("  穿模帧 %.0f%%  最深 %.1f 厘米  %s" % (100 * frac, 100 * depth, pair))
