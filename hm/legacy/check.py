"""一次把所有该查的都量出来，不只看速度。

对每一处衔接、每一种构造，量：
    骨长        相对首帧的最大偏差
    膝肘弯曲    过渡中的范围，以及两端点自己的值（超出端点范围多少）
    脚部滑动    贴地帧里有多少在滑动、最大滑动速度
    穿地        最低脚点低于地面多少
    骨间距离    任意两根骨头之间的最小距离（穿模的代理指标）
    端点衔接    首步/末步与相邻动作段自己的角速度对比
    路径        与测地线的最大偏离（0 表示只是改了快慢，不是改了轨迹）
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hm import config, to_joints, trial as TR  # noqa: E402

LEG = {"L": (1, 4, 7, 10), "R": (2, 5, 8, 11)}
ARM = {"L": (16, 18, 20), "R": (17, 19, 21)}


def bend(a, b, c):
    u = np.asarray(b) - np.asarray(a)
    v = np.asarray(c) - np.asarray(b)
    n = float(np.linalg.norm(u) * np.linalg.norm(v))
    if n < 1e-12:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / n, -1, 1))))


def measure(fr, JA, JB):
    out = {}
    L0 = 0.5 * (TR.dirs_len(JA[-1])[1] + TR.dirs_len(JB[0])[1])
    dev = 0.0
    for f in fr:
        Lf = TR.dirs_len(f)[1]
        dev = max(dev, float(np.max(np.abs(Lf - L0) / np.maximum(L0, 1e-9))))
    out["骨长最大偏差"] = 100.0 * dev

    lines = []
    for side, (hip, knee, ankle, toe) in LEG.items():
        mid = [bend(f[hip], f[knee], f[ankle]) for f in fr]
        e0 = bend(JA[-1, hip], JA[-1, knee], JA[-1, ankle])
        e1 = bend(JB[0, hip], JB[0, knee], JB[0, ankle])
        lines.append((side, min(mid), max(mid), e0, e1))
    out["膝"] = lines

    alines = []
    for side, (sh, el, wr) in ARM.items():
        mid = [bend(f[sh], f[el], f[wr]) for f in fr]
        e0 = bend(JA[-1, sh], JA[-1, el], JA[-1, wr])
        e1 = bend(JB[0, sh], JB[0, el], JB[0, wr])
        alines.append((side, min(mid), max(mid), e0, e1))
    out["肘"] = alines

    slide = {}
    for side, (_, _, _, toe) in LEG.items():
        h = np.array([f[toe, 1] for f in fr])
        v = np.array([np.linalg.norm(fr[i + 1][toe, [0, 2]] - fr[i][toe, [0, 2]])
                      for i in range(len(fr) - 1)])
        low = h[:-1] < 0.06
        moving = v > 0.01
        slide[side] = (int(low.sum()), int((low & moving).sum()),
                       float(v[low].max()) if low.any() else 0.0)
    out["滑动"] = slide
    out["最低脚点"] = min(float(f[toe, 1]) for f in fr for toe in (10, 11))

    # 骨间最小距离：任意两根非相邻骨头
    pairs = [(i, j) for i in range(len(TR.BONES)) for j in range(i + 1, len(TR.BONES))
             if TR.BONES[i][0] not in TR.BONES[j] and TR.BONES[j][0] not in TR.BONES[i]
             and TR.BONES[i][1] not in TR.BONES[j] and TR.BONES[j][1] not in TR.BONES[i]]
    dmin = 1e9
    which = (0, 0, 0)
    for fi, f in enumerate(fr):
        seg = [(f[TR.BONES[k][0]], f[TR.BONES[k][1]]) for k in range(len(TR.BONES))]
        for i, j in pairs:
            d = seg_dist(seg[i], seg[j])
            if d < dmin:
                dmin = d
                which = (i, j, fi)
    out["骨间最小距离"] = dmin
    out["最近骨对"] = (TR.BONES[which[0]], TR.BONES[which[1]], which[2] + 1)
    deep = min(range(len(fr)), key=lambda t: min(fr[t][10, 1], fr[t][11, 1]))
    out["穿地帧"] = (deep + 1, min(fr[deep][10, 1], fr[deep][11, 1]))
    return out


def seg_dist(s1, s2):
    a, b = s1
    c, d = s2
    u = b - a
    v = d - c
    w = a - c
    A = float(np.dot(u, u))
    B = float(np.dot(u, v))
    C = float(np.dot(v, v))
    D = float(np.dot(u, w))
    E = float(np.dot(v, w))
    den = A * C - B * B
    if den < 1e-12:
        return min(np.linalg.norm(a - c), np.linalg.norm(a - d),
                   np.linalg.norm(b - c), np.linalg.norm(b - d))
    s = (B * E - C * D) / den
    t = (A * E - B * D) / den
    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(a + s * u - (c + t * v)))


def axis_dev(fr, JA, JB):
    d0 = TR.dirs_len(JA[-1])[0]
    d1 = TR.dirs_len(JB[0])[0]
    worst = 0.0
    for f in fr:
        df = TR.dirs_len(f)[0]
        for k in range(len(TR.BONES)):
            a0 = TR.rot_between(d0[k], d1[k])
            af = TR.rot_between(d0[k], df[k])
            if a0[1] < 0.05 or af[1] < 0.05:
                continue
            worst = max(worst, np.degrees(np.arccos(
                float(np.clip(np.dot(a0[0], af[0]), -1, 1)))))
    return worst


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "R01-wave-turn-walk"
    nfr = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    run = config.OUT_ROOT / name
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))["plan"]
    J = [to_joints.from_feats(np.load(run / ("feats_%d.npy" % s["id"]))) for s in plan["segments"]]
    names = [s["label"] for s in plan["segments"]]

    for k in range(len(J) - 1):
        JA = J[k]
        JB = TR.align_pair(J[k], J[k + 1])
        r_tail = float(TR.bone_rates(JA, tail=True).max())
        r_head = float(TR.bone_rates(JB, tail=False).max())
        print("  === %s 结束 → %s 开始（%d 帧）===" % (names[k], names[k + 1], nfr))
        for limb in ("rot", "ik"):
            fr = TR.make(JA, JB, nfr, "shaped", 0.0, "blend", limb=limb)
            m = measure(fr, JA, JB)
            g0 = max(math.degrees(TR.rot_between(
                TR.dirs_len(JA[-1])[0][i], TR.dirs_len(fr[0])[0][i])[1]) for i in range(len(TR.BONES)))
            gN = max(math.degrees(TR.rot_between(
                TR.dirs_len(fr[-1])[0][i], TR.dirs_len(JB[0])[0][i])[1]) for i in range(len(TR.BONES)))
            print("    [%s] 骨长 %.1e%%   首步 %.1f 末步 %.1f（邻段末 %.1f 首 %.1f）" % (
                limb, m["骨长最大偏差"], g0, gN, r_tail, r_head))
            for side, a, b, e0, e1 in m["膝"]:
                print("         膝%s 过渡 %5.1f~%5.1f   端点 %5.1f / %5.1f" % (side, a, b, e0, e1))
            for side, a, b, e0, e1 in m["肘"]:
                print("         肘%s 过渡 %5.1f~%5.1f   端点 %5.1f / %5.1f" % (side, a, b, e0, e1))
            for side, (low, sl, vmax) in m["滑动"].items():
                print("         脚%s 贴地 %2d 帧  滑动 %2d 帧  最大 %.3f 米/帧" % (side, low, sl, vmax))
            pr = m["最近骨对"]
            print("         最低脚点 %.3f 米（第 %d 帧）   骨间最小 %.3f 米（骨%s 与 骨%s，第 %d 帧）" % (
                m["穿地帧"][1], m["穿地帧"][0], m["骨间最小距离"], pr[0], pr[1], pr[2]))
            print("         路径偏离测地线 %.1f 度" % axis_dev(fr, JA, JB))
        print()


if __name__ == "__main__":
    main()
