"""过渡实验台：手动拨参数，出中间帧，渲骨架，看效果。

只用两段动作的关节坐标。不涉及 LLM、不涉及生成模型、不涉及数据集。

输入
    A 段全部帧的 22 关节坐标
    B 段全部帧的 22 关节坐标

能拨的旋钮
    --n N            过渡帧数
    --time 模式      每根骨头的时间参数：linear / shaped / stagger
    --arc 度         路径绕行：0 表示走最短弧，正数表示中途往外鼓出这么多度
    --root 模式      blend 按两端速度混合；hold 保持 A 末速度

输出
    runs/<run>/trial_<tag>/joints.npy   （A 尾 20 + 过渡 N + B 头 20）
    runs/<run>/trial_<tag>/out.mp4      （骨架渲染，同一相机，便于比较）
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from hm import config, to_joints

# 21 根骨头：(子关节, 父关节)
BONES = [(1, 0), (4, 1), (7, 4), (10, 7),
         (2, 0), (5, 2), (8, 5), (11, 8),
         (3, 0), (6, 3), (9, 6), (12, 9),
         (15, 12), (13, 9), (16, 13), (18, 16), (20, 18),
         (14, 9), (17, 14), (19, 17), (21, 19)]


def dirs_len(J):
    d = np.zeros((len(BONES), 3))
    L = np.zeros(len(BONES))
    for i, (c, p) in enumerate(BONES):
        v = J[c] - J[p]
        L[i] = max(np.linalg.norm(v), 1e-9)
        d[i] = v / L[i]
    return d, L


def rot_between(a, b):
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    ang = math.acos(dot)
    if s < 1e-9:
        t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, t)
        axis /= max(np.linalg.norm(axis), 1e-9)
        return axis, (0.0 if dot > 0 else math.pi)
    return axis / s, ang


def rotate_about(v, axis, ang):
    """罗德里格斯旋转。"""
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1 - c)


def bone_rates(J, w=3, tail=True):
    """每根骨头在段末或段首的平均角速度（度/帧）。"""
    wr = J[-1 - w:] if tail else J[:1 + w]
    out = np.zeros(len(BONES))
    for k in range(len(BONES)):
        acc = []
        for i in range(len(wr) - 1):
            da, _ = dirs_len(wr[i])
            db, _ = dirs_len(wr[i + 1])
            cs = float(np.clip(np.dot(da[k], db[k]), -1, 1))
            acc.append(math.degrees(math.acos(cs)))
        out[k] = float(np.mean(acc)) if acc else 0.0
    return out


def time_params(n, mode, amp, r_in, r_out):
    t = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    if mode == "linear":
        return np.repeat(t[:, None], len(BONES), axis=1)
    if mode == "stagger":
        w = np.clip(0.5 + 0.5 * (amp / max(amp.max(), 1e-9)), 0.35, 1.0)
        st = 1.0 - w
        return np.clip((t[:, None] - st[None, :]) / np.maximum(1e-6, 1 - st[None, :]), 0, 1)
    if mode == "shaped":
        out = np.zeros((n, len(BONES)))
        for k in range(len(BONES)):
            if amp[k] < 1e-9:
                out[:, k] = t
                continue
            m0 = min(max(r_in[k] * (n + 1) / amp[k], 0.0), 3.0)
            m1 = min(max(r_out[k] * (n + 1) / amp[k], 0.0), 3.0)
            t2, t3 = t * t, t * t * t
            out[:, k] = np.clip((t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) + (t3 - t2) * m1, 0, 1)
        return out
    raise ValueError(mode)


def make(JA, JB, n, time_mode="shaped", arc_deg=0.0, root_mode="blend", foot="off", limb="rot"):
    """构造过渡帧。返回 (n, 22, 3)。"""
    A, B = JA[-1], JB[0]
    dA, LA = dirs_len(A)
    dB, LB = dirs_len(B)
    L = 0.5 * (LA + LB)
    amp = np.array([rot_between(dA[k], dB[k])[1] for k in range(len(BONES))])
    r_in = bone_rates(JA, tail=True)
    r_out = bone_rates(JB, tail=False)
    # amp 是弧度，r_in/r_out 是度/帧。必须统一单位，否则比值被放大 57 倍，
    # 所有斜率都被夹到上限 3.0，shaped 退化成「两端都取最大斜率」。
    us = time_params(n, time_mode, np.degrees(amp), r_in, r_out)

    # 根：位置与速度
    if root_mode == "hold":
        v_in = (JA[-1, 0] - JA[-4, 0]) / 3.0 if len(JA) >= 4 else np.zeros(3)
        vel = np.repeat(v_in[None, :], n, axis=0)
    else:
        w = 3
        v_in = (JA[-1, 0] - JA[-1 - w, 0]) / w if len(JA) > w else np.zeros(3)
        v_out = (JB[w, 0] - JB[0, 0]) / w if len(JB) > w else v_in
        u = np.arange(1, n + 1) / (n + 1.0)
        s_in = float(np.hypot(v_in[0], v_in[2]))
        s_out = float(np.hypot(v_out[0], v_out[2]))
        h_in = math.atan2(v_in[0], v_in[2]) if s_in > 1e-9 else 0.0
        h_out = math.atan2(v_out[0], v_out[2]) if s_out > 1e-9 else h_in
        dh = h_out - h_in
        while dh > math.pi:
            dh -= 2 * math.pi
        while dh < -math.pi:
            dh += 2 * math.pi
        vel = np.zeros((n, 3))
        for i, uu in enumerate(u):
            sp = s_in * (1 - uu) + s_out * uu
            h = h_in + dh * uu
            vel[i] = [sp * math.sin(h), v_in[1] * (1 - uu) + v_out[1] * uu, sp * math.cos(h)]

    frames = np.zeros((n, 22, 3))
    frames[:, 0] = A[0][None, :] + np.cumsum(vel, axis=0)
    for i in range(n):
        for k, (c, p) in enumerate(BONES):
            ax, ang = rot_between(dA[k], dB[k])
            uu = float(us[i, k])
            d = rotate_about(dA[k], ax, ang * uu) if ang > 1e-9 else dA[k].copy()
            if arc_deg > 0.0 and ang > 0.02:
                # 绕行：在过渡中点把骨头朝「背离躯干中轴」的方向额外转一点
                w_out = frames[i, p] - frames[i, 0]
                w_out[1] = 0.0
                if np.linalg.norm(w_out) > 1e-6:
                    side = np.cross(d, w_out / np.linalg.norm(w_out))
                    if np.linalg.norm(side) > 1e-6:
                        d = rotate_about(d, side, math.radians(arc_deg) * math.sin(math.pi * uu))
            frames[i, c] = frames[i, p] + L[k] * d
    if limb == "ik":
        frames = apply_limb_ik(frames, JA, JB, arc_deg=arc_deg)
    if foot in ("L", "R"):
        frames = plant_foot(frames, JA, JB, foot)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--junction", type=int, default=1, help="第几个接缝，从 1 起")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--time", default="shaped", choices=["linear", "stagger", "shaped"])
    ap.add_argument("--arc", type=float, default=0.0, dest="arc_deg")
    ap.add_argument("--root", default="blend", choices=["blend", "hold"])
    ap.add_argument("--limb", default="rot", choices=["rot", "ik"])
    ap.add_argument("--ctx", type=int, default=20, help="两侧各带多少帧上下文")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    if not run.is_absolute():
        run = config.OUT_ROOT / run.name
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))["plan"]
    segs = plan["segments"]
    J = [to_joints.from_feats(np.load(run / ("feats_%d.npy" % s["id"]))) for s in segs]

    k = a.junction - 1
    JA, JB = J[k], J[k + 1]
    fr = make(JA, JB, a.n, a.time, a.arc_deg, a.root, limb=a.limb)

    if a.tag:
        tag = a.tag
    else:
        # 自动编号：V01-j1-n12-shaped-arc0-blend
        used = []
        for d in run.glob("V*"):
            try:
                used.append(int(d.name[1:3]))
            except ValueError:
                pass
        nxt = (max(used) + 1) if used else 1
        tag = "V%02d-j%d-n%d-%s-arc%d-%s-%s" % (
            nxt, a.junction, a.n, a.time, int(a.arc_deg), a.root, a.limb)
    out = run / tag
    out.mkdir(parents=True, exist_ok=True)
    ctx = min(a.ctx, len(JA), len(JB))
    seq = np.concatenate([JA[-ctx:], fr, JB[:ctx]], axis=0)
    np.save(out / "joints.npy", seq)

    # 量三个硬指标
    def bone_gap(f0, f1):
        d0, _ = dirs_len(f0)
        d1, _ = dirs_len(f1)
        return max(math.degrees(math.acos(float(np.clip(np.dot(d0[i], d1[i]), -1, 1))))
                   for i in range(len(BONES)))
    dev = 0.0
    for f in fr:
        _, Lf = dirs_len(f)
        dev = max(dev, float(np.max(np.abs(Lf - (0.5 * (dirs_len(JA[-1])[1] + dirs_len(JB[0])[1]))) /
                                    np.maximum(dirs_len(JA[-1])[1], 1e-9))))
    print("  接缝%d  段%d→段%d   过渡 %d 帧   time=%s  arc=%g  root=%s" % (
        a.junction, k + 1, k + 2, a.n, a.time, a.arc_deg, a.root))
    print("    骨长最大偏差 %.2e%%    首步 %.1f 度（邻段 %.1f）   末步 %.1f 度（邻段 %.1f）" % (
        100 * dev,
        bone_gap(JA[-1], fr[0]), bone_rates(JA, tail=True).max(),
        bone_gap(fr[-1], JB[0]), bone_rates(JB, tail=False).max()))
    print("    上下文 %d + %d + %d = %d 帧" % (ctx, a.n, ctx, len(seq)))

    cmd = [str(config.BLENDER), "--background", "--python",
           str(Path(__file__).with_name("render_skeleton.py")), "--",
           str(out), str(out / "render")]
    subprocess.run(cmd, check=False)
    print("  视频 %s" % (out / "render" / "out.mp4"))



# ---------- 脚部接触与单腿逆运动学 ----------
# 左腿骨头下标 1,2,3（髋→膝→踝→脚），关节 1,4,7,10
# 右腿骨头下标 5,6,7，关节 2,5,8,11
LEG = {"L": {"bones": (1, 2, 3), "hip": 1, "knee": 4, "ankle": 7, "toe": 10},
       "R": {"bones": (5, 6, 7), "hip": 2, "knee": 5, "ankle": 8, "toe": 11}}


def contact_score(J, side, w=3):
    """这只脚在段末是否贴地：返回 (高度, 水平速度)。"""
    toe = LEG[side]["toe"]
    h = float(J[-1, toe, 1])
    v = float(np.linalg.norm((J[-1, toe, [0, 2]] - J[-1 - w, toe, [0, 2]]) / w))
    return h, v


def ik_leg(hip, target, knee_hint, Lt, Ls):
    """两段腿的逆运动学。返回 (knee, ankle)。

    给定髋位置、期望的踝位置、一个用来定膝盖朝哪的参考点，
    用余弦定理定膝角，膝盖落在这个平面内。
    够不到时把踝缩到可达距离。
    """
    hip = np.asarray(hip, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    d = target - hip
    dist = float(np.linalg.norm(d))
    lo = abs(Lt - Ls) + 1e-4
    hi = (Lt + Ls) * 0.9995
    dist = float(np.clip(dist, lo, hi))
    u = d / max(float(np.linalg.norm(d)), 1e-9)
    ankle = hip + u * dist
    # 髋处的夹角
    cos_a = (Lt * Lt + dist * dist - Ls * Ls) / (2 * Lt * dist)
    a = math.acos(float(np.clip(cos_a, -1.0, 1.0)))
    # 膝盖朝向：把参考方向投影到垂直于 u 的平面
    w = np.asarray(knee_hint, dtype=np.float64) - hip
    w = w - u * float(np.dot(w, u))
    if np.linalg.norm(w) < 1e-6:
        w = np.cross(u, np.array([1.0, 0.0, 0.0]))
    w = w / max(float(np.linalg.norm(w)), 1e-9)
    knee = hip + Lt * (math.cos(a) * u + math.sin(a) * w)
    return knee, ankle


def plant_foot(frames, JA, JB, side):
    """过渡期间把一只脚钉在 A 段末帧的落地位置，用逆运动学反解那条腿。"""
    L = LEG[side]
    b_thigh, b_shin, b_foot = L["bones"]
    hip_j, knee_j, ankle_j, toe_j = L["hip"], L["knee"], L["ankle"], L["toe"]

    dA, LA = dirs_len(JA[-1])
    Lt, Ls, Lf = float(LA[b_thigh]), float(LA[b_shin]), float(LA[b_foot])

    # 钉住的踝位置与脚掌朝向
    ankle_A = JA[-1, ankle_j].copy()
    ankle_B = JB[0, ankle_j].copy()
    foot_dir_A = dA[b_foot].copy()
    dB, _ = dirs_len(JB[0])
    foot_dir_B = dB[b_foot].copy()

    n = len(frames)
    for i in range(n):
        u = (i + 1) / (n + 1.0)
        hip = frames[i, hip_j].copy()
        # 踝目标：从 A 的实际踝位置到 B 的实际踝位置线性过渡，
        # 但起点必须精确等于落地位置，所以前几帧几乎不动
        tgt = ankle_A * (1 - u) + ankle_B * u
        knee_hint = frames[i, knee_j]
        knee, ankle = ik_leg(hip, tgt, knee_hint, Lt, Ls)
        frames[i, knee_j] = knee
        frames[i, ankle_j] = ankle
        fd = foot_dir_A * (1 - u) + foot_dir_B * u
        fd = fd / max(float(np.linalg.norm(fd)), 1e-9)
        frames[i, toe_j] = ankle + Lf * fd
    return frames

# ---------- 四肢：末端走笛卡尔轨迹，肘膝用逆运动学解 ----------
# 左臂  9 颈 →13 肩 →16 肘 →18 腕 →20 手
# 右臂  9 颈 →14 肩 →17 肘 →19 腕 →21 手
# 左腿  0 骨盆 →1 髋 →4 膝 →7 踝 →10 脚
# 右腿  0 骨盆 →2 髋 →5 膝 →8 踝 →11 脚
ARMS = {"L": dict(chain=(9, 13, 16, 18, 20), up=14, lo=15, hand=16),
        "R": dict(chain=(9, 14, 17, 19, 21), up=18, lo=19, hand=20)}


def ik_chain(root_p, target, hint, l1, l2):
    """两段链的逆运动学：给定根位置和目标位置，解出中间关节。

    中间关节落在一个圆上（两球交线），用 hint 指定朝哪个方向。
    够不到时把目标缩到可达距离。
    """
    root_p = np.asarray(root_p, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    d = target - root_p
    dist = float(np.linalg.norm(d))
    dist = float(np.clip(dist, abs(l1 - l2) + 1e-4, (l1 + l2) * 0.9995))
    u = d / max(float(np.linalg.norm(d)), 1e-9)
    end = root_p + u * dist
    cos_a = (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)
    a = math.acos(float(np.clip(cos_a, -1.0, 1.0)))
    w = np.asarray(hint, dtype=np.float64) - root_p
    w = w - u * float(np.dot(w, u))
    if np.linalg.norm(w) < 1e-6:
        w = np.cross(u, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(w) < 1e-6:
            w = np.cross(u, np.array([1.0, 0.0, 0.0]))
    w = w / max(float(np.linalg.norm(w)), 1e-9)
    mid = root_p + l1 * (math.cos(a) * u + math.sin(a) * w)
    return mid, end


def apply_limb_ik(frames, JA, JB, arm=True, leg=True, arc_deg=0.0):
    """把手臂和腿的末端改成笛卡尔直线（或拱起）轨迹，肘膝用 IK 解。"""
    dA, LA = dirs_len(JA[-1])
    dB, LB = dirs_len(JB[0])
    n = len(frames)
    L = 0.5 * (LA + LB)

    def cart(i, jn):
        u = (i + 1) / (n + 1.0)
        p0 = JA[-1, jn].copy()
        p1 = JB[0, jn].copy()
        p = p0 * (1 - u) + p1 * u
        if arc_deg > 0:
            # 拱起：在中点抬高，两端为 0
            p = p.copy()
            p[1] += math.tan(math.radians(arc_deg)) * 0.25 * math.sin(math.pi * u) * np.linalg.norm(p1 - p0)
        return p

    for i in range(n):
        if leg:
            for side, hip, knee, ankle, toe in (("L", 1, 4, 7, 10), ("R", 2, 5, 8, 11)):
                l1 = float(L[1] if side == "L" else L[5])
                l2 = float(L[2] if side == "L" else L[6])
                lf = float(L[3] if side == "L" else L[7])
                tgt = cart(i, ankle)
                # 中间关节的朝向参考必须从 A 的关节位置过渡到 B 的，
                # 否则首帧的 IK 解与 A 的真实姿态不一致，一上来就跳。
                u_ = (i + 1) / (n + 1.0)
                hint = JA[-1, knee] * (1 - u_) + JB[0, knee] * u_
                mid, end = ik_chain(frames[i, hip], tgt, hint, l1, l2)
                frames[i, knee] = mid
                frames[i, ankle] = end
                fd = dB[3 if side == "L" else 7]
                frames[i, toe] = end + lf * fd
        if arm:
            for side in ("L", "R"):
                cfg = ARMS[side]
                up, lo, hand = cfg["up"], cfg["lo"], cfg["hand"]
                l1, l2, lh = float(L[up]), float(L[lo]), float(L[hand])
                sh, el, wr, hd = cfg["chain"][1], cfg["chain"][2], cfg["chain"][3], cfg["chain"][4]
                tgt = cart(i, wr)
                u_ = (i + 1) / (n + 1.0)
                hint = JA[-1, el] * (1 - u_) + JB[0, el] * u_
                mid, end = ik_chain(frames[i, sh], tgt, hint, l1, l2)
                frames[i, el] = mid
                frames[i, wr] = end
                fd = dB[hand]
                frames[i, hd] = end + lh * fd
    return frames


if __name__ == "__main__":
    main()
