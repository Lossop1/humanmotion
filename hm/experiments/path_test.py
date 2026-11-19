"""过渡路径实验台。只吃两段关节坐标，输出中间帧的关节坐标。

不接生成、不接拟合、不接渲染。目的是把「中间路径怎么走」单独拿出来比较。

硬指标
   骨长偏差   每根骨头在中间帧的长度与端点相比的最大相对偏差。必须接近 0。
   最近距离   不相邻的骨段之间的最小距离。越小说明肢体越接近交叉。
   端点角速度 过渡第一步/最后一步的角度变化，与两端实测角速度的差。

候选方案
   方向均匀    每根骨头沿最短弧匀速转
   方向塑形    用三次曲线按两端角速度塑形
   方向错开    每根骨头按需转角大小错开启动时刻
   位置线性    直接对关节位置插值（作为对照，会破坏骨长）
"""
import math

import numpy as np

# 21 根骨头：(子关节, 父关节)
BONES = [(1, 0), (4, 1), (7, 4), (10, 7),
         (2, 0), (5, 2), (8, 5), (11, 8),
         (3, 0), (6, 3), (9, 6), (12, 9),
         (15, 12), (13, 9), (16, 13), (18, 16), (20, 18),
         (14, 9), (17, 14), (19, 17), (21, 19)]


def _dirs(J):
    """每帧 21 根骨头的单位方向与长度。"""
    d = np.zeros((len(BONES), 3))
    L = np.zeros(len(BONES))
    for i, (c, p) in enumerate(BONES):
        v = J[c] - J[p]
        L[i] = np.linalg.norm(v)
        d[i] = v / max(L[i], 1e-9)
    return d, L


def _rot_between(a, b):
    """把单位向量 a 转到 b 的最短弧：返回轴和角度。"""
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    ang = math.acos(dot)
    if s < 1e-9:
        if dot > 0:
            return np.array([0.0, 0.0, 1.0]), 0.0
        # 反向：任取一根垂直轴
        t = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            t = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, t)
        axis /= max(np.linalg.norm(axis), 1e-9)
        return axis, math.pi
    return axis / s, ang


def _slerp(a, b, u):
    axis, ang = _rot_between(a, b)
    if ang < 1e-9:
        return a.copy()
    c, s = math.cos(ang * u), math.sin(ang * u)
    return (a * c + np.cross(axis, a) * s +
            axis * np.dot(axis, a) * (1 - c))


def _angles(J):
    """每帧相邻帧之间的最大骨头转角（度）。"""
    out = np.zeros(max(1, len(J) - 1))
    for i in range(len(J) - 1):
        da, _ = _dirs(J[i])
        db, _ = _dirs(J[i + 1])
        m = 0.0
        for k in range(len(BONES)):
            cs = float(np.clip(np.dot(da[k], db[k]), -1.0, 1.0))
            m = max(m, math.degrees(math.acos(cs)))
        out[i] = m
    return out


def _rate(J, tail=True, w=3):
    """段末或段首的最大骨头角速度（度/帧），用 w 帧平均。"""
    a = _angles(J)
    if len(a) == 0:
        return 0.0
    if tail:
        return float(np.mean(a[-w:]))
    return float(np.mean(a[:w]))


def _u_sequences(n, scheme, amp):
    """四种方案的路径参数序列，形状 (n, 21)。"""
    t = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    if scheme == "uniform":
        return np.repeat(t[:, None], len(BONES), axis=1)
    if scheme == "stagger":
        w = 0.5 + 0.5 * (amp / max(amp.max(), 1e-9))
        w = np.clip(w, 0.35, 1.0)
        start = 1.0 - w
        uk = (t[:, None] - start[None, :]) / np.maximum(1e-6, 1.0 - start[None, :])
        return np.clip(uk, 0.0, 1.0)
    raise ValueError(scheme)


def make_frames(JA, JB, n, scheme="uniform"):
    """生成中间帧的关节坐标。

    JA, JB: (T,22,3)。用 JA 的末帧和 JB 的首帧作为两端。
    骨长取两端骨长的平均，全程固定。
    """
    A, B = JA[-1], JB[0]
    dA, LA = _dirs(A)
    dB, LB = _dirs(B)
    L = 0.5 * (LA + LB)
    amp = np.array([_rot_between(dA[k], dB[k])[1] for k in range(len(BONES))])

    if scheme == "linear_pos":
        frames = np.zeros((n, 22, 3))
        for i in range(n):
            u = (i + 1) / (n + 1.0)
            frames[i] = A * (1 - u) + B * u
        return frames

    if scheme == "shaped":
        r_in, r_out = _per_bone_rates(JA, JB)
        us = _shaped_u(n, amp, r_in, r_out)
    else:
        us = _u_sequences(n, scheme, amp)
    frames = np.zeros((n, 22, 3))
    for i in range(n):
        # 根位置线性过渡
        u_root = (i + 1) / (n + 1.0)
        frames[i, 0] = A[0] * (1 - u_root) + B[0] * u_root
        for k, (c, p) in enumerate(BONES):
            frames[i, c] = frames[i, p] + L[k] * _slerp(dA[k], dB[k], us[i, k])
    return frames


def bone_deviation(frames, JA, JB):
    """每根骨头在中间帧的长度，与两端平均相比的最大相对偏差。"""
    _, LA = _dirs(JA[-1])
    _, LB = _dirs(JB[0])
    L = 0.5 * (LA + LB)
    worst = 0.0
    for f in frames:
        _, Lf = _dirs(f)
        worst = max(worst, float(np.max(np.abs(Lf - L) / np.maximum(L, 1e-9))))
    return worst


def _seg_seg(p1, q1, p2, q2):
    """两条线段之间的最短距离。"""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(p1 - p2))
    if a < 1e-12:
        s, tt = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e < 1e-12:
            tt, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
            tt = (b * s + f) / e
            if tt < 0:
                tt, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif tt > 1:
                tt, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * tt)))


def min_separation(frames, exclude_shared=True):
    """不相邻骨段之间的最小距离（米）。"""
    pairs = []
    for i in range(len(BONES)):
        for j in range(i + 1, len(BONES)):
            ci, pi = BONES[i]
            cj, pj = BONES[j]
            if exclude_shared and ({ci, pi} & {cj, pj}):
                continue
            pairs.append((i, j))
    worst = 1e9
    for f in frames:
        for i, j in pairs:
            ci, pi = BONES[i]
            cj, pj = BONES[j]
            dd = _seg_seg(f[pi], f[ci], f[pj], f[cj])
            worst = min(worst, dd)
    return worst


def endpoint_rates(JA, JB, frames):
    """过渡第一步、最后一步的角度变化，与两端实测速率对比。"""
    ra = _rate(JA, tail=True)
    rb = _rate(JB, tail=False)
    step_in = 0.0
    step_out = 0.0
    if len(frames) > 0:
        dA, _ = _dirs(JA[-1])
        d1, _ = _dirs(frames[0])
        step_in = max(math.degrees(math.acos(float(np.clip(np.dot(dA[k], d1[k]), -1, 1))))
                      for k in range(len(BONES)))
        dN, _ = _dirs(frames[-1])
        dB, _ = _dirs(JB[0])
        step_out = max(math.degrees(math.acos(float(np.clip(np.dot(dN[k], dB[k]), -1, 1))))
                       for k in range(len(BONES)))
    return ra, step_in, step_out, rb

def _per_bone_rates(JA, JB, w=3):
    """每根骨头在 A 末段、B 首段的平均角速度（度/帧）。"""
    dA = [_dirs(J)[0] for J in JA[-1 - w:]]
    dB = [_dirs(J)[0] for J in JB[:1 + w]]
    def rate(seq):
        out = np.zeros(len(BONES))
        for k in range(len(BONES)):
            acc = []
            for i in range(len(seq) - 1):
                cs = float(np.clip(np.dot(seq[i][k], seq[i + 1][k]), -1, 1))
                acc.append(math.degrees(math.acos(cs)))
            out[k] = float(np.mean(acc)) if acc else 0.0
        return out
    return rate(dA), rate(dB)


def _shaped_u(n, amp, r_in, r_out):
    """每根骨头一条三次曲线：起止点的步长等于该骨头的实测角速度。"""
    out = np.zeros((n, len(BONES)))
    t = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    for k in range(len(BONES)):
        if amp[k] < 1e-9:
            out[:, k] = t
            continue
        m0 = float(r_in[k]) * (n + 1.0) / amp[k]
        m1 = float(r_out[k]) * (n + 1.0) / amp[k]
        m0 = min(max(m0, 0.0), 3.0)
        m1 = min(max(m1, 0.0), 3.0)
        t2, t3 = t * t, t * t * t
        u = (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) + (t3 - t2) * m1
        out[:, k] = np.clip(u, 0.0, 1.0)
    return out

def closest_pair(frames):
    """返回 (最小距离, (骨i, 骨j), 帧号)。"""
    pairs = []
    for i in range(len(BONES)):
        for j in range(i + 1, len(BONES)):
            ci, pi = BONES[i]
            cj, pj = BONES[j]
            if {ci, pi} & {cj, pj}:
                continue
            pairs.append((i, j))
    best = (1e9, None, -1)
    for f_i, f in enumerate(frames):
        for i, j in pairs:
            ci, pi = BONES[i]
            cj, pj = BONES[j]
            dd = _seg_seg(f[pi], f[ci], f[pj], f[cj])
            if dd < best[0]:
                best = (dd, (i, j), f_i)
    return best


def subtree(bone_index):
    children = {}
    for k, (c, p) in enumerate(BONES):
        children.setdefault(p, []).append(k)
    out = []
    stack = [bone_index]
    while stack:
        k = stack.pop()
        out.append(k)
        c, _p = BONES[k]
        stack.extend(children.get(c, []))
    return out


def apply_detour(frames, bone_index, deg):
    """把 bone_index 及其下游绕它的根关节往外转 deg 度（按 sin 塑形，两端为 0）。"""
    import math as _m
    n = len(frames)
    c0, p0 = BONES[bone_index]
    sub = subtree(bone_index)
    joints = {BONES[k][0] for k in sub} | {p0}
    out = frames.copy()
    for i in range(n):
        f = frames[i]
        u = (i + 1) / (n + 1.0)
        th = _m.radians(deg) * _m.sin(_m.pi * u)
        if abs(th) < 1e-9:
            continue
        d = f[c0] - f[p0]
        d = d / max(np.linalg.norm(d), 1e-9)
        w = f[p0] - f[0]
        w[1] = 0.0
        if np.linalg.norm(w) < 1e-6:
            w = np.array([1.0, 0.0, 0.0])
        w = w / np.linalg.norm(w)
        ax = np.cross(d, w)
        na = np.linalg.norm(ax)
        if na < 1e-9:
            continue
        ax = ax / na
        C, S = _m.cos(th), _m.sin(th)
        for jj in joints:
            v = f[jj] - f[p0]
            v = v * C + np.cross(ax, v) * S + ax * float(np.dot(ax, v)) * (1 - C)
            out[i, jj] = f[p0] + v
    return out

def separation_vs_baseline(frames, JA, JB):
    """每个骨对在过渡中的最小距离，与该对在两个端点姿态里的距离比较。

    只有「过渡中比两端都更近」的骨对才算可疑。这样能排除解剖上本来就靠得近的
    组合（例如胸与头），不会被当成碰撞。
    返回 (最严重的一对, 该对的最小距离, 该对的端点基线, 帧号, 全部越界对数)
    """
    pairs = []
    for i in range(len(BONES)):
        for j in range(i + 1, len(BONES)):
            ci, pi = BONES[i]
            cj, pj = BONES[j]
            if {ci, pi} & {cj, pj}:
                continue
            pairs.append((i, j))

    def dist(f, i, j):
        ci, pi = BONES[i]
        cj, pj = BONES[j]
        return _seg_seg(f[pi], f[ci], f[pj], f[cj])

    A, B = JA[-1], JB[0]
    worst = None
    n_bad = 0
    for i, j in pairs:
        base = min(dist(A, i, j), dist(B, i, j))
        mn, mf = 1e9, -1
        for fi, f in enumerate(frames):
            dd = dist(f, i, j)
            if dd < mn:
                mn, mf = dd, fi
        if mn < base - 1e-4:
            n_bad += 1
            drop = base - mn
            if worst is None or drop > worst[0]:
                worst = (drop, (i, j), mn, base, mf)
    if worst is None:
        return None, None, None, -1, 0
    drop, pair, mn, base, mf = worst
    return pair, mn, base, mf, n_bad
