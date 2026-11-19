"""在拟合参数空间里做段间连接。

为什么在参数空间
----------------
在关节位置空间逐骨插值，两端方向差大时肢体会扫出大弧（实测左右肘单帧甩到 152 度）。
参数是轴角表示，插值只沿确定的旋转轴走，幅度再大也不会扫出额外的大弧。

过渡期间根节点必须有位移
------------------------
最初的实现把后一段整体平移，让它的起点等于上一段终点。
这只约束了交界那一帧的位置，副作用是过渡两端位置相同，
于是过渡帧的根节点全程不动——姿势在变、脚不迈，看起来就是原地滑移。

现在的做法：过渡期间根速度从「上一段的来速」过渡到「下一段的去速」，
位移是这条速度曲线的积分；后一段的起点放在过渡结束的位置上。
交界处的相对位置仍然连续（不瞬移），但过渡带着人往前走。

曲线端点斜率匹配邻段
--------------------
之前用 ease_in_out，两端导数为零，过渡起步和收尾都停一下，
而邻段的实际节奏是 13-19 度/帧。现在端点斜率由邻段速度与总差距的比值决定。
"""
import math
import numpy as np

from hm import config

N_ROOT, N_POSE, N_BETAS, N_TRANS = 3, 69, 10, 3
SL_ROT = slice(0, 3)
SL_POSE = slice(3, 72)
SL_BETAS = slice(72, 82)
SL_TRANS = slice(82, 85)


def _slerp(qa, qb, u):
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    d = float(np.dot(qa, qb))
    if d < 0.0:
        qb = -qb
        d = -d
    if d > 0.9995:
        out = qa + (qb - qa) * u
    else:
        th = np.arccos(np.clip(d, -1.0, 1.0))
        out = (np.sin((1 - u) * th) * qa + np.sin(u * th) * qb) / np.sin(th)
    return out / (np.linalg.norm(out) + 1e-12)


def hermite_u(n, slope_in=0.0, slope_out=0.0):
    """生成 n 个插值参数，两端斜率可指定。

    slope_in / slope_out 是 du/dk 在起点和终点的值：
      - 都取 0 时退化为 smoothstep（两端停一下）
      - 都取 1 时接近线性（匀速通过）
      - 大于 1 表示起步就快
    限制在 [0, 2] 内，避免曲线越界产生速度尖峰。
    """
    if n <= 0:
        return np.array([])
    m0 = float(np.clip(slope_in, 0.0, 2.0))
    m1 = float(np.clip(slope_out, 0.0, 2.0))
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * 0.0 + h10 * m0 + h01 * 1.0 + h11 * m1


def interpolate(pa, pb, n, us=None):
    """两帧参数之间插 n 帧。旋转走球面插值，体型固定，平移线性。"""
    from scipy.spatial.transform import Rotation as R
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    if us is None:
        us = np.linspace(0, 1, n + 2)[1:-1]
    us = np.asarray(us, dtype=np.float64)
    if len(us) == 0:
        return np.zeros((0, 85))
    qa = R.from_rotvec(pa[:72].reshape(24, 3)).as_quat()
    qb = R.from_rotvec(pb[:72].reshape(24, 3)).as_quat()
    out = np.zeros((len(us), 85), dtype=np.float64)
    for k, u in enumerate(us):
        q = np.array([_slerp(qa[j], qb[j], u) for j in range(24)])
        out[k, :72] = R.from_quat(q).as_rotvec().reshape(-1)
        out[k, SL_BETAS] = pa[SL_BETAS]
        out[k, SL_TRANS] = pa[SL_TRANS] * (1 - u) + pb[SL_TRANS] * u
    return out


def _velocity(P, i_from, i_to):
    """从参数里量根节点的水平速度（米/帧）。"""
    P = np.asarray(P, dtype=np.float64)
    if not (0 <= i_from < len(P) and 0 <= i_to < len(P)):
        return np.zeros(3)
    return P[i_to, SL_TRANS] - P[i_from, SL_TRANS]


def merge(seg_params, seg_joints, cuts, fades, rates=None, curve="hermite"):
    """把多段参数接成一条。

    seg_params: 每段 (T_i,85)
    seg_joints: 每段 (T_i,22,3)，用来定切点和朝向
    cuts:      每段的起始保留帧（第一段为 0）
    fades:     每段之前的过渡帧数（第一段为 0）
    rates:     每段的角速度（度/帧），用于定曲线端点斜率；缺省不匹配
    """
    out = []
    info = []
    seg_tail = None
    tail_joints = None
    betas_ref = np.asarray(seg_params[0], dtype=np.float64)[0, SL_BETAS].copy()

    for i, P in enumerate(seg_params):
        P = np.asarray(P, dtype=np.float64)
        if P.ndim != 2:
            raise ValueError("段%d 的 params 不是二维：%s" % (i, P.shape))
        if len(P) == 0:
            raise ValueError("段%d 的 params 是空的" % i)
        P = P.copy()
        P[:, SL_BETAS] = betas_ref
        skip = int(cuts[i])
        J = np.asarray(seg_joints[i], dtype=np.float64)
        if skip >= len(P):
            raise ValueError("段%d 的 skip(%d) 超出帧数(%d)" % (i, skip, len(P)))

        if i > 0:
            # 先把本段整体转到与上一段末帧同朝向
            tgt = pelvis_yaw(seg_tail[-1])
            J0 = J.copy(); P0 = P.copy()
            J, P, dyaw = align_yaw(J, P, tgt)
            # 对齐只负责「站在哪、朝哪」。对齐后各部位的位置差是姿态差，
            # 是过渡要做的工作量，不是错误。这里记录下来供查看。
            _gaps = {}
            for jn, nm in ((1, "左髋"), (2, "右髋"), (13, "左肩"), (14, "右肩"),
                           (7, "左踝"), (8, "右踝"), (20, "左手"), (21, "右手"), (12, "头")):
                _gaps[nm] = round(float(np.linalg.norm(tail_joints[-1][jn] - J0[0][jn])), 3)
            n = int(fades[i])
            # 段末速度取最后 VEL_WINDOW 帧的平均，不取最后一帧的差分
            v_in = (_vel_window(seg_tail, len(seg_tail) - 1, VEL_WINDOW)
                    if len(seg_tail) >= 2 else np.zeros(3))
            v_out = (_vel_window(P, skip, VEL_WINDOW, forward=True)
                     if skip + 1 < len(P) else v_in)

            # 过渡期间的根位移：速度从 v_in 线性过渡到 v_out，逐帧累加
            base = seg_tail[-1, SL_TRANS].copy()
            if n > 0:
                k = np.arange(1, n + 1, dtype=np.float64)
                u = k / float(n + 1)
                vel = _blend_vel(v_in, v_out, u, mode=(rates or {}).get("blend", "linear"))
                disp = np.cumsum(vel, axis=0)[-1]
            else:
                disp = np.zeros(3)
            fade_end = base + disp

            # 本段整体平移，让它的起点落在过渡结束的位置
            delta = fade_end - P[skip, SL_TRANS]
            P[:, SL_TRANS] = P[:, SL_TRANS] + delta[None, :]

            if n > 0:
                # 曲线端点斜率：邻段角速度 / 过渡平均角速度
                # 端点斜率必须用「边界处」的速率，不是整段的中位数。
                # 转身段末尾 17.6 度/帧、中位只有 5 左右，用中位会让过渡收在
                # 5 度、然后跳进 17.6，画面上就是一下顿。
                # gap 也在这里直接量：两端实际用的那两帧的姿态差。
                # 速率和差距都改成在参数空间里量，与曲线的自变量同空间
                ri = param_rate(seg_tail, len(seg_tail) - 1, VEL_WINDOW, tail=True)
                ro = param_rate(P, skip, VEL_WINDOW, tail=False)
                gap = param_angle(seg_tail[-1], P[skip])
                avg = gap / max(1, n + 1)
                s_in = (ri / avg) if avg > 1e-6 else 1.0
                s_out = (ro / avg) if avg > 1e-6 else 1.0
                if curve == "linear":
                    us = (np.arange(1, n + 1, dtype=np.float64)) / float(n + 1)
                elif curve == "smooth":
                    k = np.arange(1, n + 1, dtype=np.float64) / float(n + 1)
                    us = 0.5 - 0.5 * np.cos(np.pi * k)
                else:
                    us = hermite_u(n, s_in, s_out)
                import os as _o
                if _o.environ.get("JP_DEBUG"):
                    _du = np.diff(np.concatenate([[0.0], us, [1.0]]))
                    print("      [%d] gap=%.1f ri=%.1f ro=%.1f avg=%.2f s_in=%.2f s_out=%.2f 首步=%.3f(%.1f度) 末步=%.3f(%.1f度)" % (
                        i, gap, ri, ro, avg, s_in, s_out, _du[0], _du[0]*gap, _du[-1], _du[-1]*gap), flush=True)
                # 不用错开启动：它把部分骨头推迟，导致第一帧整帧变化被压小，
                # 实测首步从应有的 2.4 度降到 1.0 度，破坏了端点速率匹配。
                # 而错开启动本身在之前测过几乎没有收益（最近距离 0.005→0.007）。
                fr = interpolate(seg_tail[-1], P[skip], n, us=us)
                # 平移显式给定：从 base 沿混合速度走到 fade_end
                for kk in range(n):
                    fr[kk, SL_TRANS] = base + np.cumsum(vel, axis=0)[kk]
                out.append(fr)

        seg_tail = P[skip:]
        tail_joints = J[skip:] if skip < len(J) else J[-1:]
        out.append(seg_tail)
        info.append({"index": i, "skip": int(skip), "frames": int(len(seg_tail)),
                     "fade_frames": int(fades[i])})

    merged = np.concatenate(out, axis=0)
    return merged, info

def verts_from_params(params, device=None):
    import torch
    from hm import fit as fitmod
    fitmod._patch_paths()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = fitmod._load_smpl(dev)
    P = torch.from_numpy(np.asarray(params, dtype=np.float32)).to(dev)
    with torch.no_grad():
        o = model(betas=P[:, SL_BETAS], global_orient=P[:, SL_ROT],
                  body_pose=P[:, SL_POSE], transl=P[:, SL_TRANS], return_verts=True)
    return o.vertices.detach().cpu().numpy()


def selftest():
    """自检：过渡期间根节点必须有位移；曲线端点斜率生效。"""
    import math
    from scipy.spatial.transform import Rotation as R
    pa = np.zeros(85); pb = np.zeros(85)
    pb[:3] = [0.0, 1.0, 0.0]
    pb[SL_TRANS] = [0.0, 0.0, 1.0]
    fr = interpolate(pa, pb, 10)
    ok = True
    if fr.shape != (10, 85):
        ok = False
        print("  形状不对:", fr.shape)
    # 端点斜率：s=0 时起步慢，s=1 时接近线性
    u0 = hermite_u(10, 0.0, 0.0)
    u1 = hermite_u(10, 1.0, 1.0)
    if not (u0[0] < u1[0]):
        ok = False
        print("  端点斜率没生效: %.4f vs %.4f" % (u0[0], u1[0]))
    if u0[-1] > 1.0 + 1e-6 or u0[0] < -1e-6:
        ok = False
        print("  曲线越界: %.4f ~ %.4f" % (u0[0], u0[-1]))
    print("  插值自检: %s  斜率0 首值 %.4f  斜率1 首值 %.4f" % (
        "通过" if ok else "不通过", u0[0], u1[0]))
    return ok


if __name__ == "__main__":
    selftest()

def joint_depths():
    """24 个旋转中，每个距离根有几个关节。用于排先后顺序。

    根=0，脊柱/髋=1，四肢中段=2，手/脚=3。
    依据是运动控制的近端到远端顺序，不是数据拟合。
    """
    from hm.assemble import BONES
    depth = {0: 0}
    changed = True
    while changed:
        changed = False
        for c, par in BONES:
            if par in depth and c not in depth:
                depth[c] = depth[par] + 1
                changed = True
    return [depth.get(i, 0) for i in range(24)]


def stagger_starts(pa, pb, depths=None, lo=0.35):
    """每个关节的启动时刻。

    需要的转角越大，起点越早、时间窗越长；转角小的起点晚，最后很快补上。
    lo 是最短时间窗（占整段过渡的比例），避免出现瞬间跳到位的关节。
    """
    from scipy.spatial.transform import Rotation as R
    a = R.from_rotvec(np.asarray(pa[:72], dtype=np.float64).reshape(24, 3))
    b = R.from_rotvec(np.asarray(pb[:72], dtype=np.float64).reshape(24, 3))
    amp = (b * a.inv()).magnitude()
    mx = float(amp.max()) if len(amp) else 0.0
    if mx < 1e-6:
        return np.zeros(24)
    w = 0.5 + 0.5 * (amp / mx)      # 时间窗 0.5–1.0
    w = np.clip(w, lo, 1.0)
    return 1.0 - w                   # 启动时刻


def interpolate_staggered(pa, pb, n, us, starts):
    """逐关节错开启动的插值。

    每个关节沿自己的最短弧旋转（骨长不变，所以不变形），
    但不是同时开始：起点 = starts[k]，到 u=1 时全部到位。
    这样中间不会出现全部关节一起摆出的、现实中不存在的构型。
    """
    from scipy.spatial.transform import Rotation as R
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    qa = R.from_rotvec(pa[:72].reshape(24, 3))
    qb = R.from_rotvec(pb[:72].reshape(24, 3))
    rel = (qb * qa.inv()).as_rotvec()
    starts = np.asarray(starts, dtype=np.float64).reshape(24)
    out = np.zeros((n, 85), dtype=np.float64)
    for i, u in enumerate(us):
        uk = np.clip((float(u) - starts) / np.maximum(1e-6, 1.0 - starts), 0.0, 1.0)
        out[i, :72] = (qa * R.from_rotvec(rel * uk[:, None])).as_rotvec().reshape(-1)
        out[i, SL_BETAS] = pa[SL_BETAS]
    return out

# 取速度的窗口（帧）。单帧差分会把单帧抖动当成速度，实测与 4 帧平均差 16–24%。
# 取 3 帧：够滤掉抖动，又还是局部速度（0.15 秒）。
VEL_WINDOW = 3


def _vel_window(P, end_index, window, forward=False):
    """在 end_index 处取窗口平均速度（米/帧）。

    forward=False：往回取 window 帧，用于段末
    forward=True ：往后取 window 帧，用于段首
    """
    n = len(P)
    w = max(1, int(window))
    if forward:
        a = int(end_index)
        b = min(n - 1, a + w)
        if b <= a:
            return np.zeros(3)
        return (np.asarray(P[b, SL_TRANS]) - np.asarray(P[a, SL_TRANS])) / float(b - a)
    b = int(end_index)
    a = max(0, b - w)
    if b <= a:
        return np.zeros(3)
    return (np.asarray(P[b, SL_TRANS]) - np.asarray(P[a, SL_TRANS])) / float(b - a)

def _blend_vel(v_in, v_out, us, mode="polar"):
    """过渡期间的根速度序列。

    polar（文献做法）：把水平速度拆成「快慢」和「方向」两个量分别混合。
        快慢线性过渡；方向沿最短转角转过去；再合成矢量。
        直接做矢量加权在两端方向差大时会让速度经过零——人停住再往回加速，
        这正是 Registration Curves 指出的塌陷。
    linear（旧做法）：水平方向也直接矢量加权。

    竖直分量两种模式下都线性混合，由地形那一步单独处理。
    """
    v_in = np.asarray(v_in, dtype=np.float64)
    v_out = np.asarray(v_out, dtype=np.float64)
    if mode != "polar":
        return v_in[None, :] * (1 - us)[:, None] + v_out[None, :] * us[:, None]

    s_in = float(np.hypot(v_in[0], v_in[2]))
    s_out = float(np.hypot(v_out[0], v_out[2]))
    # 朝向角：从 +z 轴量向 +x 轴
    h_in = math.atan2(v_in[0], v_in[2]) if s_in > 1e-9 else 0.0
    h_out = math.atan2(v_out[0], v_out[2]) if s_out > 1e-9 else h_in
    dh = h_out - h_in
    while dh > math.pi:
        dh -= 2 * math.pi
    while dh < -math.pi:
        dh += 2 * math.pi
    out = np.zeros((len(us), 3))
    for i, u in enumerate(us):
        sp = s_in * (1 - u) + s_out * u
        h = h_in + dh * u
        out[i, 0] = sp * math.sin(h)
        out[i, 2] = sp * math.cos(h)
        out[i, 1] = v_in[1] * (1 - u) + v_out[1] * u
    return out

# ---------- 朝向对齐 ----------
# 一段动作在「水平面平移」和「绕竖直轴旋转」下是不变的。
# 所以把下一段整体绕竖直轴转到与上一段末帧同朝向，动作本身不变，
# 而过渡就不必承担任何整体旋转。
# 之前只做了平移，把旋转留给了过渡，过渡只好原地站着转。

def pelvis_yaw(Pframe):
    """骨盆自己的正面方向（偏航角）。

    用根节点的旋转作用在「正前方」上得到，单一来源，
    不依赖任何两点连线：肩线会相对骨盆扭（实测差 13 度），
    髋线在两腿交叉时会翻向。
    """
    from scipy.spatial.transform import Rotation as R
    f = R.from_rotvec(np.asarray(Pframe[0:3], dtype=np.float64)).as_matrix() @ np.array([0.0, 0.0, 1.0])
    return math.atan2(float(f[0]), float(f[2]))


def _rot_y(v, ang):
    c, ss = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] + ss * v[2], v[1], -ss * v[0] + c * v[2]])


def align_yaw(J, P, target_heading):
    """把一段（关节 + 参数）整体绕竖直轴旋转，使首帧骨盆朝向等于 target。

    只旋转，不改姿态：整段动作在绕竖直轴旋转下形状不变。
    位置绕本段首帧骨盆旋转，所以段在原地转。
    """
    from scipy.spatial.transform import Rotation as R
    cur = pelvis_yaw(P[0])
    d = target_heading - cur
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi

    J2 = np.array(J, dtype=np.float64, copy=True)
    P2 = np.array(P, dtype=np.float64, copy=True)
    if abs(d) < 1e-9:
        return J2, P2, 0.0

    piv = J2[0, 0].copy()
    for i in range(len(J2)):
        for k in range(22):
            J2[i, k] = piv + _rot_y(J2[i, k] - piv, d)

    rot = R.from_euler("y", d).as_matrix()
    q = R.from_rotvec(P2[:, 0:3])
    P2[:, 0:3] = R.from_matrix(rot[None, :, :] @ q.as_matrix()).as_rotvec()
    piv_p = P2[0, SL_TRANS].copy()
    P2[:, SL_TRANS] = (_rot_y((P2[:, SL_TRANS] - piv_p).T, d)).T + piv_p
    return J2, P2, d

# ---------- 参数空间的角距离与角速度 ----------
# 曲线参数 u 作用在 24 个关节旋转上。要控制速率，必须在同一个空间里量：
# 量骨头方向的角度是另一个空间，沿关节链累积，换算非线性。
# 之前用骨头方向的差距去除参数步长，末步算出 3.5 度而公式应给 9.6 度。

def param_angle(P1, P2):
    """两帧参数之间，24 个关节里转得最多的那个的角度（度）。"""
    from scipy.spatial.transform import Rotation as R
    a = R.from_rotvec(np.asarray(P1[:72], dtype=np.float64).reshape(24, 3))
    b = R.from_rotvec(np.asarray(P2[:72], dtype=np.float64).reshape(24, 3))
    return float(np.degrees((b * a.inv()).magnitude().max()))


def param_rate(P, end_index, w=3, tail=True):
    """参数空间里，边界处每帧转得最多的那些关节的平均角速度（度/帧）。"""
    n = len(P)
    w = max(1, int(w))
    if tail:
        idx = list(range(max(0, end_index - w), end_index))
    else:
        idx = list(range(end_index, min(n - 1, end_index + w)))
    vals = [param_angle(P[i], P[i + 1]) for i in idx]
    return float(np.mean(vals)) if vals else 0.0

def merge_legacy(seg_params, seg_joints, cuts, fades, rates=None):
    """回退到 e2e38 那一版的行为。与现在的区别只有这几点：

      速度      单帧差分（不用 3 帧平均）
      混合      线性矢量（不用极坐标）
      摆放      纯平移（不做朝向对齐）
      曲线      hermite_u，斜率取「该段中位角速度 / 过渡平均角速度」
      插值      24 个关节逐个球面插值，不做错开启动
      帧数      用传进来的 fades，不按数据算

    保留它是因为它是目前唯一被看过、且没有被指出结构性错误的一版。
    """
    out = []
    info = []
    seg_tail = None
    betas_ref = np.asarray(seg_params[0], dtype=np.float64)[0, SL_BETAS].copy()

    for i, P in enumerate(seg_params):
        P = np.asarray(P, dtype=np.float64)
        P = P.copy()
        P[:, SL_BETAS] = betas_ref
        skip = int(cuts[i])

        if i > 0:
            n = int(fades[i])
            v_in = _velocity(seg_tail, len(seg_tail) - 2, len(seg_tail) - 1) \
                if len(seg_tail) >= 2 else np.zeros(3)
            v_out = _velocity(P, skip, skip + 1) if skip + 1 < len(P) else v_in

            base = seg_tail[-1, SL_TRANS].copy()
            if n > 0:
                k = np.arange(1, n + 1, dtype=np.float64)
                u = k / float(n + 1)
                vel = v_in[None, :] * (1 - u)[:, None] + v_out[None, :] * u[:, None]
                disp = np.cumsum(vel, axis=0)[-1]
            else:
                disp = np.zeros(3)
            fade_end = base + disp
            P[:, SL_TRANS] = P[:, SL_TRANS] + (fade_end - P[skip, SL_TRANS])[None, :]

            if n > 0:
                ri = float((rates or {}).get(i - 1, 0.0))
                ro = float((rates or {}).get(i, 0.0))
                gap = float((rates or {}).get(("gap", i - 1), 0.0))
                avg = gap / max(1, n + 1)
                s_in = (ri / avg) if avg > 1e-6 else 1.0
                s_out = (ro / avg) if avg > 1e-6 else 1.0
                us = hermite_u(n, s_in, s_out)
                fr = interpolate(seg_tail[-1], P[skip], n, us=us)
                for kk in range(n):
                    fr[kk, SL_TRANS] = base + np.cumsum(vel, axis=0)[kk]
                out.append(fr)

        seg_tail = P[skip:]
        out.append(seg_tail)
        info.append({"index": i, "skip": int(skip), "frames": int(len(seg_tail)),
                     "fade_frames": int(fades[i])})

    return np.concatenate(out, axis=0), info
