"""段间连接：位置对齐 + 按角速度限速插帧 + 匹配端点速度的过渡轨迹。

设计原则
--------
1 位置不许跳
  MoMask 每段都从原点出发，轨迹由模型自己决定（263 维特征前四维）。
  所以后一段整体【平移】到上一段实际的终点上，不平移以外的任何改动。

2 帧数由角速度上限决定，不用拍脑袋的阈值
  人做快速肢体动作大约 8 度/帧（20 fps 下 160 度/秒）。
  N = 姿态差 / 8，夹在 4 到 24 之间。差得越多插得越多。

3 过渡轨迹两端要接上前后两段的速度，否则会「顿一下」
  之前用 smoothstep，两端导数都是零，看着就是起步停一下、收尾又停一下。
  改成三次 Hermite：起点斜率取上一段收尾的实际角速度，
  终点斜率取下一段起步的实际角速度，中间不断流。
  插值仍在旋转空间做，骨长保持不变。
"""
import math

import numpy as np

from hm import config
from hm.assemble import BONES

# 过渡帧数的限速基准。
# 不要设成固定值：实测 MoMask 自身输出的节奏是
#   walk 平均 12.8 度/帧、turn 6.0、run 17.7、wave 6.2，单帧最大到 25.9 度。
# 把过渡限成 8 度/帧会比前后动作慢一倍，画面上就是突然顿一下。
# 所以用前后两段实际的角速度作基准，夹在下面这个范围内。
RATE_MIN = 6.0
RATE_MAX = 22.0
MIN_FRAMES = 3
MAX_FRAMES = 40
# 每帧最多转多少度。180 度转身用 1 秒是自然的，20 fps 下就是 9 度/帧。
# 超过这个值会看不清，所以时长有物理下限。
MAX_DEG_PER_FRAME = 9.0


def pose_gap(pa, pb, detail=False):
    """两帧姿态的差异（度）。

    按骨长加权，并把最差的那根骨头也纳入：
      - 只取平均会把一根手臂差 180 度稀释掉（21 根接近 0 度的骨头拉低均值）
      - 大腿转 10 度和手指转 10 度不该同权，所以按骨长加权
    返回值 = 0.5 × 加权平均 + 0.5 × 最大值，兼顾整体和局部。
    """
    ang, w = [], []
    for c, p in BONES:
        da, db = pa[c] - pa[p], pb[c] - pb[p]
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        if na < 1e-9 or nb < 1e-9:
            continue
        cos = float(np.clip(np.dot(da / na, db / nb), -1.0, 1.0))
        ang.append(math.degrees(math.acos(cos)))
        w.append(0.5 * (na + nb))
    if not ang:
        return (0.0, 0.0, 0.0) if detail else 0.0
    a = np.asarray(ang)
    wt = np.asarray(w)
    wmean = float((a * wt).sum() / wt.sum())
    vmax = float(a.max())
    score = 0.5 * wmean + 0.5 * vmax
    if detail:
        return score, wmean, vmax
    return score


def pose_speed(prev, cur):
    """这一帧到下一帧的姿态变化速度（度/帧）。"""
    return pose_gap(prev, cur)


def translate_to(seg, xy):
    """纯平移。姿态一点不动。"""
    seg = np.asarray(seg, dtype=np.float64).copy()
    seg[..., 0] += float(xy[0]) - seg[0, 0, 0]
    seg[..., 2] += float(xy[1]) - seg[0, 0, 2]
    return seg


def find_cut(ta, hb, tail=16, head=16, height_weight=0.25, max_bone_deg=110.0):
    """在两端附近找最合适的一对帧，从那里切开。

    判据是【最差的那根骨头】，不是平均。
    原因：一根骨头要翻 180 度时，无论走最短弧还是走两段路，
    中间都必须经过与两端垂直的方向。对髋、腿这类骨头，那就是整个人被甩开，
    画面上就是原地转圈；而走最短弧在 180 度附近数值不稳定，会单帧跳 159 度。
    两种异常都来自同一个原因，所以只能从选帧这一层解决：不选这种帧对。

    实测：三个接缝的候选里，最差骨角最小是 23.7 / 58.6 / 53.3 度，
    根本不需要接近 180 度的帧对，是原来的平均打分数把好的候选压掉了。

    返回 (A 截断到第几帧, B 从第几帧起, 最差骨角, 高度差)。
    """
    na, nb = len(ta), len(hb)
    i0 = max(1, na - tail)
    j1 = min(nb, head)
    best = None
    for i in range(i0, na):
        for j in range(0, j1):
            m, _ = max_bone_angle(ta[i], hb[j])
            if m > max_bone_deg:
                continue
            dh = abs(float(ta[i][0][1]) - hb[j][0][1])
            score = m + height_weight * dh * 100.0
            if best is None or score < best[0]:
                best = (score, i + 1, j, m, dh)
    if best is None:
        # 没有一对待满足约束，退回最差骨角最小的一对
        fallback = None
        for i in range(i0, na):
            for j in range(0, j1):
                m, _ = max_bone_angle(ta[i], hb[j])
                if fallback is None or m < fallback[0]:
                    dh = abs(float(ta[i][0][1]) - hb[j][0][1])
                    fallback = (m, i + 1, j, m, dh)
        _, ci, cj, m, dh = fallback
        return ci, cj, m, dh
    _, ci, cj, m, dh = best
    return ci, cj, m, dh


def max_bone_angle(pa, pb):
    """两帧之间转动最大的那根骨骼的夹角（度），以及是哪个关节。"""
    worst, wj = 0.0, -1
    for c, p in BONES:
        da, db = pa[c] - pa[p], pb[c] - pb[p]
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        if na < 1e-9 or nb < 1e-9:
            continue
        cosv = float(np.clip(np.dot(da / na, db / nb), -1.0, 1.0))
        ang = math.degrees(math.acos(cosv))
        if ang > worst:
            worst, wj = ang, c
    return worst, wj

def frame_count(gap_deg, max_deg=None, rate=None):
    """算需要几帧。

    角度用【最差那根骨头】，速度用【前后两段的实际角速度】。
    看得见的速度取决于动得最快的那根骨头，也取决于周围动作有多快。
    """
    ref = float(max_deg) if max_deg is not None else float(gap_deg)
    r = float(rate) if rate is not None else 12.0
    r = max(RATE_MIN, min(RATE_MAX, r))
    n = int(math.ceil(ref / r))
    return max(MIN_FRAMES, min(MAX_FRAMES, n))


def hermite_params(n, v_in, v_out, gap):
    """生成 n 个中间参数 u，端点斜率接上前后两段的速度。

    v_in / v_out：上一段收尾、下一段起步的角速度（度/帧）。
    gap：两端总姿态差（度）。
    导数 = 速度 / 总差，这样过渡的整体节奏和前后运动连贯。
    """
    if n <= 0:
        return np.array([])
    # 端点斜率必须限制在 1.5 以内。
    # 放到 2.0 时三次曲线会冲过 1 再回来，参数被裁剪后就是一次速度尖峰：
    # 实测出现过单帧 26.3 度的跳变。1.5 以内能保证单调不越界。
    m0 = float(np.clip(v_in / max(gap, 1e-6) * n, 0.0, 1.5))
    m1 = float(np.clip(v_out / max(gap, 1e-6) * n, 0.0, 1.5))
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * 0.0 + h10 * m0 + h01 * 1.0 + h11 * m1


WATCH_JOINTS = (16, 17, 18, 19, 20, 21)
TORSO_R = 0.15


def _push_out(pa, pb, strength=1.0):
    """把两端的手/前臂先往外推一点，避免最短弧路径从躯干内部穿过。

    做法：把这几根关节相对躯干的水平偏移方向放大。
    只改插值的端点，不改原始段（调用方传进来的已经是要插的两帧）。
    """
    pa = np.asarray(pa, dtype=np.float64).copy()
    pb = np.asarray(pb, dtype=np.float64).copy()
    for P in (pa, pb):
        a = P[0].copy()
        b = P[12].copy()
        axis = b - a
        for j in WATCH_JOINTS:
            v = P[j] - a
            t = float(np.clip(np.dot(v, axis) / (np.dot(axis, axis) + 1e-12), 0.0, 1.0))
            proj = a + axis * t
            radial = v - axis * t
            r = float(np.linalg.norm(radial))
            if r < 1e-6:
                continue
            need = TORSO_R + 0.05
            if r < need:
                P[j] = proj + radial / r * need * strength
    return pa, pb


# 左右手臂链：肩 -> 肘 -> 腕 -> 手
ARM_CHAINS = {
    (13, 16, 18, 20): (13, 9),   # 左臂，根在 关节9
    (14, 17, 19, 21): (14, 9),   # 右臂
}


def _fix_penetration_frame(pose, max_iter=6):
    """把穿进躯干的手臂绕肩向外转，直到离开躯干。旋转保骨长。"""
    pose = np.asarray(pose, dtype=np.float64).copy()
    a = pose[0].copy()
    b = pose[12].copy()
    axis = b - a
    for chain, (shoulder, _root) in ARM_CHAINS.items():
        for _ in range(max_iter):
            # 找这条链里最靠近躯干轴的那个关节
            worst_j, worst_d = None, 1e9
            for j in chain:
                v = pose[j] - a
                tt = float(np.clip(np.dot(v, axis) / (np.dot(axis, axis) + 1e-12), 0.0, 1.0))
                d = float(np.linalg.norm(v - axis * tt))
                if d < worst_d:
                    worst_d, worst_j = d, j
            if worst_d >= TORSO_R + 0.03:
                break
            # 绕肩关节，把该关节朝外旋一个固定角度
            pivot = pose[shoulder].copy()
            v = pose[worst_j] - a
            tt = float(np.clip(np.dot(v, axis) / (np.dot(axis, axis) + 1e-12), 0.0, 1.0))
            radial = v - axis * tt
            rn = np.linalg.norm(radial)
            if rn < 1e-6:
                radial = np.array([1.0, 0.0, 0.0])
                rn = 1.0
            out_dir = radial / rn
            # 旋转轴 = 躯干轴 × 外方向
            ax = np.cross(axis / (np.linalg.norm(axis) + 1e-12), out_dir)
            an = np.linalg.norm(ax)
            if an < 1e-6:
                break
            ax = ax / an
            ang = math.radians(8.0)
            c, s = math.cos(ang), math.sin(ang)
            for j in chain:
                v = pose[j] - pivot
                pose[j] = pivot + (v * c + np.cross(ax, v) * s +
                                   ax * np.dot(ax, v) * (1 - c))
    return pose


# 上半身关节：脊柱、颈、头、左右臂
UPPER = {3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}
LOWER = set(range(22)) - UPPER


def blend_frames(pa, pb, params, scope="full"):
    """插帧。先分离整体偏航，再处理残余姿态。

    为什么先分解偏航：
    两端整体朝向差接近 180 度时，逐骨最短弧的旋转轴是病态的，
    插到中间会突然翻面。实测出现过单帧 171.9 度的跳变。
    把偏航拿出来单独按角度线性插，就不会翻面。

    scope 控制哪些关节走旋转插值：
      full   全身
      upper  只有上半身走旋转插值，下半身用直接插值
      lower  反过来
    """
    from hm import decompose as DC

    dyaw = DC.yaw_gap(pa, pb)                      # 弧度
    pb_al = DC.rotate_yaw(pb, -dyaw)               # 对齐到 pa 的朝向
    out = []
    for u in params:
        u = float(np.clip(u, 0.0, 1.0))
        pose = _blend_one(pa, pb_al, u, scope)
        pose = DC.rotate_yaw(pose, dyaw * u)       # 按比例把偏航加回去
        out.append(pose)
    return out


def _blend_one(pa, pb, u, scope):
    """单帧姿态插值（偏航已对齐）。骨长保持不变。"""
    from hm.assemble import _quat_from_to, _slerp_pow, _rotate, BONES as B
    # 不要在这里调 _push_out。它直接把手臂关节推到躯干外侧固定距离，
    # 会改动骨长，而且每插一帧就把端点再改一次，
    # 导致过渡第一帧不等于上一段的真实最后一帧（交界处跳），
    # 网格也会被拉变形。实测它是单帧 159 度跳变的主要来源。
    active = set(UPPER) if scope == "upper" else (
        set(LOWER) if scope == "lower" else set(range(22)))
    u = float(np.clip(u, 0.0, 1.0))
    pose = np.zeros((22, 3), dtype=np.float64)
    pose[0] = pa[0] * (1.0 - u) + pb[0] * u
    for c, p in B:
        da, db = pa[c] - pa[p], pb[c] - pb[p]
        la, lb = np.linalg.norm(da), np.linalg.norm(db)
        if la < 1e-9 or lb < 1e-9:
            pose[c] = pose[p] + (da * (1.0 - u) + db * u)
            continue
        na_, nb_ = da / la, db / lb
        cosA = float(np.clip(np.dot(na_, nb_), -1.0, 1.0))
        ang = math.degrees(math.acos(cosA))
        L = la * (1.0 - u) + lb * u
        # 关键：不要对任何骨骼做「方向向量线性相加」。
        # 两端方向差 180 度时 da*0.5+db*0.5 约等于零，骨头会塌成零长度。
        # 之前 scope=upper 让下半身走这条线性分支，腿直接插没了，
        # 造成单帧 159 度的跳变、骨长波动 3-11%。
        # 现在所有骨骼都走旋转插值，骨长恒等于两端的线性插值。
        # 只走最短弧。转角超过 110 度的帧对在选切点时已经被排除，
        # 所以这里的角度都在数值稳定的范围内，不会翻面。
        # 之前那条「先转到垂直方向」的两段路分支已删除：
        # 它会让髋、腿这类骨头在过渡中间被甩到与两端都垂直的方向，
        # 画面上就是原地转圈（实测 0.25 秒内朝向甩出 140 度）。
        d = _rotate(na_, _slerp_pow(_quat_from_to(na_, nb_), u))
        pose[c] = pose[p] + (d / (np.linalg.norm(d) + 1e-12)) * L
    return pose


def build(segments, junction_params=None, actions=None, fade=True, verbose=False):
    """多段 → 一条序列。

    位置：纯平移，后一段起点 = 前一段实际终点。
    动作：按 junction_params 里每一处过渡的时长与曲线插中间帧。
          junction_params[i] 是第 i 段与第 i+1 段之间的参数，
          缺省时按两侧动作快慢取默认值。
    """
    from hm import transition_params as TP

    info_actions = dict(enumerate(actions or []))

    parts = []
    info = []
    cur = (0.0, 0.0)
    prev_tail = None
    for i, seg in enumerate(segments):
        a = translate_to(seg, cur)
        gap_pose, gap_h, n_fade = 0.0, 0.0, 0
        prm = None
        if prev_tail is not None:
            cut, skip, gap_pose, gap_h = find_cut(prev_tail, a)
            head = a[skip:]
            if len(head) < 6:
                head = a
                skip = 0
            # 切点必须真正作用到上一段上。
            # 之前算了 cut 却没用，parts 里仍是完整上一段，
            # 于是插值起点是上一段的真实结尾、gap 却按切点量，
            # 两者不是同一对帧：实际要跨的角度远大于量到的值，
            # 结果 10 帧里出现单帧 161 度的跳变。
            if cut < len(parts[-1]):
                parts[-1] = parts[-1][:cut]
            if fade:
                # 用含偏航的真实差距，不用只看骨骼方向的旧指标。
                # 实测过渡两端整体转了 179 度，旧指标只报 59.6 度，差了 3 倍，
                # 时长按 59.6 度给，结果 0.55 秒转半圈。
                from hm import decompose as DC
                prm = TP.clamp_params((junction_params or {}).get(i - 1) or
                                      TP.default_params(info_actions.get(i - 1, ""),
                                                        info_actions.get(i, "")))
                g = DC.total_gap(prev_tail[cut - 1], head[0])
                span = g["span_deg"]
                prm["span_deg"] = round(span, 1)
                prm["yaw_deg"] = round(g["yaw_deg"], 1)
                # 时长取两者较大：LLM 给的是「该快还是该慢」的意图，
                # MAX_DEG_PER_FRAME 是「这么快会不会看不清」的物理下限。
                # span 是度，MAX_DEG_PER_FRAME 是度/帧，
                # 相除得到的是【帧数】，要再除以帧率才是秒。
                # 之前漏了这一步，18.7 度被算成 2.08 秒（实际应为 0.10 秒）。
                need_frames = span / MAX_DEG_PER_FRAME
                need_s = need_frames / float(config.FPS)
                dur = max(float(prm["duration_s"]), need_s)
                prm["needed_s"] = round(need_s, 3)
                prm["final_s"] = round(dur, 3)
                n_fade = max(1, int(round(min(dur, TP.DURATION_RANGE[1] * 2.0) * config.FPS)))
                if prm.get("anticipate"):
                    uu = TP.anticipate_params(n_fade, prm["ease"])
                else:
                    uu = TP.ease_params(n_fade, prm["ease"])
                prm["fade_start_frame"] = int(sum(len(x) for x in parts))
                fr = blend_frames(parts[-1][-1], head[0], uu, scope=prm.get("scope", "full"))
                if fr:
                    parts.append(np.stack(fr, axis=0))
            parts.append(head)
            a = head
        else:
            parts.append(a)
        cur = (float(a[-1, 0, 0]), float(a[-1, 0, 2]))
        prev_tail = a
        # 记录在整条序列里的【真实】起止帧号。
        # 之前只记了截断前的帧数，导致外部按 info 推边界时全部错位。
        total = sum(len(x) for x in parts)
        info.append({
            "start_frame": int(total - len(a)),
            "end_frame": int(total),
            "index": i,
            "action": info_actions.get(i, "") if info_actions else "",
            "frames": int(len(a)),
            "gap_pose_deg": round(gap_pose, 1),
            "gap_height_m": round(gap_h, 3),
            "fade_frames": int(n_fade),
            "transition": prm,
            "start_xy": (round(float(a[0, 0, 0]), 3), round(float(a[0, 0, 2]), 3)),
            "end_xy": (round(float(a[-1, 0, 0]), 3), round(float(a[-1, 0, 2]), 3)),
        })
    seq = np.concatenate(parts, axis=0)
    return seq, info

def selftest():
    """自检：位置不跳；帧数够把每帧角度限制在阈值附近；骨长不塌。"""
    from hm import generate, to_joints
    prompts = ["a person walks forward", "a person turns around",
               "the person is running", "a person waves goodbye"]
    outs = generate.generate(prompts, [56, 40, 56, 40])
    Js = [to_joints.from_feats(F) for F in outs]
    seq, info = build(Js, fade=True)
    r = seq[:, 0, :]
    d = np.linalg.norm(np.diff(r, axis=0), axis=1)
    ok = True
    if d.max() > 0.15:
        ok = False
        print("  位置跳变 %.3f 米" % d.max())
    # 逐帧姿态变化速度（按最差骨头）不能超过阈值太多
    speeds = []
    for k in range(len(seq) - 1):
        _s, _w, _v = pose_gap(seq[k], seq[k + 1], detail=True)
        speeds.append(_v)
    speeds = np.asarray(speeds)
    if speeds.max() > RATE_MAX * 1.2:
        ok = False
        print("  单帧最大 %.1f 度（第 %d 帧），超过阈值的 1.8 倍" % (
            speeds.max(), int(np.argmax(speeds)) + 1))
    for c, p in BONES:
        L = np.linalg.norm(seq[:, c] - seq[:, p], axis=-1)
        if L.min() < 0.02:
            ok = False
            print("  关节%d 骨长塌陷" % c)
            break
    ys = seq[:, 0, 1]
    print("  连接自检: %s  总帧 %d  最大单帧位移 %.3f 米  高度摆动 %.3f 米" % (
        "通过" if ok else "不通过", len(seq), d.max(), ys.max() - ys.min()))
    for x in info:
        print("    段%d %3d帧  姿态差 %5.1f度  插 %2d 帧  每帧 %.1f 度" % (
            x["index"], x["frames"], x["gap_pose_deg"], x["fade_frames"],
            x["gap_pose_deg"] / max(1, x["fade_frames"] + 1)))
    return ok


if __name__ == "__main__":
    selftest()

def find_cut_active(ta, hb, tail=16, head=16, max_bone_deg=110.0):
    """挑「两侧都在动」的切点，而不是「两侧最像」的切点。

    原 find_cut 的目标是最差骨角最小，那么选出来的天然是两边都最接近站立的帧，
    过渡就建在两个站姿之间，插值期间人站着（实测过渡的位姿变化率中位 0.0 度/帧）。
    这里改成：先在姿态差不超过上限的候选里，取 min(两侧速率) 最大的一对。
    """
    na, nb = len(ta), len(hb)
    i0 = max(1, na - tail)
    j1 = min(nb - 1, head)

    def rate(seq, i):
        if i <= 0 or i + 1 >= len(seq):
            return 0.0
        return float(max_bone_angle(seq[i - 1], seq[i])[0])

    best = None
    for i in range(i0, na):
        for j in range(1, j1 + 1):
            m, _ = max_bone_angle(ta[i], hb[j])
            if m > max_bone_deg:
                continue
            act = min(rate(ta, i), rate(hb, j))
            key = (act, -m)
            if best is None or key > best[0]:
                dh = abs(float(ta[i][0][1]) - hb[j][0][1])
                best = (key, i + 1, j, m, dh)
    if best is None:
        return find_cut(ta, hb, tail, head, max_bone_deg=max_bone_deg)
    _, ci, cj, m, dh = best
    return ci, cj, m, dh
