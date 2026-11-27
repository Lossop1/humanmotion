"""可控性执行：朝向、地形、速度的纯几何实现。

单独成模块的原因：装配层 assemble_params 会导入 fit/motius（依赖 GPU 环境），
而这三件事是纯 numpy 运算，应该能在任何机器上单测。
"""
import math

import numpy as np


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _disp_dir(J, min_m=0.05):
    """这段的水平位移方向（度，世界系：atan2(x, z)）与距离。位移太小返回 (None, 距离)。"""
    d = np.asarray(J[-1, 0] - J[0, 0], dtype=np.float64)
    h = np.array([d[0], 0.0, d[2]])
    n = float(np.linalg.norm(h))
    if n < min_m:
        return None, n
    return math.degrees(math.atan2(float(h[0]), float(h[2]))), n


def _rot_y(J, ang_deg):
    """绕竖直轴把整段旋转 ang_deg（绕该段起始骨盆位置）。"""
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    piv = np.asarray(J[0, 0], dtype=np.float64).copy()
    out = np.array(J, dtype=np.float64, copy=True)
    for i in range(len(out)):
        for k in range(22):
            v = out[i, k] - piv
            out[i, k] = piv + np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])
    return out


DIR_ANGLE = {"forward": 0.0, "left": 90.0, "right": -90.0, "backward": 180.0}


def junction_frames(gap_deg, rate_deg_per_frame, cap=120, ratio_max=1.4, floor=2):
    """过渡帧数：慢到不超两侧速率的 ratio_max 倍，快也别超过 cap 帧。

    反例（轮25 S20）：姿态差 120.6°、两侧 0.6–1.0 度/帧，cap=60 时过渡被迫
    2.01 度/帧（速率比 2.01 不合格）。cap=120 时 n=ceil(120.6/(1.4*1.0))=87，
    速率比降到 1.39。
    """
    rate = max(float(rate_deg_per_frame), 1e-6)
    n = max(floor, int(round(float(gap_deg) / rate)))
    need = int(__import__("math").ceil(float(gap_deg) / (ratio_max * rate)))
    return int(min(int(cap), max(n, need)))


def body_travel_deg(J):
    """身体朝向与行进方向的夹角（度）：侧移约 90，直走约 0–15。"""
    J = np.asarray(J, dtype=np.float64)
    hip = J[0, 2] - J[0, 1]
    body = np.array([hip[2], 0.0, -hip[0]])
    disp = (J[-1, 0] - J[0, 0])[[0, 2]]
    nb, nd = float(np.linalg.norm(body)), float(np.linalg.norm(disp))
    if nb < 1e-9 or nd < 1e-9:
        return 0.0
    b = math.degrees(math.atan2(float(body[0]), float(body[2])))
    t = math.degrees(math.atan2(float(disp[0]), float(disp[1])))
    return float(abs((t - b + 180.0) % 360.0 - 180.0))


def enforce_facing(J, direction, global_forward_deg=0.0, prompt=""):
    """把这一段的世界行进方向转到请求的方向上，返回 (J2, info)。

    为什么整体旋转有效：段内姿态与运动的关系不变（侧移还是侧移、倒退还是倒退），
    改的只是这个世界里的行进方向；同向对齐由 align_pair 负责，两者不冲突。
    """
    name = str(direction or "").strip().lower()
    if name not in DIR_ANGLE:
        return J, None
    if name in ("left", "right"):
        # 左右能不能整体旋转，取决于这段本身是不是「侧着走」：
        # 实测 "walks to the right" 身体与行进夹角只有 4–12 度（等于直走），旋转是伪造；
        # "sidesteps to the right" 是 60–90 度，旋转只改世界方向、保留侧移姿态，是合法的。
        phi0, dist0 = _disp_dir(J)
        if phi0 is None:
            return J, {"requested": name, "skipped": "位移过小（%.2f 米）" % dist0}
        _ang = body_travel_deg(J)
        if _ang < 45.0:
            return J, {"requested": name,
                       "skipped": "身体与行进只差 %.0f 度，不是侧移（旋转会变成直走）" % _ang}
        target0 = _wrap_deg(global_forward_deg + DIR_ANGLE[name])
        delta0 = _wrap_deg(target0 - phi0)
        J2 = _rot_y(J, delta0) if abs(delta0) > 1e-6 else J
        return J2, {"requested": name, "before_deg": round(_wrap_deg(phi0 - global_forward_deg), 1),
                    "target_deg": round(DIR_ANGLE[name], 1), "applied_deg": round(delta0, 1),
                    "distance_m": round(dist0, 2), "body_travel_deg": round(_ang, 1)}
    phi, dist = _disp_dir(J)
    if phi is None:
        return J, {"requested": name, "skipped": "位移过小（%.2f 米）" % dist}
    target = _wrap_deg(global_forward_deg + DIR_ANGLE[name])
    delta = _wrap_deg(target - phi)
    # 大角度换向（>20 度）绕支撑脚转，避免「站在转盘上」整体旋转；
    # 小角度修正仍绕自身，避免把正常走路的步态掰歪。
    _is_turn = "turn" in str(prompt or "").lower()
    if _is_turn and abs(delta) > 20.0:
        try:
            from hm import contact as _contact
            J2, _tinfo = _contact.turn_with_feet(J, delta)
        except Exception:
            J2 = _rot_y(J, delta)
    else:
        J2 = _rot_y(J, delta) if abs(delta) > 1e-6 else J
    return J2, {"requested": name,
                "before_deg": round(_wrap_deg(phi - global_forward_deg), 1),
                "target_deg": round(DIR_ANGLE[name], 1),
                "applied_deg": round(delta, 1),
                "distance_m": round(dist, 2)}


def apply_terrain(J, ground_y):
    """按指定地面高度整体抬升/下降：以最低关节点对齐，返回 (J2, 修正量)。"""
    if ground_y is None:
        return J, None
    lowest = float(np.min(J[:, :, 1]))
    dy = float(ground_y) - lowest
    if abs(dy) < 1e-6:
        return J, 0.0
    J2 = np.array(J, dtype=np.float64, copy=True)
    J2[:, :, 1] += dy
    return J2, round(dy, 4)


# 速度默认不改帧数：实测使用者把「快跑」看成「快放」（整段播快了），
# 那正是帧数伸缩的副作用。现在只通过提示词里的 slowly/quickly 影响生成；
# 需要时间伸缩时设 HM_SPEED_WARP=1。
SPEED_FACTOR = {"slow": 1.25, "normal": 1.0, "fast": 0.8}
_USE_WARP = __import__("os").environ.get("HM_SPEED_WARP", "0") == "1"


def _mps(J, fps=20.0):
    """平均水平速度（米/秒）：看整段的位移，不看瞬时抖动。"""
    if len(J) < 2:
        return 0.0
    d = np.asarray(J[-1, 0] - J[0, 0], dtype=np.float64)
    return float(np.hypot(d[0], d[2])) / (len(J) / fps)


def repair_spikes(params, min_deg=30.0, factor=2.5):
    """把「单帧突然扭一大下」的帧用前后帧插值替换。

    人工标定锚点：big_s08 第 263 帧根关节一帧转 113 度（前后帧 7–38 度），
    画面上就是一次可见的扭曲。修法：该帧的全部旋转用 t-1 与 t+1 的球面插值替代，
    平移线性插值；只动被判定为尖峰的帧，其余帧不变。
    返回 (params2, 修掉的帧号列表)。
    """
    from scipy.spatial.transform import Rotation as R
    P = np.array(params, dtype=np.float64, copy=True)
    n = len(P)
    if n < 4 or P.shape[1] < 85:
        return P, []
    rv = P[:, 0:72].reshape(n, 24, 3)
    a = R.from_rotvec(rv[1:].reshape(-1, 3))
    b = R.from_rotvec(rv[:-1].reshape(-1, 3))
    per = np.degrees((a * b.inv()).magnitude().reshape(n - 1, 24)).max(axis=1)  # 帧 i→i+1
    fixed = []
    for t in range(1, n - 1):
        prev = max(per[t - 2] if t >= 2 else 0.0, 1e-6)
        nxt = per[t + 1] if t + 1 < len(per) else prev
        local = max(prev, nxt, 1e-6)
        if per[t - 1] > min_deg and per[t - 1] > factor * local:
            ra = R.from_rotvec(P[t - 1, 0:72].reshape(24, 3))
            rb = R.from_rotvec(P[t + 1, 0:72].reshape(24, 3))
            # 球面中点：先求相对旋转，再取一半，左乘回 ra。
            # 之前写成 rb.inv()**0.5 是错的（把相对旋转算反了，修完反而更大）。
            rel = rb * ra.inv()
            mid = (ra * R.from_rotvec(rel.as_rotvec() * 0.5)).as_rotvec().ravel()
            P[t, 0:72] = mid
            P[t, 82:85] = 0.5 * (P[t - 1, 82:85] + P[t + 1, 82:85])
            fixed.append(t)
    return P, fixed


def apply_speed(J, speed):
    """按 slow/normal/fast 做时间伸缩，返回 (J2, info)。

    做法：帧数按因子重采样（slow 多帧、fast 少帧），关节位置线性插值；
    为什么不在生成时改：后端没有速度接口，改帧数是唯一可控且可测的手段。
    """
    name = str(speed or "").strip().lower()
    factor = SPEED_FACTOR.get(name)
    if not _USE_WARP or factor in (None, 1.0) or len(J) < 8:
        return J, None
    before = _mps(J)
    n_new = max(8, int(round(len(J) * factor)))
    src = np.linspace(0.0, len(J) - 1.0, n_new)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(J) - 1)
    w = (src - i0)[:, None, None]
    J2 = J[i0] * (1.0 - w) + J[i1] * w
    after = _mps(J2)
    info = {"speed": name, "factor": factor, "frames_before": int(len(J)),
            "frames_after": int(n_new),
            "mps_before": round(before, 2), "mps_after": round(after, 2),
            "err_pct": round(100.0 * (after - before / factor) / max(before / factor, 1e-6), 1)}
    return J2, info


def turn_wobble(params):
    """转身的回摆量（度）：用参数里的根旋转算，不受脊柱朝向噪声影响。

    实测「原地转身两圈」：净转 37 度、回退 56 度 → 画面上像倒放。
    返回 (净转角, 最大回退角)。
    """
    from scipy.spatial.transform import Rotation as R
    P = np.asarray(params, dtype=np.float64)
    if len(P) < 3:
        return 0.0, 0.0
    rv = P[:, 0:3]
    r = R.from_rotvec(rv.reshape(-1, 3))
    # 取绕竖直轴的分量：把旋转作用到 +z 上，看方位角
    fwd = r.apply(np.array([0.0, 0.0, 1.0]))
    yaw = np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]))
    yaw = np.degrees(np.unwrap(np.radians(yaw)))
    net = float(yaw[-1] - yaw[0])
    back = float(np.max(np.maximum.accumulate(yaw) - yaw))
    return round(abs(net), 1), round(back, 1)


def params_disp(params):
    """参数空间里的水平位移方向（度）与距离。只看根平移。"""
    P = np.asarray(params, dtype=np.float64)
    if len(P) < 2:
        return None, 0.0
    d = P[-1, 82:85] - P[0, 82:85]
    h = np.array([d[0], 0.0, d[2]])
    n = float(np.linalg.norm(h))
    if n < 0.05:
        return None, n
    return float(np.degrees(np.arctan2(h[0], h[2]))), n


def rotate_params(params, ang_deg, pivot=None):
    """把整段参数绕竖直轴转 ang_deg（根朝向与根平移一起转）。

    为什么需要：朝向在执行层是「拟合前转关节」，但拟合（opt）会重新解根轨迹，
    实测「向右走」拟合前 −90°、拟合后变成 0°（直走）——朝向被执行层改了回来。
    所以在拟合后按参数再校一次。
    """
    from scipy.spatial.transform import Rotation as R
    P = np.array(params, dtype=np.float64, copy=True)
    a = np.radians(float(ang_deg))
    c, sn = np.cos(a), np.sin(a)
    piv = np.array(P[0, 82:85], dtype=np.float64) if pivot is None else np.asarray(pivot, dtype=np.float64)
    for i in range(len(P)):
        v = P[i, 82:85] - piv
        P[i, 82:85] = piv + np.array([c * v[0] + sn * v[2], v[1], -sn * v[0] + c * v[2]])
    rot = R.from_rotvec(P[:, 0:3])
    new = R.from_euler("y", float(ang_deg), degrees=True) * rot
    P[:, 0:3] = new.as_rotvec()
    return P


def apply_in_place(J, prompt):
    """提示词写了 in place 时，把整段的水平漂移线性扣掉。

    实测：想要「原地高抬腿」只能写 "marches in place with high knees"
    （膝角 111 度，但模型仍往前走 2.20 米）；写 "marches in place" 则膝角只有 56 度，
    跟走路没差别。所以取前者并在这里把漂移扣掉——这句话本来就要求别往前走。
    返回 (J2, 扣掉的漂移米数)。
    """
    if "in place" not in str(prompt or "").lower():
        return J, 0.0
    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    if T < 3:
        return J, 0.0
    d = J[-1, 0] - J[0, 0]
    drift = np.array([d[0], 0.0, d[2]])
    n = float(np.linalg.norm(drift))
    if n < 0.05:
        return J, 0.0
    u = np.linspace(0.0, 1.0, T)[:, None, None]
    out = J - u * drift[None, None, :]
    return out, round(n, 2)
