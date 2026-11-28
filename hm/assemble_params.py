"""A16 的参数空间版本：逐段拟合，在 SMPL 参数空间做过渡。

与关节空间版本的区别：
    关节空间  对齐 → 关节方向插值 → 整条拟合（过渡不受骨骼约束，会内翻）
    参数空间  对齐 → 逐段拟合 → 关节旋转做球面插值 → 直接输出参数（不会内翻）

代价：每段拟合一次，比整条拟合多花约 (段数-1)×40 秒。
"""
import json
import sys
from pathlib import Path

import math

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hm import control as CTL
from hm import fit as FM, finish_job as FJ, job as jobmod

FM._patch_paths()

ROT_DIM = 72          # 0:3 根 + 3:72 身体
BETAS = (72, 82)
TRANSL = (82, 85)


TRANSITION_MAX_RATE = 9.0   # 过渡允许的最大骨向变化速度（度/帧）


def joint_speed(J, tail=True, w=3):
    """相邻帧之间「骨向变化」的最大值（度/帧），结尾或开头若干帧的均值。"""
    from hm.job import BONES as _B
    idx = range(max(0, len(J) - w), len(J)) if tail else range(0, min(w, len(J)))
    vals = []
    for i in idx:
        if i + 1 >= len(J):
            break
        a = np.array([J[i + 1, c] - J[i + 1, p2] for c, p2 in _B])
        b = np.array([J[i, c] - J[i, p2] for c, p2 in _B])
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        cos = np.clip((a * b).sum(axis=1), -1, 1)
        vals.append(float(np.degrees(np.arccos(cos)).max()))
    return float(np.mean(vals)) if vals else 1.0


def hermite_s(n, ratio_in, ratio_out):
    """返回 n 个递增的 s 值（0–1），两端斜率 = 角速度比。"""
    m0 = float(np.clip(ratio_in, 0.0, 3.0))
    m1 = float(np.clip(ratio_out, 0.0, 3.0))
    us = np.linspace(0.0, 1.0, n + 2)[1:-1]
    h00 = 2 * us ** 3 - 3 * us ** 2 + 1
    h10 = us ** 3 - 2 * us ** 2 + us
    h01 = -2 * us ** 3 + 3 * us ** 2
    h11 = us ** 3 - us ** 2
    s = h00 * 0.0 + h10 * m0 + h01 * 1.0 + h11 * m1
    s = np.clip(s, 0.0, 1.0)
    return np.maximum.accumulate(s)


def slerp_param_transition(Pa, Pb, n, s_curve=None):
    """逐关节四元数插值 + 平移线性；体型冻结为 Pa 的值。"""
    a = Pa[0:ROT_DIM]
    b = Pb[0:ROT_DIM]
    out = np.zeros((n, 85))
    ts = np.linspace(0.0, 1.0, n + 2)[1:-1] if s_curve is None else s_curve
    ra = R.from_rotvec(a.reshape(-1, 3))
    rb = R.from_rotvec(b.reshape(-1, 3))
    qa = ra.as_quat()
    qb = rb.as_quat()
    flip = (qa * qb).sum(axis=1) < 0
    qb[flip] *= -1.0
    for i, t in enumerate(ts):
        q = qa * (1 - t) + qb * t
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        v = R.from_quat(q).as_rotvec().ravel()
        out[i, 0:ROT_DIM] = v
        out[i, BETAS[0]:BETAS[1]] = Pa[BETAS[0]:BETAS[1]]
        out[i, TRANSL[0]:TRANSL[1]] = Pa[TRANSL[0]:TRANSL[1]] * (1 - t) + \
                                      Pb[TRANSL[0]:TRANSL[1]] * t
    return out


def realized_speed(params, idx):
    """参数序列中第 idx→idx+1 帧的最大关节转动角（度）。"""
    from scipy.spatial.transform import Rotation as _R
    a = _R.from_rotvec(params[idx, 0:72].reshape(24, 3))
    b = _R.from_rotvec(params[idx + 1, 0:72].reshape(24, 3))
    rel = b * a.inv()
    return float(np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1)).max())


def joints_of_params(P):
    """参数 → 22 关节位置（用于装配后的几何检查）。"""
    import torch
    m = FM._load_smpl("cuda")
    T = torch.from_numpy(np.asarray(P, dtype=np.float32)).to(next(m.parameters()).device)
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True)
    return o.joints.detach().cpu().numpy()[:, :22, :], \
           o.vertices.detach().cpu().numpy()


def build(job, sb, verbose=True):
    """返回 (params (T,85), bounds, verts)"""
    # 单一来源：A12 选中的候选就是最终候选。
    # 之前这里又跑一次 DP，把 A12 的选择换掉了（seg_001 cand_04→cand_01、seg_002 cand_05→cand_03），
    # 记录与成片不符，质量判断也就失效了。
    _fsb = [(sg, int(str(sg["selected"]).split("_")[-1]))
            for sg in sb["segments"] if sg.get("selected")]
    chosen = (_fsb if len(_fsb) == len(sb["segments"])
              else FJ.choose_combination(job, sb, verbose=verbose))

    pieces, seg_params, bounds = [], [], []
    prev = None
    _terrain = sb.get("terrain") or {}
    _profile = {int(p.get("at_segment", 0)): float(p.get("y", 0.0))
                for p in (_terrain.get("profile") or []) if isinstance(p, dict)}
    # 世界前方固定为 0（人面朝 +z）：这样每段的目标行进方向是绝对角，
    # 不会因为「第一段自然方向」而自我满足（实测「向右走」曾因此只拿到 60°）。
    _g_forward = 0.0
    for _idx, (seg, n) in enumerate(chosen, start=1):
        J = np.load(job.dir / "segments" / seg["segment_id"] /
                    ("candidate_%d" % n) / "joints22.npy")
        _k = int(seg.get("trim_tail") or 0)
        if 0 < _k < len(J) - 8:
            J = J[: len(J) - _k]
        if prev is None:
            placed, dyaw = J, 0.0
        else:
            placed, dyaw = FJ.align_pair(prev, J)

        placed, _face = CTL.enforce_facing(placed, seg.get("facing") or seg.get("direction"),
                                       _g_forward, prompt=seg.get("text_gen_final") or "")
        # 地形执行：把本段整体对齐到规划给出的地面高度
        # 地形按「原段号」取高度：合并会让段序号错位，不能用循环下标
        try:
            _sid = int(str(seg["segment_id"]).split("_")[-1])
        except (ValueError, KeyError):
            _sid = _idx
        _ground = _profile.get(_sid, _terrain.get("start_y"))
        placed, _dy = CTL.apply_terrain(placed, _ground)
        # 脚部接触约束：把支撑脚钉住（实测候选滑动 4–5 mm/帧、位移只少 2%）
        # 旋转在前、钉脚在后 → 转身变成「绕着支撑脚转」，而不是整体刚性旋转
        try:
            from hm import contact as _contact
            placed, _pin = _contact.pin_contacts(placed)
            if _pin.get("frames"):
                job.log("钉脚：%s 支撑 %d 帧，滑动 %.0f→%.0f mm/帧，扣掉漂移 %.2f 米"
                        % (seg["segment_id"], _pin["frames"], _pin.get("slide_before_mm", 0),
                           _pin.get("slide_after_mm", 0), _pin.get("fixed_m", 0)))
        except Exception as _e:
            job.log("钉脚跳过：%s" % str(_e)[:60])
        # 原地约束：提示词含 in place 时扣掉水平漂移（实测高抬腿写法会往前走 2.2 米）
        placed, _drift = CTL.apply_in_place(placed, seg.get("text_gen_final") or "")
        if _drift:
            job.log("原地约束：%s 扣掉水平漂移 %.2f 米" % (seg["segment_id"], _drift))
        # 速度执行：默认不伸缩帧数（HM_SPEED_WARP=1 才开）
        placed, _spd = CTL.apply_speed(placed, seg.get("speed"))
        if verbose:
            print("    拟合 %s ..." % seg["segment_id"], flush=True)
        _m = __import__("os").environ.get("HM_FIT_METHOD", "opt")
        _ref = int(__import__("os").environ.get("HM_FIT_REFINE", "0"))
        _it = int(__import__("os").environ.get("HM_FIT_ITERS", "8"))
        from hm import cache as _cache
        _ck = _cache.key(("fit", np.asarray(placed).tobytes(), _m, _it, _ref))
        _cf = _cache.get_file("fit", _ck, "params.npy") if _cache.enabled() else None
        if _cf is not None:
            P = np.load(_cf)
        else:
            _, P = FM.fit(placed, method=_m, refine=_ref, iters=_it)
            if _cache.enabled():
                import tempfile as _tf
                _tmp = Path(_tf.mkdtemp()) / "p.npy"
                np.save(_tmp, np.asarray(P))
                _cache.put_file("fit", _ck, "params.npy", _tmp)
        P = np.asarray(P, dtype=np.float64)
        # 拟合翻转兜底：优化法在个别帧会把肢体翻过去（候选关节是平滑的，拟合后单帧转动 118 度）。
        # 实测同句 HM_FIT_METHOD=ik 只有 22.5 度，所以「扭」的时候用解析 IK 重拟一遍取更平滑的。
        _rs = __import__("hm.quality_metrics", fromlist=["rot_spike"]).rot_spike(P)
        # 区分「拟合翻转」与「本来就是快速动作」：候选自身的骨骼突变作为参照。
        # 锚点：s08 候选 23 度、拟合后 118 度（5 倍）→ 翻转；s17 候选与拟合都 33 度左右 → 快速动作，不是缺陷。
        _qm = __import__("hm.quality_metrics", fromlist=["bone_turn_spike", "rot_spike"])
        _cand_spike = _qm.bone_turn_spike(placed)
        _ratio = _rs["max_deg"] / max(_cand_spike, 1e-6)
        if _m != "ik" and _rs["max_deg"] > 30.0 and _ratio > 1.5:
            _, _P_ik = FM.fit(placed, method="ik", refine=0, iters=4)
            _rs_ik = __import__("hm.quality_metrics", fromlist=["rot_spike"]).rot_spike(_P_ik)
            if _rs_ik["max_deg"] < _rs["max_deg"]:
                job.log("拟合兜底：%s 单帧转动 %.1f 度 → 改用解析 IK 后 %.1f 度"
                        % (seg["segment_id"], _rs["max_deg"], _rs_ik["max_deg"]))
                P = np.asarray(_P_ik, dtype=np.float64)
                _rs = _rs_ik
        pieces.append(placed)
        seg_params.append(P)
        # 拟合后补一次朝向校正：opt 拟合会重解根轨迹，实测把「向右走」解成了直走
        _req = str(seg.get("facing") or seg.get("direction") or "").strip().lower()
        if __import__("os").environ.get("HM_DEBUG_FACE") == "1":
            _dbg = CTL.params_disp(P)
            job.log("调试朝向：req=%r g_forward=%r 拟合后方向=%r 距离=%.2f"
                    % (_req, _g_forward, None if _dbg[0] is None else round(_dbg[0], 1), _dbg[1]))
        if _req in CTL.DIR_ANGLE and _g_forward is not None:
            _phi_p, _d_p = CTL.params_disp(P)
            if _phi_p is not None:
                _target_p = CTL._wrap_deg(_g_forward + CTL.DIR_ANGLE[_req])
                _delta_p = CTL._wrap_deg(_target_p - _phi_p)
                if abs(_delta_p) > 10.0:
                    P = CTL.rotate_params(P, _delta_p)
                    job.log("拟合后朝向校正：%s 请求 %s，拟合后偏差 %.0f 度，已转回"
                            % (seg["segment_id"], _req, _delta_p))
                    _face = dict(_face or {})
                    _face["post_fit_deg"] = round(_delta_p, 1)

        _turn_net, _turn_back = CTL.turn_wobble(P)
        _is_turn = "turn" in (seg.get("text_gen_final") or "").lower()
        if _is_turn and _turn_back > 30.0:
            job.log("转身回摆：%s 净转 %.0f 度、回退 %.0f 度（>30 度像倒放，标记可疑）"
                    % (seg["segment_id"], _turn_net, _turn_back))
        bounds.append({"segment_id": seg["segment_id"], "candidate": "cand_%02d" % n,
                       "turn_net_deg": _turn_net, "turn_back_deg": _turn_back,
                       "turn_wobble": bool(_is_turn and _turn_back > 30.0),
                       "frames": int(len(placed)), "align_yaw_deg": round(dyaw, 2),
                       "facing": _face, "terrain_dy_m": _dy, "ground_y": _ground,
                       "speed": _spd, "rot_spike_deg": _rs["max_deg"],
                       "cand_spike_deg": round(_cand_spike, 1), "spike_ratio": round(_ratio, 2),
                       "stage": "retimed"})
        prev = placed
        seg["selected"] = "cand_%02d" % n

    # 体型统一：全部冻结成第一段的值，避免过渡中骨长变化
    betas0 = seg_params[0][0, BETAS[0]:BETAS[1]]
    for P in seg_params:
        P[:, BETAS[0]:BETAS[1]] = betas0

    merged, junctions = [seg_params[0]], []
    for i in range(1, len(seg_params)):
        A_joint, B_joint = pieces[i - 1][-1], pieces[i][0]
        gap = FJ.pose_gap(A_joint, B_joint)
        rate = max(FJ.bone_rate(pieces[i - 1], tail=True),
                   FJ.bone_rate(pieces[i], tail=False), 1e-6)
        # 最小过渡 6 帧（0.3 秒）：实测使用者判「2 秒左右走转挥手过渡太快 不自然」，
        # 那一段交界只有 7 帧且姿态差 61 度，看着像瞬间切换。
        # 交界长度上限 12 帧：实测 19–26 帧的交界反而缝处更差（12.7–20.5），
        # 因为长交界里脚要长时间「悬着找位置」。上限 12（≈0.6 秒）。
        n = int(max(6, min(12, round(gap / rate))))
        # 交界帧数还要看两端根速度：从 1.2 米/秒一帧刹到 0 = 2.4g，物理上做不到。
        # 按减速度 ≤0.25 米/秒/帧（≈5 m/s²）反推所需帧数。
        try:
            _pa = pieces[_idx - 2] if _idx >= 2 else pieces[0]
            _pb = pieces[_idx - 1]
            _va = float(np.linalg.norm(_pa[-1, 0] - _pa[-2, 0]) * 20) if len(_pa) > 1 else 0.0
            _vb = float(np.linalg.norm(_pb[1, 0] - _pb[0, 0]) * 20) if len(_pb) > 1 else 0.0
            # 验收要求缝处加速度 ≤5 米/秒² → 每帧速度变化 ≤5×0.05=0.25 米/秒，
            # 但二阶差分对称，实际需要两倍长度才稳，所以乘 2 作为安全系数
            _v_req = int(np.ceil(2.0 * abs(_va - _vb) / 0.25))
            _v_need = _v_req
            if _v_req > n:
                job.log("交界加长：两端根速 %.2f→%.2f 米每秒，需要 %d 帧（原 %d）"
                        % (_va, _vb, _v_req, n))
                n = int(min(60, _v_req))
        except Exception:
            pass
        _v_need = locals().get("_v_need", 0)
        n = int(min(n, max(2, FJ.DEFAULT_TRANSITION)) if gap <= FJ.EXTREME_GAP_DEG else n)
        # 过渡本身也有速度上限：姿态差再大，也不能一两帧甩完
        # 过渡速度上限不应用固定值，而应与两侧运动速度对比：
        # 实测 S03 交界 86 度、两侧 15.9 度/帧，固定 9 度/帧地板把过渡压到 8.6（比 0.54），
        # 视觉上就是两侧很快、中间慢下来。改为允许到两侧速率的 1.1 倍。
        # 速率比是判据：过渡速度不应超过两侧的 1.4 倍（否则比两侧快，看着像闪一下）。
        # 实测轮14 出现 4.07 倍（13 度用 2 帧走完），就是没有这条上限。
        # 帧数：交给 control.junction_frames（含速率比上限与 120 帧总上限，可单测）
        _cap_n = int(__import__("os").environ.get("HM_TRANS_CAP", "120"))
        n = CTL.junction_frames(gap, rate, cap=_cap_n)
        # junction_frames 会覆盖前面的计算结果：日志说「需要 10 帧」而实际仍 6 帧，
        # 就是因为这一行在后面把速度需求抹掉了。这里统一取大，上限 16 帧（0.8 秒）。
        # 二阶差分峰值约为平均减速度的两倍 → 要通过「缝 ≤5 米/秒²」，
        # 交界至少需要 8·|Δv| 帧（实测 1.07 米/秒 需要 ≥9 帧，8 帧时缝 7.7）。
        if _v_need and _v_need > n:
            job.log("交界按下端根速取大：%d → %d 帧" % (n, min(_v_need, 18)))
            n = int(min(_v_need, 18))
        w_in = joint_speed(pieces[i - 1], tail=True)
        w_out = joint_speed(pieces[i], tail=False)
        nom = max(gap / max(n, 1), 1e-6)
        m0, m1 = w_in / nom, w_out / nom
        curve = hermite_s(n, m0, m1)
        # 闭环修正一次：按实际实现出来的首末速度调整斜率
        tr_try = slerp_param_transition(seg_params[i - 1][-1], seg_params[i][0], n, curve)
        real_in = realized_speed(tr_try, 0)
        real_out = realized_speed(tr_try, n - 2) if n >= 2 else real_in
        if real_in > 1e-6:
            m0 *= float(np.clip(w_in / real_in, 0.3, 3.0))
        if real_out > 1e-6:
            m1 *= float(np.clip(w_out / real_out, 0.3, 3.0))
        curve = hermite_s(n, m0, m1)
        if verbose and i == 1:
            print("      曲线检查：w_in %.2f, w_out %.2f, 平均 %.2f 度/帧 → 斜率 %.2f / %.2f"
                  % (w_in, w_out, nom, m0, m1), flush=True)
            print("      s 前三个值 %s（等间隔应为 0.06 0.13 0.19）"
                  % np.round(curve[:3], 3), flush=True)
        tr = slerp_param_transition(seg_params[i - 1][-1], seg_params[i][0], n, curve)
        junctions.append({"between": [chosen[i - 1][0]["segment_id"], chosen[i][0]["segment_id"]],
                          "pose_gap_deg": round(gap, 1),
                          "boundary_rate_deg_per_frame": round(rate, 2),
                          "transition_frames": n,
                          "extreme": bool(gap > FJ.EXTREME_GAP_DEG),
                          "method": "param_space_slerp",
                          "w_in_deg_per_frame": round(w_in, 2),
                          "w_out_deg_per_frame": round(w_out, 2),
                          "curve": "hermite(slope=w/mean)"})
        merged.append(tr)
        merged.append(seg_params[i])

    params = np.concatenate(merged, axis=0)
    total = sum(b["frames"] for b in bounds) + sum(j["transition_frames"] for j in junctions)
    if total != len(params):
        raise SystemExit("帧数核对失败：%d vs %d" % (total, len(params)))

    J, V = joints_of_params(params)

    # 网格层钉脚已撤回：独立复核发现它压平的正是它自己量的量（自证），
    # 且并集质心在换脚帧会给出 3 厘米/帧的横向猛拽（实测换脚帧根速 1.15–1.46 m/s）。
    # 详见 docs/证据度量.md §49。保留 contact.pin_mesh 供离线实验，不在主链调用。

    # 交界：关节空间生成 + 解析 IK 回填 + 根位移线性化（一次做完，不再层层叠加）
    try:
        from hm import contact as _C3
        _Jj, _Vj = joints_of_params(params)
        _ranges = []
        _cur = 0
        for _i, _b in enumerate(bounds):
            _cur += _b["frames"]
            if _i < len(junctions):
                _n = junctions[_i]["transition_frames"]
                _ranges.append((_cur, _cur + _n))
                _cur += _n
        for (_a, _b2) in _ranges:
            if _a < 1 or _b2 >= len(params):
                continue
            # 减速斜坡：交界前 k 帧把根速从 v0 线性降到 v_next，
            # 这样制动发生在走路段末尾（有位移可用），而不是把 1.07 米/秒
            # 硬塞进只有 0.04 米位移的交界里（实测交界第一帧速度 0.77→0.33）。
            try:
                _dt_ = 0.05
                _vp = np.linalg.norm(_Jj[_a - 1, 0, [0, 2]] - _Jj[_a - 2, 0, [0, 2]]) / _dt_
                _vn = np.linalg.norm(_Jj[_b2, 0, [0, 2]] - _Jj[_b2 - 1, 0, [0, 2]]) / _dt_
                # 每帧只降 0.15 米/秒（≈3 米/秒²），把制动摊到 8–12 帧里：
                # 之前 0.25/帧 只有 4–5 帧，二阶差分仍然把交界入口算成 12–27 米/秒²。
                _k = int(min(max(0, np.ceil(abs(_vp - _vn) / 0.15)), 12, max(0, _a - 2)))
                if _k >= 2:
                    _p_end = params[_a - 1, 82:85].copy()
                    _dirv = (params[_a - 1, 82:85] - params[_a - 2, 82:85])
                    _nv = np.linalg.norm(_dirv[[0, 2]])
                    if _nv > 1e-6:
                        _u = _dirv[[0, 2]] / _nv
                        # 从 a-1 往前积分，速度按 k 段从 vp 降到 vn，
                        # 并且**以 vn 收尾**（旧版把 a-1 留着不动，交界入口仍是 1.32 米/秒）
                        _pos = _p_end.copy()
                        for _j in range(1, _k):
                            # 越靠近交界、速度越低：j=1 是 a-2（最靠近交界）→ 应接近 vn。
                            # 之前写成 1 - j/k 是反的：结果交界前 3 帧在「加速」
                            # （实测步长 0.0289→0.0412→0.0535），交界被迫过冲再回拉。
                            _frac = _j / float(_k)
                            _v = _vn + (_vp - _vn) * max(0.0, _frac)
                            _pos = _pos.copy()
                            _pos[[0, 2]] -= _u * (_v * _dt_)
                            params[_a - 1 - _j, 82:85] = _pos
                        job.log("减速斜坡：交界前 %d 帧把根速 %.2f→%.2f 米每秒" % (_k, _vp, _vn))
            except Exception as _e7:
                job.log("减速斜坡跳过：%s" % str(_e7)[:200])

            # 先定交界内每帧的根位置（三次 Hermite，两端速度匹配）
            _m = _b2 - _a
            _r0 = _Jj[_a - 1, 0, :].copy()
            _r1 = _Jj[_b2, 0, :].copy()
            # 速度必须从「已经做过减速斜坡的 params」里取，
            # 否则 Hermite 会用斜坡前的 0.91 米/秒起步，与斜坡后的实际速度不一致（缝 9–13）
            _jj, _ = joints_of_params(params)
            _pp0 = _jj[_a - 2, 0, :] if _a >= 2 else _r0
            _pp1 = _jj[_b2 + 1, 0, :] if _b2 + 1 < len(_jj) else _r1
            _r0 = _jj[_a - 1, 0, :].copy()
            _r1 = _jj[_b2, 0, :].copy()
            # 不许倒着过交界：实测后段起点落在前段终点「后方」1.6 厘米
            # （r0 z=2.942、r1 z=2.926），交界于是倒走再回拉，二阶差分记成 14.7 米/秒²。
            # 这里把后段整体前移到与前段终点对齐，交界就不再倒退。
            try:
                _travel = _r0 - _jj[max(0, _a - 6), 0, :]
                _dv = _r1 - _r0
                _proj = float(np.dot(_travel[[0, 2]], _dv[[0, 2]]))
                if _proj < 0:
                    _shift = _r0 - _r1
                    params[_b2:, 82] += _shift[0]
                    params[_b2:, 84] += _shift[2]
                    _r1 = _r0.copy()
                    _jj, _ = joints_of_params(params)
                    job.log("交界向前对齐：后段整体前移 (%.3f, %.3f) 米，避免倒走过交界"
                            % (_shift[0], _shift[2]))
                    # 前移只保住后段内部速度，接缝处仍有台阶（实测出口 1.9→30.6）。
                    # 这里重写后段头 3 帧的根步长，从「交界末帧步长」线性过渡到它自己的步长。
                    try:
                        # 斜坡长度按速度差定：每帧最多改变 0.0125 米（≈5 米/秒²），
                        # 否则出口仍是一次台阶（实测 26–33 米/秒²）
                        _vp0 = params[_b2 - 1, 82:85] - params[_b2 - 2, 82:85]
                        _vn0 = params[min(len(params) - 1, _b2 + 4), 82:85] - params[min(len(params) - 1, _b2 + 3), 82:85]
                        _kneed = int(np.ceil(float(np.linalg.norm(_vn0 - _vp0)) / 0.0125)) if np.linalg.norm(_vn0 - _vp0) > 0 else 3
                        _kk = min(max(3, _kneed), 10, len(params) - _b2 - 2)
                        if _kk >= 1 and _b2 >= 2:
                            _vlast = params[_b2 - 1, 82:85] - params[_b2 - 2, 82:85]
                            _steps = [params[_b2 + j, 82:85] - params[_b2 + j - 1, 82:85]
                                      for j in range(_kk + 1)]
                            _pos = params[_b2 - 1, 82:85].copy()
                            for j in range(_kk):
                                _w = (j + 1.0) / (_kk + 1.0)
                                _step = _vlast * (1 - _w) + _steps[j] * _w
                                _pos = _pos + _step
                                params[_b2 + j, 82:85] = _pos
                            job.log("出口速度斜坡：后段前 %d 帧根步长由交界末速过渡" % _kk)
                    except Exception as _e11:
                        job.log("出口速度斜坡跳过：%s" % str(_e11)[:200])
            except Exception as _e10:
                job.log("交界向前对齐跳过：%s" % str(_e10)[:200])
            # 出口速度斜坡：无条件执行（之前只在「后段需要前移」的分支里，
            # 不前移时就不生效，出口台阶因此一直在）
            try:
                # 出口斜坡实测更糟（走→挥手 入口 3.3→18.6、出口 12.8→27.5）→ 默认关，仅留实验开关
                _ke = _C3.ramp_exit(params, _b2) if _gate("HM_JEXIT", "0") else 0
                if _ke:
                    job.log("出口速度斜坡：后段前 %d 帧根步长由交界末速过渡" % _ke)
            except Exception as _e12:
                job.log("出口速度斜坡跳过：%s" % str(_e12)[:200])
            _vv0 = np.clip(_r0 - _pp0, -0.10, 0.10)
            _vv1 = np.clip(_pp1 - _r1, -0.10, 0.10)
            _dd = float(_m + 1)
            _roots = np.zeros((_m, 3))
            for _k in range(_m):
                _t = (_k + 1.0) / _dd
                _h00 = 2 * _t ** 3 - 3 * _t ** 2 + 1
                _h10 = _t ** 3 - 2 * _t ** 2 + _t
                _h01 = -2 * _t ** 3 + 3 * _t ** 2
                _h11 = _t ** 3 - _t ** 2
                _roots[_k] = (_h00 * _r0 + _h10 * _dd * _vv0
                              + _h01 * _r1 + _h11 * _dd * _vv1)
            # 出口速度收尾：实测交界最后一帧步长是后段首帧的 4 倍（0.027 对 0.007 米），
            # 二阶差分直接把它算成 8–10 米/秒²。这里把最后两帧改成「按后段速度倒退」，
            # 使进入后段的那一步速度就等于后段自己的速度。
            # 入口速度顺接（与出口对称）：交界头两帧按前段（已做减速斜坡的）速度前进，
            # 使交界第一步与前一帧速度一致，二阶差分不再把入口算成 10–17 米/秒²。
            _gate = lambda _n, _d: __import__("os").environ.get(_n, _d) == "1"
            # 四个覆写互相打架会把一条光滑 Hermite 撕成锯齿（实测交界内速度
            # 0.54→0.11→0.71→1.07，缝 22.9–35.6）。因此全部降级为实验开关，默认关：
            # 交界内的根运动只由一条 Hermite 决定。
            if _m >= 5 and _gate("HM_JENTRY", "0"):
                # 入口速度必须取「已做斜坡的 params」里的实际步长：
                # 之前用的 _vv0 来自斜坡前，交界第一帧速度反而弹回 0.80（实测 f50），
                # 于是 f49→f50 出现 14.6 米/秒²。
                _v_act = params[_a - 1, 82:85] - params[_a - 2, 82:85]
                for _k in range(0, 2):
                    _roots[_k] = _r0 + _v_act * float(_k + 1)
            if _m >= 3 and _gate("HM_JTAIL", "0"):
                for _k in range(1, 3):
                    _roots[_m - _k] = _r1 - _vv1 * float(_k)
            # 单调钳制：Hermite 在位移不足时会先冲过头再回拉（实测 f51 2.9481 → f52 2.9381），
            # 回拉那一步就是 14.6 米/秒² 的来源。这里按行进方向把根位置钳成单调。
            try:
                if not _gate("HM_JCLAMP", "0"):
                    raise RuntimeError("clamp off")
                _d = (_r1 - _r0)
                for _ax in (0, 2):
                    _sgn = np.sign(_d[_ax])
                    if abs(_d[_ax]) < 1e-6:
                        continue
                    _acc = _r0[_ax]
                    for _k in range(_m):
                        _want = _sgn * (_roots[_k, _ax] - _acc)
                        if _want < 0:
                            _roots[_k, _ax] = _acc
                        else:
                            _acc = _roots[_k, _ax]
            except Exception as _e9:
                job.log("单调钳制跳过：%s" % str(_e9)[:40])
            if __import__("os").environ.get("HM_TRACE_Z") == "1":
                job.log("ROOTS_Z r0=(%.3f,%.3f) r1=(%.3f,%.3f) " % (_r0[0], _r0[2], _r1[0], _r1[2]) + " ".join("%d:%.4f" % (_a + k, _roots[k, 2]) for k in range(_m)))
            _jl = float(__import__("os").environ.get("HM_JLIFT", "0.05"))
            _Jt, _tinfo = _C3.make_transition_joints(_Jj[_a - 1], _Jj[_b2], _m,
                                                    lift_m=_jl, roots=_roots)
            _, _Pt = FM.fit(_Jt, method="ik", refine=0, iters=2)
            params[_a:_b2, :85] = np.asarray(_Pt, dtype=np.float64)[:, :85]
            # 「拟合后还原预定根轨迹」实测更糟（缝 74.9）：根被按帧拽回，边界速度差更大 → 已撤回。
            # 真正的修法是不再做「分段拟合 + 拼接」，改成整段一次拟合（见文档 §53）。
            if __import__("os").environ.get("HM_DEBUG_JUNC") == "1":
                _ja, _ = joints_of_params(params[_a:_b2])
                _err = np.linalg.norm(_ja[:, 0, :] - _roots, axis=1)
                job.log("调试交界根误差：%s（毫米）"
                        % " ".join("%.0f" % (v * 1000) for v in _err))
            job.log("交界：%d 帧关节空间过渡（支撑脚 %d，抬脚 %.2f 米），根位移已线性化"
                    % (_m, _tinfo["support"], _tinfo["lift_m"]))
        J, V = joints_of_params(params)
    except Exception as _e:
        job.log("交界生成跳过：%s" % str(_e)[:70])

    # 通用接触清理：整段（不限交界）钉住接触脚，残差用两骨 IK 消掉。
    # 与交界补丁的区别：它不关心「哪里是交界」，只看「这只脚现在踩在地上」。
    # 接触清理默认关闭：实测它把缝处加速度从 7.3 抬到 24.8 米/秒²，
    # 而交界本身的关节空间过渡已经把 DSD 和滑移做达标（DSD=0、p90≤3）。
    # 保留为实验开关（HM_CLEANUP=1），不再默认启用。
    if __import__("os").environ.get("HM_CLEANUP", "0") == "1":
        try:
            from hm import contact as _C5
            # 只改根平移，不重拟姿态：重拟会把根位移逐帧独立解，缝处必然跳
            _Jc, _offc = _C5.cleanup_root_only(J)
            params[:, 82] += _offc[:, 0]
            params[:, 84] += _offc[:, 1]
            J, V = joints_of_params(params)
            job.log("接触清理（只调根）：最大修正 %.3f 米" % float(np.abs(_offc).max()))
        except Exception as _e5:
            job.log("接触清理跳过：%s" % str(_e5)[:60])

    # 逐段贴地（网格最低点）：实测「走到挥手」的挥手段整段悬空 4.3 厘米、
    # 交界悬空 3.6 厘米 —— 按关节对齐会漏掉脚掌比关节低的那几厘米。
    # 只在平坦地形（profile 全 0 或空）时逐段对齐；有台阶时保持原来的「只用第一段对齐」。
    try:
        _tp = {int(x.get("at_segment", 0)): float(x.get("y", 0.0))
               for x in ((sb.get("terrain") or {}).get("profile") or []) if isinstance(x, dict)}
        _flat = all(abs(v) < 1e-6 for v in _tp.values()) if _tp else True
        if _flat:
            _ranges = []
            _cur = 0
            for _i, _b in enumerate(bounds):
                _ranges.append((_cur, _cur + _b["frames"]))
                _cur += _b["frames"]
                if _i < len(junctions):
                    _ranges.append((_cur, _cur + junctions[_i]["transition_frames"]))
                    _cur += junctions[_i]["transition_frames"]
            _need = False
            _shifts = []
            for (_a, _bnd) in _ranges:
                _seg = V[_a:_bnd]
                _min = float(_seg[..., 1].min())
                if abs(_min) > 0.005:
                    _need = True
                _shifts.append((_a, _bnd, -_min))
            if _need:
                # 逐段贴地会在段/交界边界处留下「竖直台阶」（实测各段偏移最大差 4.1 厘米，
                # 二阶差分把它算成 14–16 米/秒²）。这里把偏移做成逐帧线性过渡：
                # 交界范围内由左段偏移线性过渡到右段偏移。
                _T = len(params)
                _shift_arr = np.zeros(_T)
                _segs_only = [(a0, b0, d0) for (a0, b0, d0) in _shifts]
                for (a0, b0, d0) in _segs_only:
                    _shift_arr[a0:b0] = d0
                # 交界处线性插值
                _prev_end, _prev_dy = None, None
                for (a0, b0, d0) in _segs_only:
                    if _prev_end is not None and a0 > _prev_end:
                        _m = a0 - _prev_end
                        for _k in range(_m):
                            _w = (_k + 1.0) / (_m + 1)
                            _shift_arr[_prev_end + _k] = _prev_dy * (1 - _w) + d0 * _w
                    _prev_end, _prev_dy = b0, d0
                params[:, 83] += _shift_arr
                J, V = joints_of_params(params)
                job.log("竖向偏移逐帧线性过渡：最大 %.3f 米（避免段间竖直台阶）"
                        % float(np.abs(_shift_arr).max()))
                J, V = joints_of_params(params)
                job.log("逐段贴地：按网格最低点把 %d 个片段/交界移到地面（最大位移 %.3f 米）"
                        % (len(_shifts), max(abs(d) for _a, _b, d in _shifts)))
    except Exception as _e:
        job.log("逐段贴地跳过：%s" % str(_e)[:60])

    # 贴地校正：按「网格最低点」而不是关节最低点对齐地面。
    # 实测：按关节对齐时皮肤比关节低 2–5 厘米，每千帧有 768 帧顶点在地面下。
    # 但只能用第一段对齐：若用全局最低点，会把后面台阶段整体拉低/抬高，
    # 破坏逐段地形（实测台阶句 0.15/0.30 被拉成 0.13/0.36）。
    _terrain0 = sb.get("terrain") or {}
    _prof0 = {int(p.get("at_segment", 0)): float(p.get("y", 0.0))
              for p in (_terrain0.get("profile") or []) if isinstance(p, dict)}
    _first_sid = 1
    try:
        _first_sid = int(str(bounds[0].get("segment_id", "seg_001")).split("_")[-1])
    except (ValueError, IndexError):
        pass
    _ground_first = _prof0.get(_first_sid, float(_terrain0.get("start_y") or 0.0))
    _n_first = int(bounds[0].get("frames") or 0) or len(V)
    _dy = _ground_first - float(np.min(V[:_n_first, :, 1]))
    if abs(_dy) > 1e-4:
        params = np.array(params, dtype=np.float64, copy=True)
        params[:, TRANSL[0] + 1] += _dy
        J, V = joints_of_params(params)

    L = jobmod.bone_lengths(J)
    rel = np.abs(L - L.mean(axis=0)) / np.maximum(L.mean(axis=0), 1e-9)
    tr_xy = J[:, 0, [0, 2]]
    step = np.linalg.norm(np.diff(tr_xy, axis=0), axis=1)
    metrics = {"stage": "retimed", "frames": int(len(params)),
               "bone_length_rel_spread_max": float(rel.max()),
               "foot_height_min": float(min(J[:, 10, 1].min(), J[:, 11, 1].min())),
               "root_path_length": float(step.sum()),
               "root_step_median": float(np.median(step)),
               "ground_align_dy_m": round(float(_dy), 4),
               "rot_spike": __import__("hm.quality_metrics", fromlist=["rot_spike"]).rot_spike(params)}
    bj = {"job_id": getattr(job, "job_id", "?"), "segments": bounds,
          "junctions": junctions, "total_frames": total, "metrics": metrics,
          "fps": 20, "assembly": "param_space"}
    if __import__("os").environ.get("HM_TRACE_Z") == "1":
        try:
            _jl = bj.get("junctions") or []
            _cur = 0
            for _i, _b in enumerate(bj["segments"]):
                _cur += _b["frames"]
                if _i < len(_jl):
                    _n = _jl[_i]["transition_frames"]
                    _lo, _hi = max(0, _cur - 3), min(len(params), _cur + _n + 2)
                    job.log("TRACE_Z 交界%d [%d-%d]：%s" % (
                        _i + 1, _cur, _cur + _n - 1,
                        " ".join("%d:%.4f" % (t, params[t, 84]) for t in range(_lo, _hi))))
                    _cur += _n
        except Exception as _e8:
            job.log("TRACE_Z 失败：%s" % str(_e8)[:60])
    return params, bj, V, J


def main():
    job_id = sys.argv[1]
    jobdir = ROOT / "outputs" / job_id

    class J:
        pass
    j = J()
    j.job_id = job_id
    j.dir = jobdir
    j.log = lambda m: print("    " + m, flush=True)

    sb = json.loads((jobdir / "storyboard.json").read_text(encoding="utf-8"))
    params, bj, V, Jn = build(j, sb)
    d = jobdir / "assembled"
    d.mkdir(exist_ok=True)
    np.save(d / "params.npy", params)
    np.save(d / "joints22.npy", Jn)
    (d / "bounds.json").write_text(json.dumps(bj, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(bj["metrics"], ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print("\n  总帧 %d，交界：%s" % (bj["total_frames"],
          [(x["between"][1], x["pose_gap_deg"], x["transition_frames"]) for x in bj["junctions"]]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

def run(job, sb):
    """装配 → 落盘 → 渲染，返回视频路径。"""
    import shutil
    import subprocess
    from hm import config

    params, bj, V, Jn = build(job, sb)
    d = job.dir / "assembled"
    d.mkdir(exist_ok=True)
    np.save(d / "params.npy", params)
    np.save(d / "joints22.npy", Jn)
    (d / "bounds.json").write_text(json.dumps(bj, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(bj["metrics"], ensure_ascii=False, indent=1),
                                    encoding="utf-8")

    # 判据复查
    from hm import criteria_v2
    ok, hard, met, why = criteria_v2.check(np.asarray(Jn), "", None, None)
    bj["assembled_ok"] = bool(ok)
    bj["assembled_reasons"] = why
    (d / "bounds.json").write_text(json.dumps(bj, ensure_ascii=False, indent=1), encoding="utf-8")
    from hm import gates as GT
    gok, gr, gbad = GT.check(V, np.asarray(Jn), params)
    from hm import gates as GT
    _jj = GT.junction_jerk(Jn, bj["junctions"], bj["segments"],
                            backend=(sb.get('segments') or [{}])[0].get('route', {}).get('model'))
    for _x in _jj:
        if _x["suspect"]:
            job.log("A16 交界可疑：%s 峰值 %.1f 度/帧（前段中位的 %.1f 倍）"
                    % ("→".join(_x["between"] or []), _x["peak_deg_per_frame"], _x["ratio"]))
    job.log("A16 自检：rot 中位 %.2f 度/帧 ｜ hf_ratio %.2f ｜ %s"
            % (gr["rot_med"], gr["hf_ratio"],
               "通过" if gok else "不通过：" + ",".join(gbad)))
    bj["selfcheck"] = dict(gr, ok=bool(gok), bad=gbad)

    job.log("A16 参数空间装配：%d 帧，骨长误差 %.1f 厘米，反向帧 %.0f%%，几何检查 %s"
            % (bj["total_frames"], met.get("bone_err_cm", 0),
               100 * met.get("rev_frac", 0), "通过" if ok else "不通过"))

    rd = job.dir / "render"
    rd.mkdir(exist_ok=True)
    np.save(rd / "verts.npy", np.asarray(V, dtype=np.float64))
    np.save(rd / "params.npy", params)
    faces = ROOT / "runs" / "R01-wave-turn-walk" / "faces.npy"
    if faces.exists():
        shutil.copyfile(faces, rd / "faces.npy")
    import os as _os
    quality = _os.environ.get("HM_RENDER_QUALITY", "fast")
    from hm import cache as _cache
    _rk = _cache.key(("render", np.asarray(params).tobytes(), quality,
                      _os.environ.get("HM_RENDER_RES", ""),
                      _os.environ.get("HM_RENDER_SAMPLES", "")))
    _hit = _cache.get_file("render", _rk, "out.mp4") if _cache.enabled() else None
    if _hit is not None:
        import shutil as _sh
        (rd / "r").mkdir(parents=True, exist_ok=True)
        _sh.copyfile(_hit, rd / "r" / "out.mp4")
        job.log("A18 渲染命中缓存（参数与画面设置未变）")
    else:
        subprocess.run([str(config.BLENDER), "--background", "--python",
                        str(ROOT / "hm" / "render_blender.py"), "--", str(rd), str(rd / "r"),
                        "video", "", "0.78", "full", "1.0", quality],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if _cache.enabled() and (rd / "r" / "out.mp4").exists():
            _cache.put_file("render", _rk, "out.mp4", rd / "r" / "out.mp4")
    mp4 = rd / "r" / "out.mp4"
    if mp4.exists():
        shutil.copyfile(mp4, rd / "final.mp4")
    # 交付件帧数校验：成片帧数必须 == 装配帧数（逐帧解码计数，不能读容器头部）
    try:
        from hm.frame_crop import count_frames as _cf
        _n_asm = int(len(params))
        _n_mp4 = _cf(rd / "final.mp4")
        if _n_mp4 != _n_asm:
            job.log("交付件帧数校验失败：装配 %d 帧，成片 %d 帧（按缺陷处理）" % (_n_asm, _n_mp4))
        else:
            job.log("交付件帧数校验通过：%d 帧" % _n_asm)
    except Exception as _e13:
        job.log("交付件帧数校验跳过：%s" % str(_e13)[:200])
    job.log("A18 渲染完成（画质 %s）：%s" % (quality, rd / "final.mp4"))
    import os as _os2
    if _os2.environ.get("HM_DELIVER", "1") == "1":
        try:
            from hm import frame_kit
            frame_kit.deliver(job, job.job_id)
            job.log("A19 交付件：final_cropped.mp4、final_captioned.mp4、分镜.png")
        except Exception as e:
            job.log("A19 交付件生成失败：%s" % str(e)[:120])
    return rd / "final.mp4", bj
