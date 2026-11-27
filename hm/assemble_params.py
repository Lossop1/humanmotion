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
    _g_forward = None          # 世界"前方"：第一段有位移的段的自然行进方向
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

        # 朝向执行：把本段的位移方向转到请求的世界方向
        if _g_forward is None:
            _phi0, _d0 = CTL._disp_dir(placed)
            if _phi0 is not None:
                # 世界前方不能直接取第一段的自然方向，否则第一段请求 backward 时
                # 目标 = 自己+180°，会被自己转 180°，把正确的后退改成前进。
                _req0 = str(seg.get("facing") or seg.get("direction") or "").strip().lower()
                _g_forward = _phi0 - CTL.DIR_ANGLE[_req0] if _req0 in CTL.DIR_ANGLE else _phi0
        placed, _face = CTL.enforce_facing(placed, seg.get("facing") or seg.get("direction"),
                                       _g_forward if _g_forward is not None else 0.0)
        # 地形执行：把本段整体对齐到规划给出的地面高度
        # 地形按「原段号」取高度：合并会让段序号错位，不能用循环下标
        try:
            _sid = int(str(seg["segment_id"]).split("_")[-1])
        except (ValueError, KeyError):
            _sid = _idx
        _ground = _profile.get(_sid, _terrain.get("start_y"))
        placed, _dy = CTL.apply_terrain(placed, _ground)
        # 速度执行：slow/fast 用帧数伸缩实现（后端没有速度接口）
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
        bounds.append({"segment_id": seg["segment_id"], "candidate": "cand_%02d" % n,
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
        n = int(max(2, min(60, round(gap / rate))))
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
    job.log("A18 渲染完成（画质 %s）：%s" % (quality, rd / "final.mp4"))
    import os as _os2
    if _os2.environ.get("HM_DELIVER", "1") == "1":
        try:
            import frame_kit
            frame_kit.deliver(job, job.job_id)
            job.log("A19 交付件：final_cropped.mp4、final_captioned.mp4、分镜.png")
        except Exception as e:
            job.log("A19 交付件生成失败：%s" % str(e)[:120])
    return rd / "final.mp4", bj
