"""A16 的参数空间版本：逐段拟合，在 SMPL 参数空间做过渡。

与关节空间版本的区别：
    关节空间  对齐 → 关节方向插值 → 整条拟合（过渡不受骨骼约束，会内翻）
    参数空间  对齐 → 逐段拟合 → 关节旋转做球面插值 → 直接输出参数（不会内翻）

代价：每段拟合一次，比整条拟合多花约 (段数-1)×40 秒。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import fit as FM, finish_job as FJ, job as jobmod

FM._patch_paths()

ROT_DIM = 72          # 0:3 根 + 3:72 身体
BETAS = (72, 82)
TRANSL = (82, 85)


def slerp_param_transition(Pa, Pb, n):
    """逐关节四元数插值 + 平移线性；体型冻结为 Pa 的值。"""
    a = Pa[0:ROT_DIM]
    b = Pb[0:ROT_DIM]
    out = np.zeros((n, 85))
    ts = np.linspace(0.0, 1.0, n + 2)[1:-1]
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
    chosen = FJ.choose_combination(job, sb, verbose=verbose)

    pieces, seg_params, bounds = [], [], []
    prev = None
    for seg, n in chosen:
        J = np.load(job.dir / "segments" / seg["segment_id"] /
                    ("candidate_%d" % n) / "joints22.npy")
        if prev is None:
            placed, dyaw = J, 0.0
        else:
            placed, dyaw = FJ.align_pair(prev, J)
        if verbose:
            print("    拟合 %s ..." % seg["segment_id"], flush=True)
        _, P = FM.fit(placed)
        P = np.asarray(P, dtype=np.float64)
        pieces.append(placed)
        seg_params.append(P)
        bounds.append({"segment_id": seg["segment_id"], "candidate": "cand_%02d" % n,
                       "frames": int(len(placed)), "align_yaw_deg": round(dyaw, 2),
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
        tr = slerp_param_transition(seg_params[i - 1][-1], seg_params[i][0], n)
        junctions.append({"between": [chosen[i - 1][0]["segment_id"], chosen[i][0]["segment_id"]],
                          "pose_gap_deg": round(gap, 1),
                          "boundary_rate_deg_per_frame": round(rate, 2),
                          "transition_frames": n,
                          "extreme": bool(gap > FJ.EXTREME_GAP_DEG),
                          "method": "param_space_slerp"})
        merged.append(tr)
        merged.append(seg_params[i])

    params = np.concatenate(merged, axis=0)
    total = sum(b["frames"] for b in bounds) + sum(j["transition_frames"] for j in junctions)
    if total != len(params):
        raise SystemExit("帧数核对失败：%d vs %d" % (total, len(params)))

    J, V = joints_of_params(params)
    L = jobmod.bone_lengths(J)
    rel = np.abs(L - L.mean(axis=0)) / np.maximum(L.mean(axis=0), 1e-9)
    tr_xy = J[:, 0, [0, 2]]
    step = np.linalg.norm(np.diff(tr_xy, axis=0), axis=1)
    metrics = {"stage": "retimed", "frames": int(len(params)),
               "bone_length_rel_spread_max": float(rel.max()),
               "foot_height_min": float(min(J[:, 10, 1].min(), J[:, 11, 1].min())),
               "root_path_length": float(step.sum()),
               "root_step_median": float(np.median(step))}
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
    subprocess.run([str(config.BLENDER), "--background", "--python",
                    str(ROOT / "hm" / "render_blender.py"), "--", str(rd), str(rd / "r")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mp4 = rd / "r" / "out.mp4"
    if mp4.exists():
        shutil.copyfile(mp4, rd / "final.mp4")
    job.log("A18 渲染完成：%s" % (rd / "final.mp4"))
    return rd / "final.mp4", bj
