"""A16–A18：装配、SMPL 拟合、渲染，产出合体视频。

用法：
    python -m hm.finish_job job_test04

依据《设计说明书_原子步骤》：
    A16 依据目标位置和朝向做全局平移/旋转；段间用默认简单过渡；
        只有诊断为极端姿态差时才用旋转空间插值或单独桥接。
        输出 assembled/joints22.npy 与 bounds.json，帧数必须可相加核对，
        装配后重新测量位移指标，并标注指标所在阶段。
    A17 严格按 [pose(72), betas(10), cam_t(3)] 保存，报告关节误差分布而非只报均值。
    A18 沿用已验证的渲染路径，输出 final.mp4 与 render_config.json。
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from hm import config, fit as fitmod
from hm import job as jobmod

FACES_SRC = config.OUT_ROOT / "R01-wave-turn-walk" / "faces.npy"
BONES = jobmod.BONES
EXTREME_GAP_DEG = 90.0     # 超过这个姿态差，交界标记为极端，但仍走默认简单过渡
DEFAULT_TRANSITION = 10    # 简单过渡的默认帧数（数据算出的 N 与它取较接近的那个）


# ---------------- 几何工具 ----------------
def dirs_of(J):
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


def rot_about(v, axis, ang):
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1 - c)


def body_forward(Jf):
    acc = np.zeros(3)
    for a, t in ((7, 10), (8, 11)):
        v = np.asarray(Jf[t]) - np.asarray(Jf[a])
        v[1] = 0.0
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            acc += v / n
    n = float(np.linalg.norm(acc))
    if n > 1e-6:
        return acc / n
    hips = np.asarray(Jf[2]) - np.asarray(Jf[1])
    hips[1] = 0.0
    f = np.cross(np.array([0.0, 1.0, 0.0]), hips)
    nf = float(np.linalg.norm(f))
    return f / nf if nf > 1e-6 else np.array([0.0, 0.0, 1.0])


def align_pair(JA, JB):
    """把 B 段整体转到与 A 末帧同朝向，并把骨盆平移到 A 末帧位置。"""
    fA, fB = body_forward(JA[-1]), body_forward(JB[0])
    ang = -math.atan2(float(np.cross(fA, fB)[1]), float(np.dot(fA, fB)))
    c, sn = math.cos(ang), math.sin(ang)
    piv = np.asarray(JB[0, 0], dtype=np.float64).copy()
    out = np.array(JB, dtype=np.float64, copy=True)
    for i in range(len(out)):
        for k in range(22):
            v = out[i, k] - piv
            out[i, k] = piv + np.array([c * v[0] + sn * v[2], v[1], -sn * v[0] + c * v[2]])
    out += (np.asarray(JA[-1, 0]) - piv)[None, None, :]
    return out, math.degrees(ang)


def interp_dirs(A, B, n, us=None):
    """方向空间插值：骨长取两端平均并锁定，位置只由根节点线性给出。"""
    dA, LA = dirs_of(A)
    dB, LB = dirs_of(B)
    L = 0.5 * (LA + LB)
    if us is None:
        us = (np.arange(1, n + 1, dtype=np.float64)) / (n + 1.0)
    frames = np.zeros((n, 22, 3))
    for i in range(n):
        u = float(us[i])
        frames[i, 0] = A[0] * (1 - u) + B[0] * u
        for k, (c, p) in enumerate(BONES):
            ax, ang = rot_between(dA[k], dB[k])
            d = rot_about(dA[k], ax, ang * u) if ang > 1e-9 else dA[k]
            frames[i, c] = frames[i, p] + L[k] * d
    return frames


def pose_gap(A, B):
    dA, _ = dirs_of(A)
    dB, _ = dirs_of(B)
    return max(math.degrees(rot_between(dA[k], dB[k])[1]) for k in range(len(BONES)))


def bone_rate(J, tail=True, w=3):
    idx = range(max(0, len(J) - 1 - w), len(J) - 1) if tail else range(0, min(w, len(J) - 1))
    vals = []
    for t in idx:
        d0, _ = dirs_of(J[t])
        d1, _ = dirs_of(J[t + 1])
        vals.append(max(math.degrees(rot_between(d0[k], d1[k])[1]) for k in range(len(BONES))))
    return float(np.mean(vals)) if vals else 1.0



def bend_sign(Jf, root, mid, tip):
    """关节弯曲方向：叉积在身体侧向上的分量。"""
    f = body_forward(Jf)
    side = np.cross(np.array([0.0, 1.0, 0.0]), f)
    a = Jf[mid] - Jf[root]
    b = Jf[tip] - Jf[mid]
    return float(np.dot(np.cross(a, b), side))


BEND_CHAINS = (("左肘", 16, 18, 20), ("右肘", 17, 19, 21),
               ("左膝", 1, 4, 7), ("右膝", 2, 5, 8))


def bend_flips(J):
    """每个关节在整条序列里弯曲方向的翻转帧号。"""
    out = {}
    for nm, r, m, t in BEND_CHAINS:
        signs = np.array([bend_sign(J[i], r, m, t) for i in range(len(J))])
        pos = signs > 0
        out[nm] = [int(i) for i in range(1, len(signs)) if pos[i] != pos[i - 1]]
    return out


def validate_assembled(J):
    """装配后整条序列的检查。返回 (ok, 明细)。"""
    from hm import criteria_v2
    ok, hard, met, why = criteria_v2.check(J, "", None, None)
    flips = bend_flips(J)
    detail = {
        "criteria_ok": bool(ok),
        "hard": {k: bool(v) for k, v in hard.items()},
        "reasons": why,
        "bone_rel_mean_max": met.get("bone_rel_mean_max"),
        "rev_frac": met.get("rev_frac"),
        "min_separation_m": met.get("min_separation_m"),
        "bend_flips": {k: (v[:8] if v else []) for k, v in flips.items()},
        "bend_flip_count": {k: len(v) for k, v in flips.items()},
    }
    total_flips = sum(len(v) for v in flips.values())
    detail["bend_total_flips"] = total_flips
    return (ok and total_flips == 0), detail


def segment_candidates(job, sb):
    """每段可用的候选列表（只取判据通过的）。"""
    out = []
    for seg in sb["segments"]:
        cands = []
        for c in seg.get("candidates", []):
            n = int(c["candidate_id"].split("_")[-1])
            m = job.dir / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "metrics.json"
            try:
                import json as _json
                mm = _json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                continue
            if mm.get("decision") == "pass":
                cands.append(n)
        out.append((seg, cands))
    return out


def choose_combination(job, sb, verbose=True):
    """动态规划：为每段选一个候选，使相邻段交界姿态差之和最小。"""
    segs = segment_candidates(job, sb)
    import os as _os
    policy = _os.environ.get("HM_DEGRADE", "drop")   # drop|clip|abort
    if any(not c for _, c in segs):
        bad = [s["segment_id"] for s, c in segs if not c]
        if policy == "abort":
            raise SystemExit("这些段没有通过判据的候选：%s" % bad)
        if policy == "drop":
            job.log("取舍：%s 没有通过判据的候选，从成片里去掉（HM_DEGRADE=drop）"
                    % "、".join(bad))
            segs = [(s, c) for s, c in segs if c]
    if any(not c for _, c in segs):
        # 降级：没有通过判据的段，改用问题项最少的候选，并显式记录
        import json as _j
        relaxed = []
        for s, c in segs:
            if c:
                relaxed.append((s, c))
                continue
            best_n, best_k = None, 10 ** 6
            for cd in s.get("candidates", []):
                n = int(cd["candidate_id"].split("_")[-1])
                mp = job.dir / "segments" / s["segment_id"] / ("candidate_%d" % n) / "metrics.json"
                try:
                    mm = _j.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                k = len(mm.get("failure_reasons") or [])
                if k < best_k:
                    best_n, best_k = n, k
            if best_n is None:
                relaxed.append((s, []))
            else:
                job.log("降级：%s 无通过判据的候选，改用 candidate_%d（%d 项问题）"
                        % (s["segment_id"], best_n, best_k))
                relaxed.append((s, [best_n]))
        segs = relaxed
    if any(not c for _, c in segs):
        bad = [s["segment_id"] for s, c in segs if not c]
        raise SystemExit("这些段没有任何候选，无法装配：%s" % bad)

    cache = {}

    def aligned_last(si, ci):
        """第 si 段第 ci 个候选（已按前一段对齐后）的末帧。"""
        key = (si, ci)
        if key in cache:
            return cache[key]
        n = segs[si][1][ci]
        J = np.load(job.dir / "segments" / segs[si][0]["segment_id"] /
                    ("candidate_%d" % n) / "joints22.npy")
        if si == 0:
            cache[key] = J
            return J
        # 对齐到前一段的每个候选，取平均位置不合适；这里只用于估算，直接用首段姿态
        cache[key] = J
        return J

    # 用「各段首末帧原始姿态」估算交界差即可（对齐只改朝向，不改姿态差）
    Js = []
    for si, (seg, cands) in enumerate(segs):
        row = {}
        for ci, n in enumerate(cands):
            row[ci] = np.load(job.dir / "segments" / seg["segment_id"] /
                              ("candidate_%d" % n) / "joints22.npy")
        Js.append(row)

    def gap(a, b):
        # 必须先按真实装配方式对齐，否则估出来的姿态差和实际不一致
        b2, _ = align_pair(a, b)
        g = pose_gap(a[-1], b2[0])
        # 弯曲方向相反 → 插值必然穿过一次反向，直接重罚
        bad = 0
        for nm, r, m_, t_ in BEND_CHAINS:
            s1 = bend_sign(a[-1], r, m_, t_)
            s2 = bend_sign(b2[0], r, m_, t_)
            if s1 * s2 < 0:
                bad += 1
        return g + 120.0 * bad

    M = [len(c) for _, c in segs]
    INF = 1e18
    cost = [[0.0] * M[0]]
    back = [[-1] * M[0]]
    for si in range(1, len(M)):
        c_new, b_new = [], []
        for cb in range(M[si]):
            best, arg = INF, -1
            for ca in range(M[si - 1]):
                v = cost[si - 1][ca] + gap(Js[si - 1][ca], Js[si][cb])
                if v < best:
                    best, arg = v, ca
            c_new.append(best)
            b_new.append(arg)
        cost.append(c_new)
        back.append(b_new)

    last = int(np.argmin(cost[-1]))
    pick = [0] * len(M)
    pick[-1] = last
    for si in range(len(M) - 1, 0, -1):
        pick[si - 1] = back[si][pick[si]]

    if verbose:
        tot = cost[-1][pick[-1]]
        print("    联合选择：%s   总代价 %.1f（含弯曲不一致惩罚）"
              % (["cand_%02d" % (segs[i][1][pick[i]]) for i in range(len(M))], tot), flush=True)
    return [(segs[i][0], segs[i][1][pick[i]]) for i in range(len(M))]

# ---------------- A16 ----------------
def step_a16(job, sb, verbose_choice=True):
    segs = [s for s in sb["segments"] if s.get("selected")]
    if not segs:
        raise SystemExit("没有选中的片段，无法装配")
    if len(segs) != len(sb["segments"]):
        job.log("A16 注意：%d 段里有 %d 段被淘汰，装配只使用通过的段"
                % (len(sb["segments"]), len(segs)))

    if verbose_choice:
        pass
    chosen = choose_combination(job, sb)
    pieces, bounds = [], []
    prev = None
    for seg, n in chosen:
        s = seg
        s["selected"] = "cand_%02d" % n
        J = np.load(job.dir / "segments" / s["segment_id"] / ("candidate_%d" % n) / "joints22.npy")
        if prev is None:
            placed, dyaw = J, 0.0
        else:
            placed, dyaw = align_pair(prev, J)
        pieces.append(placed)
        bounds.append({"segment_id": s["segment_id"], "candidate": s["selected"],
                       "frames": int(len(placed)), "align_yaw_deg": round(dyaw, 2),
                       "stage": "retimed"})
        prev = placed

    # 段间过渡：先量姿态差，再定帧数；超过阈值的交界标记为极端
    merged = [pieces[0]]
    junctions = []
    for i in range(1, len(pieces)):
        A, B = pieces[i - 1][-1], pieces[i][0]
        gap = pose_gap(A, B)
        rate = max(bone_rate(pieces[i - 1], tail=True), bone_rate(pieces[i], tail=False), 1e-6)
        n = int(max(2, min(60, round(gap / rate))))
        n = int(min(n, max(2, DEFAULT_TRANSITION)) if gap <= EXTREME_GAP_DEG else n)
        tr = interp_dirs(A, B, n)
        junctions.append({
            "between": [segs[i - 1]["segment_id"], segs[i]["segment_id"]],
            "pose_gap_deg": round(gap, 1),
            "boundary_rate_deg_per_frame": round(rate, 2),
            "transition_frames": n,
            "extreme": bool(gap > EXTREME_GAP_DEG),
            "method": "direction_space_linear",
        })
        merged.append(tr)
        merged.append(pieces[i])

    seq = np.concatenate(merged, axis=0)
    total = sum(b["frames"] for b in bounds) + sum(j["transition_frames"] for j in junctions)
    if total != len(seq):
        job.log("A16 帧数不一致：分项 %d，实际 %d" % (total, len(seq)))
        raise SystemExit("帧数核对失败")

    # 装配后重新量指标（阶段标注为 retimed）
    L = jobmod.bone_lengths(seq)
    rel = np.abs(L - L.mean(axis=0)) / np.maximum(L.mean(axis=0), 1e-9)
    tr_xy = seq[:, 0, [0, 2]]
    step = np.linalg.norm(np.diff(tr_xy, axis=0), axis=1)
    metrics = {
        "stage": "retimed",
        "frames": int(len(seq)),
        "bone_length_rel_spread_max": float(rel.max()),
        "foot_height_min": float(min(seq[:, 10, 1].min(), seq[:, 11, 1].min())),
        "root_path_length": float(step.sum()),
        "root_step_median": float(np.median(step)),
        "root_step_max": float(step.max()),
    }
    out = job.assembled_dir()
    np.save(out / "joints22.npy", seq)
    ok, vdetail = validate_assembled(seq)
    bj = {"job_id": job.job_id, "segments": bounds, "junctions": junctions,
          "total_frames": total, "metrics": metrics, "fps": 20,
          "assembled_validation": vdetail, "assembled_ok": bool(ok)}
    (out / "bounds.json").write_text(json.dumps(bj, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    job.log("A16 装配完成：%d 段 + %d 处过渡 = %d 帧，最大交界姿态差 %.1f 度"
            % (len(bounds), len(junctions), total,
               max((j["pose_gap_deg"] for j in junctions), default=0.0)))
    job.log("A16 装配后检查：判据 %s，关节弯曲翻转 %d 次 %s"
            % ("通过" if vdetail["criteria_ok"] else "不通过",
               vdetail["bend_total_flips"],
               "" if vdetail["bend_total_flips"] == 0
               else str({k: v for k, v in vdetail["bend_flips"].items() if v})))
    return seq, bj


# ---------------- A17 ----------------
def step_a17(job, seq):
    V, P = fitmod.fit(seq)
    P = np.asarray(P, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    assert P.ndim == 2 and P.shape[1] == 85, "参数形状不是 (T,85)：%s" % (P.shape,)
    assert P.shape[0] == len(seq), "参数帧数与关节帧数不一致"
    out = job.smpl_dir()
    np.save(out / "params.npy", P)
    np.save(out / "verts.npy", V)
    if FACES_SRC.exists():
        shutil.copyfile(FACES_SRC, out / "faces.npy")

    # 关节误差分布：报 p50 / p95 / max，不只报均值
    from hm import fit as FM
    import torch
    FM._patch_paths()
    m = FM._load_smpl("cuda")
    T = torch.from_numpy(P.astype(np.float32)).to(next(m.parameters()).device)
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True)
    Jf = o.joints.detach().cpu().numpy()[:, :22, :]
    err = np.linalg.norm(np.asarray(seq) - Jf, axis=2)
    # R4（根朝向 vs 脚尖方向）：判整段姿态是否相对脚反向
    from scipy.spatial.transform import Rotation as R
    dots = []
    for t in range(len(P)):
        fwd = R.from_rotvec(P[t, 0:3]).apply(np.array([0.0, 0.0, 1.0]))
        foot = np.zeros(3)
        for aa, tt in ((7, 10), (8, 11)):
            v = seq[t, tt] - seq[t, aa]
            v[1] = 0.0
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                foot += v / nv
        nf, ng = float(np.linalg.norm(fwd[[0, 2]])), float(np.linalg.norm(foot[[0, 2]]))
        if nf < 1e-6 or ng < 1e-6:
            continue
        dots.append(float(np.dot(fwd[[0, 2]] / nf, foot[[0, 2]] / ng)))
    dots = np.array(dots) if dots else np.array([0.0])
    r4 = {"median": float(np.median(dots)), "min": float(dots.min()),
          "frames_negative": int((dots < 0).sum()), "frames": int(len(dots)),
          "passed": bool((dots < 0).sum() == 0)}

    stats = {
        "stage": "smpl_fit",
        "R4_root_vs_toes": r4,
        "params_shape": list(P.shape),
        "column_order": "[pose(72), betas(10), cam_t(3)]",
        "frames": int(len(P)),
        "joint_error_m": {
            "median": float(np.median(err)),
            "p95": float(np.percentile(err, 95)),
            "max": float(err.max()),
        },
    }
    (out / "metrics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    job.log("A17 拟合完成：关节误差 中位 %.3f / p95 %.3f / 最大 %.3f 米；"
            "根朝向与脚尖点积中位 %.2f，反向帧 %d/%d"
            % (stats["joint_error_m"]["median"], stats["joint_error_m"]["p95"],
               stats["joint_error_m"]["max"], r4["median"],
               r4["frames_negative"], r4["frames"]))
    return V, P, stats


# ---------------- A18 ----------------
def step_a18(job, seq, bj):
    smpl = job.smpl_dir()
    rd = job.render_dir()
    src = smpl
    subprocess.run([str(config.BLENDER), "--background", "--python",
                    str(Path(__file__).resolve().with_name("render_blender.py")), "--",
                    str(src), str(rd)], check=False)
    src_mp4 = rd / "out.mp4"
    if not src_mp4.exists():
        raise SystemExit("渲染未产出视频")
    shutil.copyfile(src_mp4, rd / "final.mp4")
    cfg = {
        "stage": "rendered",
        "frames": int(len(seq)),
        "fps": 20,
        "resolution": [720, 720],
        "camera": "投影校正，人物占画面高度 62%",
        "renderer": "blender",
        "renderer_version": "4.5.0",
        "source": str(smpl.relative_to(job.dir)),
        "bounds_file": "assembled/bounds.json",
    }
    (rd / "render_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    job.log("A18 渲染完成：%s（%d 帧）" % (rd / "final.mp4", len(seq)))
    return rd / "final.mp4"


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not job_id:
        raise SystemExit("用法：python -m hm.finish_job <job_id>")
    jobdir = jobmod.OUTPUT_ROOT / job_id
    if not jobdir.exists():
        raise SystemExit("任务目录不存在：%s" % jobdir)

    class _J:
        pass

    j = _J()
    j.job_id = job_id
    j.dir = jobdir
    j.log = lambda m: (print("    " + m, flush=True),
                       open(jobdir / "run.log", "a", encoding="utf-8").write(
                           "[%s] %s\n" % (__import__("time").strftime("%H:%M:%S"), m)))[0]
    j.assembled_dir = lambda: (jobdir / "assembled").mkdir(exist_ok=True) or (jobdir / "assembled")
    j.smpl_dir = lambda: (jobdir / "smpl").mkdir(exist_ok=True) or (jobdir / "smpl")
    j.render_dir = lambda: (jobdir / "render").mkdir(exist_ok=True) or (jobdir / "render")

    sb = json.loads((jobdir / "storyboard.json").read_text(encoding="utf-8"))
    seq, bj = step_a16(j, sb)
    V, P, stats = step_a17(j, seq)
    mp4 = step_a18(j, seq, bj)
    print("\n  合体视频：%s" % mp4)


if __name__ == "__main__":
    main()
