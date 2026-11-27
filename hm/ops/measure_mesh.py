"""网格法质量度量：手部是否插进躯干、脚是否穿地。

为什么不用胶囊
    胶囊（关节 + 固定半径）在标准集上 30/30 任务、100% 帧都被判"穿模"，
    完全没有区分度（正常站立时上臂本来就贴着躯干）。这里改用真实网格：
    1 SMPL 前向得到每帧 6890 个顶点
    2 按「顶点最近关节」把顶点分到躯干 / 手 / 脚等部位（一次算好，全帧复用）
    3 用 trimesh 的有符号距离判断手部顶点是否落在躯干面内（正 = 在里面）
    4 脚穿地直接用网格最低点与地面高度比

用法
    python -m hm.ops.measure_mesh --glob "big_s*_r20" --stride 10
输出
    每个任务：手插躯干的帧数与最大插入深度（米）、网格穿地深度、部位顶点数
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

TORSO_JOINTS = (0, 3, 6, 9, 12)      # 骨盆/脊柱/胸/颈
HAND_JOINTS = (20, 21)               # 左右手
FOOT_JOINTS = (7, 8, 10, 11)         # 脚踝与脚


def _load_faces():
    """SMPL 面片：直接用 fit.py 已维护的那份模型（它会处理目录布局）。"""
    from hm import fit as FM

    FM._patch_paths()
    m = FM._load_smpl("cpu")
    return np.asarray(m.faces, dtype=np.int64)


def _verts_of(params, stride=10):
    """参数 → 顶点（按 stride 抽帧），返回 (verts, 对应帧号)。"""
    import torch
    from hm import fit as FM

    m = FM._load_smpl("cuda" if torch.cuda.is_available() else "cpu")
    dev = next(m.parameters()).device
    out = []
    idx = list(range(0, len(params), stride))
    for i in idx:
        T = torch.from_numpy(np.asarray(params[i:i + 1], dtype=np.float32)).to(dev)
        with torch.no_grad():
            o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
                  transl=T[:, 82:85], return_verts=True)
        out.append(o.vertices.detach().cpu().numpy()[0])
    return np.asarray(out), idx


def _part_of_vertex(verts0, joints0, faces):
    """把顶点分到部位：顶点最近的关节决定它属于哪一块。"""
    d = np.linalg.norm(verts0[:, None, :] - joints0[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    torso_v = np.isin(nearest, TORSO_JOINTS)
    hand_v = np.isin(nearest, HAND_JOINTS)
    foot_v = np.isin(nearest, FOOT_JOINTS)
    # 面片必须三个顶点都在同一块，才属于该块（避免跨块的面片）
    torso_f = faces[torso_v[faces].all(axis=1)]
    return torso_v, hand_v, foot_v, torso_f


def _hull_of(verts, tri_idx):
    """躯干凸包（scipy）。凸包是封闭的，不会因为面片有洞而漏检。"""
    from scipy.spatial import ConvexHull
    pts = np.unique(verts[np.asarray(tri_idx).reshape(-1)], axis=0)
    return ConvexHull(pts), pts


def _inside_hull(hull, pts):
    """凸包内判定：所有半空间方程都 <= 0。"""
    vals = pts @ hull.equations[:, :3].T + hull.equations[:, 3][None, :]
    return np.all(vals <= 1e-9, axis=1)


def _depth_in_hull(hull, pts):
    """在内部时，到最近边界面的距离（米）；用半空间值近似。"""
    vals = pts @ hull.equations[:, :3].T + hull.equations[:, 3][None, :]
    return float(np.min(np.abs(vals.max(axis=1))))


def hand_in_torso(verts, hand_v, torso_f, ground=0.0):
    """手部顶点落在躯干凸包内的帧数与最大插入深度。"""
    frames_hit = 0
    worst = 0.0
    for v in verts:
        try:
            hull, _pts = _hull_of(v, torso_f)
        except Exception:
            continue
        pts = v[hand_v]
        ins = _inside_hull(hull, pts)
        if ins.any():
            frames_hit += 1
            worst = max(worst, _depth_in_hull(hull, pts[ins]))
    return {"sampled_frames": len(verts), "frames_hand_inside": frames_hit,
            "max_depth_m": round(worst, 4)}


def mesh_ground(verts, ground=0.0):
    low = np.min(verts[:, :, 1], axis=1) - float(ground)
    depth = np.maximum(0.0, -low)
    return {"frames_below": int(np.sum(depth > 1e-3)),
            "max_depth_m": round(float(depth.max()) if len(depth) else 0.0, 4)}


def sensitivity_check(verts, hand_v, torso_f, push_m=0.10):
    """判据自检：把手部顶点朝躯干中心推 push_m 米，看能检出多大比例。"""
    v = np.array(verts[0], dtype=np.float64, copy=True)
    try:
        hull, torso_pts = _hull_of(v, torso_f)
    except Exception:
        return None
    center = torso_pts.mean(axis=0)
    for i in np.where(hand_v)[0]:
        direction = center - v[i]
        n = float(np.linalg.norm(direction))
        if n > 1e-6:
            v[i] += direction / n * push_m
    ins = _inside_hull(hull, v[hand_v])
    return round(float(ins.sum()) / max(int(hand_v.sum()), 1), 3)


def judge_selftest(verts, torso_f, n=200, seed=0):
    """判据本体的自检：构造已知内/外的点，看判定是否对。

    与「推 10 厘米」不同：那个测的是几何覆盖是否够宽；这个只测判定本身对不对。
    """
    rng = np.random.default_rng(seed)
    hull, torso_pts = _hull_of(verts[0], torso_f)
    center = torso_pts.mean(axis=0)
    inside_pts = center[None, :] + rng.normal(scale=0.01, size=(n, 3))
    # 外点：从中心沿随机方向走 0.6 米（远超躯干半径）
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    outside_pts = center[None, :] + dirs * 0.6
    return {"inside_detected": round(float(_inside_hull(hull, inside_pts).mean()), 3),
            "outside_detected": round(float(_inside_hull(hull, outside_pts).mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="big_s*_r20")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    jobs = sorted(d for d in ROOT.joinpath("outputs").glob(a.glob)
                  if (d / "assembled" / "params.npy").exists())
    if not jobs:
        print("  没有匹配的任务：outputs/%s" % a.glob)
        return
    faces = _load_faces()
    rows = []
    for d in jobs:
        P = np.load(d / "assembled" / "params.npy")
        sb = json.loads((d / "storyboard.json").read_text(encoding="utf-8")) if (d / "storyboard.json").exists() else {}
        ground = float(((sb.get("terrain") or {}).get("start_y") or 0.0))
        verts, idx = _verts_of(P, stride=a.stride)
        # 部位划分用第一帧的姿态（顶点到关节距离足够稳）
        from hm import fit as FM
        import torch
        m = FM._load_smpl("cuda" if torch.cuda.is_available() else "cpu")
        dev = next(m.parameters()).device
        T = torch.from_numpy(np.asarray(P[idx[0]:idx[0] + 1], dtype=np.float32)).to(dev)
        with torch.no_grad():
            o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
                  transl=T[:, 82:85], return_joints=True)
        J0 = o.joints.detach().cpu().numpy()[0][:22]
        torso_v, hand_v, foot_v, torso_f = _part_of_vertex(verts[0], J0, faces)
        hi = hand_in_torso(verts, hand_v, torso_f, ground)
        hi["judge_selftest"] = judge_selftest(verts, torso_f)
        hi["selftest_push10cm_inside_ratio"] = sensitivity_check(verts, hand_v, torso_f)
        mg = mesh_ground(verts, ground)
        rows.append({"job": d.name, "frames": int(len(P)), **hi, **mg,
                     "hand_vertices": int(hand_v.sum()), "torso_faces": int(len(torso_f))})
        jt = hi["judge_selftest"]
        print("  %-16s 采样 %2d 帧｜手插躯干 %d 帧（最大 %.3f 米）｜网格穿地 %d 帧（最大 %.3f 米）｜判定自检 内%.2f/外%.2f"
              % (d.name, hi["sampled_frames"], hi["frames_hand_inside"], hi["max_depth_m"],
                 mg["frames_below"], mg["max_depth_m"], jt["inside_detected"], jt["outside_detected"]))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("  明细已写：%s" % a.json)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
