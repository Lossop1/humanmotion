"""一句话 → 拆成动作段 → 每段由模型生成 → 过渡由算法规划 → 出四个视频。

动作段是模型生成的，过渡是算法算出来的（不调模型再生成一段）。
两个动作时产出四个视频：
    A1_<动作1>.mp4     单动作 1（模型）
    A2_<动作2>.mp4     单动作 2（模型）
    T1.mp4             过渡（算法：用 A1 末帧和 A2 首帧规划）
    ALL.mp4            合体（A1 + T1 + A2）

用法：
    python -m hm.legacy.parts "一个人先挥手打招呼，然后向前走开" --out P03
"""
import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from hm import config, generate, planner, to_joints
from hm import fit as fitmod
from hm.legacy import join_params as JP

FACES_SRC = config.OUT_ROOT / "R01-wave-turn-walk" / "faces.npy"

# 这些动作生成出来的姿态经不起检查（实测 crouch 生成成跪坐、头俯仰被解到 89 度，
# 网格正面与脚尖相反），先禁用，等生成或拟合这两层有可靠解再放回来。
UNSUPPORTED = ("crouch", "squat", "sit down", "sitting", "kneel", "lie down", "lying")


def render(folder, name):
    """把该文件夹的顶点渲成视频，复制成 <name>.mp4 放在上层。"""
    subprocess.run([str(config.BLENDER), "--background", "--python",
                    str((Path(__file__).resolve().parents[1] / "render_blender.py")), "--",
                    str(folder), str(folder / "render")], check=False)
    src = folder / "render" / "out.mp4"
    if not src.exists():
        return None
    dst = folder.parent / (name + ".mp4")
    shutil.copyfile(src, dst)
    return dst


def save_piece(folder, J, P, V=None):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "joints.npy", np.asarray(J, dtype=np.float64))
    np.save(folder / "params.npy", np.asarray(P, dtype=np.float64))
    if V is not None:
        np.save(folder / "verts.npy", np.asarray(V, dtype=np.float64))
    if FACES_SRC.exists():
        shutil.copyfile(FACES_SRC, folder / "faces.npy")


def fit_feats(F, folder):
    """一段特征 → 关节 → 拟合 → 顶点。返回 (关节, 参数, 顶点)。"""
    J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
    V, P = fit_best(J)
    V, P, _d = fix_facing(V, P, J)
    save_piece(folder, J, P, V)
    return J, P, V


def transition_frames(P1, J1, P2, J2, n=None):
    """用算法规划过渡：n 帧由「两端姿态差 ÷ 边界角速度」定。"""
    if n is None:
        gap = JP.param_angle(P1[-1], P2[0])
        ri = JP.param_rate(P1, len(P1) - 1, 3, tail=True)
        ro = JP.param_rate(P2, 0, 3, tail=False)
        n = int(round(gap / max(ri, ro, 1e-6)))
        n = max(2, min(60, n))
        print("    姿态差 %.1f 度  边界速率 %.1f / %.1f 度/帧  → %d 帧"
              % (gap, ri, ro, n), flush=True)
    merged, lay = JP.merge([P1, P2], [J1, J2], [0, 0], [0, n],
                           curve="hermite", rates={"blend": "polar"})
    return merged, lay, n



# SMPL 24 关节的链：0 骨盆 →3 脊柱1 →6 脊柱2 →9 脊柱3 →12 颈 →15 头
CHAIN_TO_CHEST = (0, 3, 6, 9)
CHAIN_TO_HEAD = (0, 3, 6, 9, 12, 15)

# 末端关节：它们的旋转几乎不影响任何关节位置，拟合可以任意拧。
# 实测挥手段右上臂/腕、蹲下段头与双腕与右踝，网格表面被转过去。
# 每个末端关节配一个父关节，用它当朝向基准。
END_JOINTS = {15: (0, 3, 6, 9, 12),      # 头 ← 颈
              20: (0, 3, 6, 9, 13, 16, 18),  # 左腕 ← 左肘
              21: (0, 3, 6, 9, 14, 17, 19),  # 右腕 ← 右肘
              7: (0, 1, 4),               # 左踝 ← 左膝
              8: (0, 2, 5)}                # 右踝 ← 右膝


def _world_rot(P, frame, chain):
    from scipy.spatial.transform import Rotation as R
    acc = R.identity()
    for j in chain:
        acc = acc * R.from_rotvec(np.asarray(P[frame, j * 3:(j + 1) * 3], dtype=np.float64))
    return acc


def fix_rolls(P, thresh=70.0):
    """把末端关节绕自身轴的旋转校正到与父关节同向。

    关节位置对末端关节的旋转几乎没有约束（腕只带动手、头只带动头），
    拟合会停在任意角度，网格表面就扭了。这里用父关节的朝向当基准。
    """
    from scipy.spatial.transform import Rotation as R
    P = np.array(P, dtype=np.float64, copy=True)
    total = 0
    for t in range(len(P)):
        for jn, chain in END_JOINTS.items():
            rp = _world_rot(P, t, chain[:-1])
            rh = _world_rot(P, t, chain)
            vp = rp.apply(np.array([0.0, 0.0, 1.0]))
            vh = rh.apply(np.array([0.0, 0.0, 1.0]))
            vp[1] = vh[1] = 0.0
            np_, nh = np.linalg.norm(vp), np.linalg.norm(vh)
            if np_ < 1e-6 or nh < 1e-6:
                continue
            ang = math.degrees(math.acos(float(np.clip(np.dot(vp / np_, vh / nh), -1.0, 1.0))))
            if ang < thresh:
                continue
            axis = np.cross(vh / nh, vp / np_)
            na = np.linalg.norm(axis)
            if na < 1e-6:
                axis = np.array([0.0, 1.0, 0.0])
                na = 1.0
            rot_w = R.from_rotvec(axis / na * math.radians(ang))
            if jn == 15:
                # 头跟随颈：相对颈不额外俯仰也不额外扭转。
                # 实测蹲下段头的俯仰被解成近 90 度，看到的是头顶而不是脸。
                # 只修绕轴旋转不够，俯仰也要一起回到颈的朝向。
                P[t, jn * 3:(jn + 1) * 3] = (0.0, 0.0, 0.0)
            else:
                new_world = rot_w * rh
                new_local = rp.inv() * new_world
                P[t, jn * 3:(jn + 1) * 3] = new_local.as_rotvec()
            total += 1
    if total:
        print("    末端关节朝向校正：%d 处" % total, flush=True)
    return P


# ---------- 拟合质量：用网格形变当判据 ----------
# 不依赖任何「正面」定义。骨头绕自身轴乱转会让皮肤被剪切，边长就偏离静止姿态。
_EDGES = None


def _edges():
    global _EDGES
    if _EDGES is None and FACES_SRC.exists():
        F = np.load(FACES_SRC)
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
        E = np.unique(np.sort(E, axis=1), axis=0)[::7]
        rest = JP.verts_from_params(np.zeros((1, 85)))
        L0 = np.linalg.norm(rest[0][E[:, 0]] - rest[0][E[:, 1]], axis=1)
        _EDGES = (E, L0)
    return _EDGES


def mesh_distortion(V):
    """相对静止姿态的边长变化（中位数）。越大说明皮肤被扭得越厉害。"""
    ed = _edges()
    if ed is None:
        return 0.0
    E, L0 = ed
    vals = []
    for t in range(0, len(V), max(1, len(V) // 8)):
        L = np.linalg.norm(V[t][E[:, 0]] - V[t][E[:, 1]], axis=1)
        vals.append(np.abs(L - L0) / L0)
    return float(np.median(np.array(vals)))


def fit_best(J, tries=(20, 60, 120)):
    """拟合；形变超过阈值就加迭代重来，取形变最小的那次。"""
    best = None
    for it in tries:
        V, P = fitmod.fit(J, iters=it)
        P = fix_rolls(np.asarray(P, dtype=np.float64))
        V = np.asarray(JP.verts_from_params(P), dtype=np.float64)
        d = mesh_distortion(V)
        if best is None or d < best[0]:
            best = (d, V, P, it)
        if d < 0.055:
            break
    print("      拟合：迭代 %d，网格形变中位 %.2f%%" % (best[3], 100 * best[0]), flush=True)
    return best[1], best[2]


def fix_facing(V, P, J):
    """身体正面与脚尖方向相反时，把整段绕竖直轴转 180 度。

    实测蹲下段：拟合出来的身体正面与脚尖方向点积 -1.00，其余段 +1.00。
    整段刚性旋转 180 度不改变动作形状，只换世界朝向——而生成出来的动作
    本来就没有绝对朝向。转过来之后是正常的蹲姿（脸可见、手臂垂在身前）。
    """
    from scipy.spatial.transform import Rotation as R
    dots = []
    for t in range(len(P)):
        f = R.from_rotvec(np.asarray(P[t, 0:3], dtype=np.float64)).apply(np.array([0.0, 0.0, 1.0]))
        foot = 0.5 * ((np.asarray(J[t, 10]) - np.asarray(J[t, 7]))
                      + (np.asarray(J[t, 11]) - np.asarray(J[t, 8])))
        f[1] = 0.0
        foot[1] = 0.0
        nf, ng = np.linalg.norm(f), np.linalg.norm(foot)
        if nf < 1e-6 or ng < 1e-6:
            continue
        dots.append(float(np.dot(f / nf, foot / ng)))
    if not dots:
        return V, P, 0.0
    d = float(np.median(dots))
    if d > 0:
        return V, P, d
    P2 = np.array(P, dtype=np.float64, copy=True)
    rot = R.from_euler("y", np.pi)
    P2[:, 0:3] = (rot * R.from_rotvec(P2[:, 0:3])).as_rotvec()
    piv = P2[0, JP.SL_TRANS].copy()
    tr = P2[:, JP.SL_TRANS] - piv
    P2[:, JP.SL_TRANS] = np.stack([-tr[:, 0], tr[:, 1], -tr[:, 2]], axis=1) + piv
    V2 = np.asarray(JP.verts_from_params(P2), dtype=np.float64)
    print("    朝向相反（点积 %.2f），整段转 180 度" % d, flush=True)
    return V2, P2, d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=None, help="过渡帧数，缺省按数据算")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    segs, trans, terrain = planner.plan_full(a.sentence)
    print("  分段：")
    for i, s in enumerate(segs):
        print("    A%d  %3d 帧  %s" % (i + 1, int(s.get("frames", 0)), s.get("prompt", "")))

    keep, dropped = [], []
    for s0 in segs:
        low = (s0.get("prompt") or "").lower()
        (dropped if any(w in low for w in UNSUPPORTED) else keep).append(s0)
    if dropped:
        print("  已禁用以下动作（生成质量不可靠）：", flush=True)
        for s0 in dropped:
            print("    %s  %s" % (s0.get("label", ""), s0.get("prompt", "")), flush=True)
    for i, s0 in enumerate(keep):
        s0["id"] = i + 1
    segs = keep
    if not segs:
        print("  没有任何可用动作"); return

    out = config.OUT_ROOT / (a.out or time.strftime("parts_%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(
        json.dumps({"sentence": a.sentence, "segments": segs, "transitions": trans},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    made = []

    print("  模型生成动作段 ...", flush=True)
    feats = generate.generate([s["prompt"] for s in segs],
                              [int(s["frames"]) for s in segs])
    seg_J, seg_P = [], []
    for i, (s, F) in enumerate(zip(segs, feats)):
        name = "A%d_%s" % (i + 1, (s.get("label") or "seg%d" % (i + 1)))
        J, P, V = fit_feats(F, out / name)
        seg_J.append(J)
        seg_P.append(P)
        np.save(out / ("feats_%d.npy" % (i + 1)), F)
        if not a.no_render:
            v = render(out / name, name)
            if v:
                made.append((name, v))
                print("    %-18s %3d 帧  %s" % (name, len(J), v.name), flush=True)

    print("  算法规划过渡 ...", flush=True)
    # 先把每处过渡的帧数按数据算好，再一次拼出整条。
    # 之前是每处过渡单独拼一次并写 ALL，三段以上会被后一次覆盖，
    # 结果 ALL 里只剩最后一个接缝（实测 Q3 的 ALL 只有 94 帧）。
    fades = [0]
    for k in range(len(segs) - 1):
        gap = JP.param_angle(seg_P[k][-1], seg_P[k + 1][0])
        ri = JP.param_rate(seg_P[k], len(seg_P[k]) - 1, 3, tail=True)
        ro = JP.param_rate(seg_P[k + 1], 0, 3, tail=False)
        n = max(2, min(60, int(round(gap / max(ri, ro, 1e-6)))))
        fades.append(n)
        print("    接缝%d 姿态差 %.1f 度  边界速率 %.1f / %.1f 度/帧  → %d 帧"
              % (k + 1, gap, ri, ro, n), flush=True)

    merged, lay = JP.merge(seg_P, seg_J, [0] * len(segs), fades,
                          curve="hermite", rates={"blend": "polar"})
    total = sum(int(r.get("frames", 0)) + int(r.get("fade_frames", 0)) for r in lay)
    print("    整条 %d 帧  layout合计 %d  %s"
          % (len(merged), total, "一致" if total == len(merged) else "不一致"), flush=True)

    # 从拼好的序列里切出每处过渡
    off = 0
    for rw in lay:
        fd = int(rw.get("fade_frames", 0))
        if fd > 0:
            T = merged[off:off + fd]
            name = "T%d" % rw["index"]
            save_piece(out / name, np.zeros((len(T), 22, 3)), T,
                       np.asarray(JP.verts_from_params(T), dtype=np.float64))
            print("    %-18s %3d 帧" % (name, len(T)), flush=True)
            if not a.no_render:
                v = render(out / name, name)
                if v:
                    made.append((name, v))
                    print("    %-18s %3d 帧  %s" % (name, len(T), v.name), flush=True)
            off += fd
        off += int(rw["frames"])

    folder = out / "ALL"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "params.npy", merged)
    np.save(folder / "verts.npy", np.asarray(JP.verts_from_params(merged), dtype=np.float64))
    if FACES_SRC.exists():
        shutil.copyfile(FACES_SRC, folder / "faces.npy")
    if not a.no_render:
        v = render(folder, "ALL")
        if v:
            made.append(("ALL", v))
            print("    %-18s %3d 帧  %s" % ("ALL", len(merged), v.name), flush=True)

    print("\n  产出：")
    for n_, p_ in made:
        print("    %-20s %s" % (n_, p_))
    print("  用时 %.1f 秒" % (time.time() - t0))


if __name__ == "__main__":
    main()