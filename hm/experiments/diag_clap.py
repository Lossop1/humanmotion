"""查鼓掌：前臂是不是在胸前交叉；这个交叉是模型给的还是拟合引入的。

同时检验穿模检查的一个漏洞：按「序号差 ≤2」排除相邻对，
会把左右手腕(20,21)、左右肘(18,19)、左右肩(16,17) 一起排除掉。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hm import fit as FM, generate, job as jobmod, to_joints

FM._patch_paths()
BONES = set(jobmod.BONES)

PAIRS = [("左腕", 20, "右腕", 21), ("左肘", 18, "右肘", 19),
         ("左肩", 16, "右肩", 17), ("左腕", 20, "右肘", 19),
         ("右腕", 21, "左肘", 18), ("左腕", 20, "头", 15), ("右腕", 21, "头", 15)]


def report(J, tag):
    T = len(J)
    print("  [%s]" % tag)
    # 手腕是否越过身体中线：以左右髋的中点与朝向定侧向
    cross = 0
    dists = {k: [] for k in [p[0] + "-" + p[2] for p in PAIRS]}
    for t in range(T):
        hips = J[t, 2] - J[t, 1]
        hips[1] = 0.0
        side = hips / max(np.linalg.norm(hips), 1e-9)
        lw = float(np.dot(J[t, 20] - 0.5 * (J[t, 1] + J[t, 2]), side))
        rw = float(np.dot(J[t, 21] - 0.5 * (J[t, 1] + J[t, 2]), side))
        if lw > rw:                      # 左右手位置对调 = 交叉
            cross += 1
        for a, ia, b, ib in PAIRS:
            dists[a + "-" + b].append(float(np.linalg.norm(J[t, ia] - J[t, ib])))
    print("    双手交叉的帧：%d/%d（%.0f%%）" % (cross, T, 100 * cross / T))
    for k, v in dists.items():
        v = np.array(v)
        print("    %-10s 最小 %5.1f 厘米  中位 %5.1f 厘米" % (k, 100 * v.min(), 100 * np.median(v)))


def main():
    prompt, frames, model = "a person claps their hands", 40, "momask"
    F = generate.generate([prompt], [frames], name=model)[0]
    J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
    report(J, "模型输出（22 关节，未拟合）")

    _, P = FM.fit(J)
    import torch
    m = FM._load_smpl("cuda")
    T = torch.from_numpy(np.asarray(P, dtype=np.float32)).to(next(m.parameters()).device)
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True)
    Jf = o.joints.detach().cpu().numpy()[:, :22, :]
    print()
    report(np.asarray(Jf, dtype=np.float64), "拟合之后")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
