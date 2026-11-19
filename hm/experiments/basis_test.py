"""验证坐标系假设：把 X 与 Y 轴互换后再叠加，看轴向是否变成正确的。

点头应是绕横轴（|X| 最大），摇头应是绕竖直轴（|Y| 最大）。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hm import fit as FM, generate, to_joints
from hm.experiments.probe_rot2 import joint_rots

FM._patch_paths()
HEAD = 15
COL = 3 + (HEAD - 1) * 3

# 候选基变换：都是正交阵，代表不同的坐标约定
BASES = {
    "恒等": np.eye(3),
    "X↔Y": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    "X↔Y 带符号": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    "X↔Z": np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
}


def axis_of(H):
    rv = R.from_matrix(H @ H[0].T).as_rotvec()
    ang = np.degrees(np.linalg.norm(rv, axis=1))
    i = int(np.argmax(ang))
    if ang[i] < 1e-6:
        return np.zeros(3), 0.0
    return rv[i] / np.linalg.norm(rv[i]), ang[i]


def main():
    for key, prompt, frames, model in (
            ("nod", "a person nods their head", 32, "mdm"),
            ("shake", "a person shakes the head", 32, "mdm")):
        print("  === %s（目标：%s）===" % (key, "绕横轴 |X| 最大" if key == "nod" else "绕竖直轴 |Y| 最大"))
        F = generate.generate([prompt], [frames], name=model)[0]
        Rj = joint_rots(F)
        h = Rj[:, HEAD]
        delta = h @ h[0].T                    # 特征里的头部增量
        J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
        _, P = FM.fit(J)
        P = np.asarray(P, dtype=np.float64)
        cur = R.from_rotvec(P[:, COL:COL + 3]).as_matrix()

        for name, B in BASES.items():
            d2 = B @ delta @ B.T
            H = d2 @ cur
            ax, ang = axis_of(H)
            tag = "横轴✓" if (key == "nod" and abs(ax[0]) > max(abs(ax[1]), abs(ax[2]))) else \
                  ("竖直✓" if (key == "shake" and abs(ax[1]) > max(abs(ax[0]), abs(ax[2]))) else "不对")
            print("    %-12s 轴 (%+.2f, %+.2f, %+.2f)  角度 %5.1f 度  %s"
                  % (name, ax[0], ax[1], ax[2], ang, tag))
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
