"""查头部旋转的轴向：点头应是绕左右横轴，摇头应是绕竖直轴。

同时对比：拟合出来的头部世界朝向 vs 特征里隐含的头部世界朝向。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hm.experiments.probe_rot2 import joint_rots

OUT = ROOT / "outputs" / "_headfix"
HEAD, NECK = 15, 12
COL = 3 + (HEAD - 1) * 3


def axis_report(M, tag):
    """M: (T,3,3) 相对首帧的旋转。输出转动轴与角度。"""
    rel = np.transpose(M @ np.transpose(M[0], (1, 0))[None], (0, 1, 2))
    rv = R.from_matrix(M @ M[0].T).as_rotvec()
    ang = np.degrees(np.linalg.norm(rv, axis=1))
    i = int(np.argmax(ang))
    ax = rv[i] / max(np.linalg.norm(rv[i]), 1e-9)
    print("    %-10s 最大转动 %.1f 度，轴 (%.2f, %.2f, %.2f)"
          % (tag, ang[i], ax[0], ax[1], ax[2]))
    return ax, ang[i]


def main():
    for key in ("nod", "shake"):
        print("  === %s ===" % key)
        for stage in ("before", "after"):
            d = OUT / ("%s_%s" % (key, stage))
            if not (d / "params.npy").exists():
                continue
            P = np.load(d / "params.npy")
            Hf = R.from_rotvec(P[:, COL:COL + 3]).as_matrix()
            axis_report(Hf, stage + "（拟合参数）")
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
