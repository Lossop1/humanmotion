"""把待核的几段画成骨架关键帧条，便于逐帧确认到底做的是什么动作。"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm.job import BONES

WORK = ROOT / "outputs" / "_semcheck" / "work"
OUT = ROOT / "outputs" / "_semcheck" / "strips"
OUT.mkdir(parents=True, exist_ok=True)

CASES = ["shake_head_mdm", "nod_mdm", "hands_hips_momask", "clap_momask"]


def draw(ax, J, color="#1f4e79"):
    ax.clear()
    for c, p in BONES:
        ax.plot([J[p, 0], J[c, 0]], [J[p, 1], J[c, 1]], "-", lw=2.4, color=color,
                solid_capstyle="round")
    ax.plot(J[15, 0], J[15, 1], "o", ms=5, color="#c0392b")   # 头
    for w in (20, 21):
        ax.plot(J[w, 0], J[w, 1], "o", ms=5, color="#27ae60")  # 双腕
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(J[:, 0].min() - 0.25, J[:, 0].max() + 0.25)
    ax.set_ylim(0.0, 2.0)


def main():
    for name in CASES:
        d = WORK / name
        if not (d / "joints.npy").exists():
            print("  缺 %s" % name)
            continue
        J = np.load(d / "joints.npy")
        T = len(J)
        fig, axes = plt.subplots(2, 6, figsize=(11, 6.4))
        for i, frac in enumerate(np.linspace(0.05, 0.95, 6)):
            t = int(frac * (T - 1))
            draw(axes[0, i], J[t])
            axes[0, i].set_title("第 %d 帧 (%d%%)" % (t + 1, int(frac * 100)), fontsize=8)
            # 俯视：看头/手在水平面的关系
            ax = axes[1, i]
            ax.clear()
            for c, p in BONES:
                ax.plot([J[t, p, 0], J[t, c, 0]], [J[t, p, 2], J[t, c, 2]], "-",
                        lw=2.2, color="#a0522d", solid_capstyle="round")
            ax.plot(J[t, 15, 0], J[t, 15, 2], "o", ms=6, color="#c0392b")
            for w in (20, 21):
                ax.plot(J[t, w, 0], J[t, w, 2], "o", ms=6, color="#27ae60")
            ax.set_aspect("equal")
            ax.axis("off")
            r = 0.9
            ax.set_xlim(J[t, 0, 0] - r, J[t, 0, 0] + r)
            ax.set_ylim(J[t, 0, 2] - r, J[t, 0, 2] + r)
        fig.suptitle("%s   上排正面  下排俯视（红点=头，绿点=双腕）" % name, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = OUT / ("%s.png" % name)
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print("  %s" % p.name, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
