"""只重画联络图：从已存关节读，不重新生成。

修两处：
  1. store 键是 1..K，原代码用 0..K-1 取值，导致样本 1 空白、标题错位
  2. 远端 matplotlib 无中文字体，标题改英文，中文只留在文件名
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hm import job as jobmod
from hm.capability import ACTIONS

ROOT = Path("/home/chuan/motion-work")
OUT = ROOT / "outputs" / "_capsheet"
BONES = jobmod.BONES
MODELS = ("momask", "mdm")
K = 3
FRACS = (0.15, 0.5, 0.85)


def draw(ax, J, color):
    ax.clear()
    for c, p in BONES:
        ax.plot([J[p, 0], J[c, 0]], [J[p, 1], J[c, 1]], "-", lw=2.4, color=color,
                solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(J[:, 0].min() - 0.30, J[:, 0].max() + 0.30)
    ax.set_ylim(J[:, 1].min() - 0.12, J[:, 1].max() + 0.12)


def sheet(key, zh, store):
    fig, axes = plt.subplots(len(MODELS) * 2, K * len(FRACS),
                             figsize=(K * len(FRACS) * 1.5, len(MODELS) * 2 * 1.7))
    for mi, model in enumerate(MODELS):
        for ki in range(K):
            J = store.get((model, ki + 1))          # 修正：键从 1 开始
            for fi, frac in enumerate(FRACS):
                for row in (0, 1):
                    ax = axes[mi * 2 + row, ki * len(FRACS) + fi]
                    if J is None:
                        ax.axis("off")
                        continue
                    t = int(frac * (len(J) - 1))
                    Jf = J[t] if row == 0 else J[t][:, [2, 1, 0]]
                    draw(ax, Jf, "#1f4e79" if model == "momask" else "#a0522d")
                    if row == 0:
                        ax.set_title("%s #%d  %d%%" % (model, ki + 1, int(frac * 100)),
                                     fontsize=8)
    fig.suptitle("%s   rows: momask(front/side), mdm(front/side)   cols: 3 samples x 3 keyframes"
                 % key, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / ("sheet_%s.png" % key)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main():
    for key, zh, prompt, kws, frames in ACTIONS:
        store = {}
        for model in MODELS:
            for i in range(1, K + 1):
                f = OUT / "joints" / ("%s_%s_%d.npy" % (key, model, i))
                if f.exists():
                    store[(model, i)] = np.load(f)
        p = sheet(key, zh, store)
        print("  %s  %d 段已画" % (p.name, len(store)), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
