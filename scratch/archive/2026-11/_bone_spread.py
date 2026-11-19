"""量各段骨架的骨长是否一致：如果生成侧本身不一致，拟合并死体型就必然对不齐。"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm.job import BONES, BONE_NAMES

job = Path("/home/chuan/motion-work/outputs/job_shape")
sb = json.loads((job / "storyboard.json").read_text(encoding="utf-8"))
rows = []
for s in sb["segments"]:
    n = int(str(s.get("selected", "cand_01")).split("_")[-1])
    J = np.load(job / "segments" / s["segment_id"] / ("candidate_%d" % n) / "joints22.npy")
    L = np.stack([np.median(np.linalg.norm(J[:, c] - J[:, p], axis=1)) for c, p in BONES])
    rows.append((s["segment_id"], L))

M = np.stack([r[1] for r in rows])
ref = M[0]
print("  各段骨长相对第一段的最大偏差（厘米）：")
for (sid, L) in rows:
    d = np.abs(L - ref) * 100
    worst = int(np.argmax(d))
    print("    %s  平均 %4.1f  最大 %4.1f（%s）" % (sid, d.mean(), d.max(), BONE_NAMES[worst]))
print()
print("  全段逐骨离散度（同一根骨在六段之间的标准差，厘米）：")
sd = M.std(axis=0) * 100
print("    平均 %.1f  最大 %.1f（%s）" % (sd.mean(), sd.max(), BONE_NAMES[int(np.argmax(sd))]))
