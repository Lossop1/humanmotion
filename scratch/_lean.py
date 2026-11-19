"""转身段为什么怪：量躯干前倾。

从截图看，转身时人明显前倾弯腰。这里量一下：
    躯干轴（骨盆→脊柱3）与竖直方向的夹角，逐帧取值，报中位与最大。
对 MDM 与 MoMask 各 6 个候选做对照。
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import generate, to_joints

PELVIS, SPINE3 = 0, 9


def lean(J):
    v = J[:, SPINE3] - J[:, PELVIS]
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip(v[:, 1], -1, 1)))
    return float(np.median(ang)), float(ang.max())


print("  == 现有 S8 转身段的逐候选（MDM）==")
jd = Path("/home/chuan/motion-work/outputs/eval_s8")
for d in sorted(glob.glob(str(jd / "segments" / "seg_005" / "candidate_*" / "joints22.npy"))):
    J = np.load(d)
    med, mx = lean(J)
    m = json.loads((Path(d).parent / "metrics.json").read_text(encoding="utf-8"))
    print("    %s  中位前倾 %5.1f 度  最大 %5.1f 度  判据 %s"
          % (Path(d).parent.name, med, mx, m.get("decision")))

print()
print("  == 同样提示词、MoMask 六个候选 ==")
feats = generate.generate(["a person turns around in place"] * 6, [40] * 6, name="momask")
for i, F in enumerate(feats, 1):
    J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
    med, mx = lean(J)
    print("    cand_%02d  中位前倾 %5.1f 度  最大 %5.1f 度" % (i, med, mx))

print()
print("  == 对照：走路与站立的正常前倾范围 ==")
for prompt in ("a person walks forward", "a person stands still"):
    F = generate.generate([prompt] * 3, [40] * 3, name="momask")
    vals = [lean(np.asarray(to_joints.from_feats(f), dtype=np.float64)) for f in F]
    print("    %-24s 中位 %s" % (prompt, ["%.1f" % v[0] for v in vals]))
