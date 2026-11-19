"""对已有任务批量跑自检门限，输出判定表。"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/chuan/motion-work")
from hm import fit as FM
from hm.gates import gates

FM._patch_paths()
m = FM._load_smpl("cuda")

LIMITS = {"rot_med": 5.0, "hf_ratio": 3.0, "accel_max": 0.20}


def check(jobdir):
    p = Path(jobdir) / "assembled" / "params.npy"
    j = Path(jobdir) / "assembled" / "joints22.npy"
    if not (p.exists() and j.exists()):
        return None
    P = np.load(p)
    J = np.asarray(np.load(j), dtype=np.float64)
    with torch.no_grad():
        t = torch.from_numpy(P).float().cuda()
        V = m(betas=t[:, 72:82], global_orient=t[:, 0:3], body_pose=t[:, 3:72],
              transl=t[:, 82:85], return_verts=True).vertices.cpu().numpy()
    r = gates(np.asarray(V, np.float64), J, P)
    bad = [k for k, lim in LIMITS.items() if r[k] > lim]
    return r, bad


rows = []
for d in sorted(glob.glob("/home/chuan/motion-work/outputs/eval_s*/assembled")):
    job = d.split("/")[-2]
    res = check(Path(d).parent)
    if res is None:
        continue
    r, bad = res
    rows.append((job, r, bad))
    print("  %-10s rot中位 %6.2f  hf_ratio %6.2f  accel_max %.3f  %s"
          % (job, r["rot_med"], r["hf_ratio"], r["accel_max"],
             "通过" if not bad else "不通过：" + ",".join(bad)), flush=True)

ok = sum(1 for _, _, b in rows if not b)
print("\n  自检通过 %d/%d" % (ok, len(rows)))
