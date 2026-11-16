"""把参数空间装配的结果渲成视频，并做几何检查。"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import config, criteria_v2, finish_job as FJ, job as jobmod
from hm.assemble_params import joints_of_params

JOB = sys.argv[1] if len(sys.argv) > 1 else "job_long04"
jd = ROOT / "outputs" / JOB
FACES = ROOT / "runs" / "R01-wave-turn-walk" / "faces.npy"

P = np.load(jd / "assembled" / "params.npy")
J, V = joints_of_params(P)
print("  参数 %s  关节 %s" % (P.shape, J.shape))

# 几何检查
ok, hard, met, why = criteria_v2.check(np.asarray(J), "", None, None)
print("  几何检查：%s  %s" % ("通过" if ok else "不通过", hard))
print("  骨长离散 %.2f%%  反向帧占比 %.0f%%  间距 %.3f 米"
      % (100 * (met.get("bone_length_rel_mean_max") or 0),
         100 * (met.get("rev_frac") or 0), met.get("min_separation_m") or 0))
print("  交界：%s" % [(x["between"][1], x["pose_gap_deg"], x["transition_frames"])
                    for x in json.loads((jd / "assembled/bounds.json").read_text(encoding="utf-8"))["junctions"]])

# 渲染
d = jd / "render_params"
d.mkdir(parents=True, exist_ok=True)
np.save(d / "verts.npy", np.asarray(V, dtype=np.float64))
np.save(d / "params.npy", P)
if FACES.exists():
    shutil.copyfile(FACES, d / "faces.npy")
subprocess.run([str(config.BLENDER), "--background", "--python",
                str(ROOT / "hm" / "render_blender.py"), "--", str(d), str(d / "render")],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
mp4 = d / "render" / "out.mp4"
print("  渲染：%s" % ("ok" if mp4.exists() else "FAIL"))
