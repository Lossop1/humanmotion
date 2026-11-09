"""测各环节耗时：生成、特征还原、拟合、渲染。"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import config, fit as FM, generate, to_joints

FM._patch_paths()
OUT = ROOT / "outputs" / "_speed"
OUT.mkdir(parents=True, exist_ok=True)
SEG = ROOT / "outputs" / "job_p01" / "segments" / "seg_001" / "candidate_1" / "joints22.npy"


def tic():
    return time.time()


print("  一、生成（MoMask，40 帧，1 条）")
t = tic()
F = generate.generate(["a person walks forward"], [40], name="momask")[0]
t_gen = time.time() - t
print("     %.1f 秒" % t_gen)

print("  二、263 维 → 22 关节")
t = tic()
J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
t_dec = time.time() - t
print("     %.2f 秒（%d 帧）" % (t_dec, len(J)))

print("  三、SMPL 拟合（40 帧）")
J2 = np.load(SEG)
t = tic()
V, P = FM.fit(J2)
t_fit = time.time() - t
print("     %.1f 秒（%d 帧）" % (t_fit, len(J2)))

print("  四、Blender 渲染（40 帧）")
d = OUT / "seg"
d.mkdir(exist_ok=True)
np.save(d / "verts.npy", np.asarray(V, dtype=np.float64))
np.save(d / "params.npy", np.asarray(P, dtype=np.float64))
faces = ROOT / "runs" / "R01-wave-turn-walk" / "faces.npy"
if faces.exists():
    import shutil
    shutil.copyfile(faces, d / "faces.npy")
t = tic()
subprocess.run([str(config.BLENDER), "--background", "--python",
                str(ROOT / "hm" / "render_blender.py"), "--", str(d), str(d / "r")],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t_rnd = time.time() - t
print("     %.1f 秒" % t_rnd)

print()
print("  ==== 折算到一条 5 段、约 250 帧的句子 ====")
print("    生成 5 段        %5.1f 秒" % (t_gen * 5))
print("    特征还原 5 次     %5.1f 秒" % (t_dec * 5))
print("    拟合 5 段         %5.1f 秒" % (t_fit * 5))
print("    渲染 250 帧       %5.1f 秒" % (t_rnd * 250 / 40))
tot = t_gen * 5 + t_dec * 5 + t_fit * 5 + t_rnd * 250 / 40
print("    合计             %5.1f 秒（约 %.1f 分钟）" % (tot, tot / 60))
print()
print("    占比：生成 %.0f%%，还原 %.0f%%，拟合 %.0f%%，渲染 %.0f%%" % (
    100 * t_gen * 5 / tot, 100 * t_dec * 5 / tot,
    100 * t_fit * 5 / tot, 100 * t_rnd * 250 / 40 / tot))
