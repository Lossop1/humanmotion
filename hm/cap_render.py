"""单段：关节 → SMPL 拟合 → 渲成 mp4。

供 xargs 并行调用：
    ls outputs/_capsheet/joints/*.npy | xargs -P 3 -I{} python -m hm.cap_render {}
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import config, fit as FM

FM._patch_paths()
FACES = ROOT / "runs" / "R01-wave-turn-walk" / "faces.npy"


def main():
    src = Path(sys.argv[1])
    J = np.load(src)
    V, P = FM.fit(J)
    d = ROOT / "outputs" / "_capsheet" / "render" / src.stem
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "verts.npy", np.asarray(V, dtype=np.float64))
    np.save(d / "params.npy", np.asarray(P, dtype=np.float64))
    if FACES.exists():
        shutil.copyfile(FACES, d / "faces.npy")
    subprocess.run([str(config.BLENDER), "--background", "--python",
                    str(ROOT / "hm" / "render_blender.py"), "--", str(d), str(d / "render")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    mp4 = d / "render" / "out.mp4"
    print("%s %s" % (src.stem, "ok" if mp4.exists() else "FAIL"), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
