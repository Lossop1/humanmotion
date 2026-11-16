"""为语义核查渲染 44 段视频：每个动作用它在能力表里最好的后端，各 1 段。

产出：
    outputs/_semcheck/videos/<动作>_<后端>.mp4
    outputs/_semcheck/checklist.csv      供人工逐条判「动作是否真的做出来」
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import config, fit as FM, generate, to_joints
from hm.actions import ACTIONS_ALL

FM._patch_paths()
OUT = ROOT / "outputs" / "_semcheck"
(OUT / "videos").mkdir(parents=True, exist_ok=True)
(OUT / "work").mkdir(parents=True, exist_ok=True)
FACES = ROOT / "runs" / "R01-wave-turn-walk" / "faces.npy"

cap = json.loads((ROOT / "outputs/_reference/capability_v4.json").read_text(encoding="utf-8"))["table"]


def main():
    rows = []
    for key, zh, prompt, frames in ACTIONS_ALL:
        v = cap.get(key)
        if not v:
            continue
        if v["status"] not in ("support", "approx"):
            continue
        model = v["best"]
        passed = v["models"][model]["passed"]
        print("  %-10s %-6s %s 通过 %d/10" % (zh, model, prompt[:40], passed), flush=True)
        F = generate.generate([prompt], [frames], name=model)[0]
        J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
        d = OUT / "work" / ("%s_%s" % (key, model))
        d.mkdir(parents=True, exist_ok=True)
        V, P = FM.fit(J)
        np.save(d / "verts.npy", np.asarray(V, dtype=np.float64))
        np.save(d / "params.npy", np.asarray(P, dtype=np.float64))
        np.save(d / "joints.npy", J)
        if FACES.exists():
            shutil.copyfile(FACES, d / "faces.npy")
        subprocess.run([str(config.BLENDER), "--background", "--python",
                        str(ROOT / "hm" / "render_blender.py"), "--", str(d), str(d / "render")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mp4 = d / "render" / "out.mp4"
        if mp4.exists():
            shutil.copyfile(mp4, OUT / "videos" / ("%s_%s.mp4" % (key, model)))
        rows.append({"file": "%s_%s.mp4" % (key, model), "动作": zh, "键": key,
                     "后端": model, "英文提示词": prompt,
                     "几何通过": "%d/10" % passed,
                     "人判(是/否/部分)": "", "备注": ""})

    with open(OUT / "checklist.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "动作", "键", "后端", "英文提示词",
                                          "几何通过", "人判(是/否/部分)", "备注"])
        w.writeheader()
        w.writerows(rows)
    print("\n  共 %d 段，清单 %s" % (len(rows), OUT / "checklist.csv"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
