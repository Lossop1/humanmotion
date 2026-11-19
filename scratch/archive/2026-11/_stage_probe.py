"""出拳段的环节隔离：原始候选 vs 装配后。

三份东西
    A 原始   seg_002 选中候选的 joints22，单独拟合、单独渲染（不经装配、不加过渡）
    B 装配后 最终成片里同一段的帧
    C 差值   装配后关节 vs 原始关节，逐帧最大偏差（看装配是否改动了这段）
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/chuan/motion-work")
from hm import config, fit as FM
from hm.job import BONE_NAMES

JOB = Path("/home/chuan/motion-work/outputs/job_bank4")
OUT = JOB / "stage_probe"
OUT.mkdir(exist_ok=True)


def main():
    sb = json.loads((JOB / "storyboard.json").read_text(encoding="utf-8"))
    seg = [s for s in sb["segments"] if "punch" in (s.get("text_gen_final") or "")][0]
    n = int(str(seg["selected"]).split("_")[-1])
    J = np.load(JOB / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy")
    print("  段 %s 候选 %d，原始 %d 帧" % (seg["segment_id"], n, len(J)))

    # A：单独拟合 + 单独渲染
    FM._patch_paths()
    V, P = FM.fit(J, method="ik")
    np.save(OUT / "verts.npy", np.asarray(V, dtype=np.float64))
    np.save(OUT / "params.npy", np.asarray(P, dtype=np.float64))
    faces = Path("/home/chuan/motion-work/outputs/job_bank4/render/faces.npy")
    if faces.exists():
        import shutil
        shutil.copyfile(faces, OUT / "faces.npy")
    subprocess.run([str(config.BLENDER), "--background", "--python",
                    str(Path("/home/chuan/motion-work/hm/render_blender.py")), "--",
                    str(OUT), str(OUT / "r"), "video", "0.68", "full", "1.0", "fast"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  A 原始渲染：%s" % (OUT / "r" / "out.mp4"))

    # C：装配后关节与原始关节差多少
    bj = json.loads((JOB / "assembled" / "bounds.json").read_text(encoding="utf-8"))
    seq = np.load(JOB / "assembled" / "joints22.npy")
    t = 0
    span = None
    for i, b in enumerate(bj["segments"]):
        if b["segment_id"] == seg["segment_id"]:
            span = (t, t + b["frames"])
            break
        t += b["frames"]
        if i < len(bj.get("junctions", [])):
            t += bj["junctions"][i]["transition_frames"]
    a, z = span
    print("  装配后同段位于帧 %d–%d" % (a, z))
    m = min(z - a, len(J))
    diff = np.linalg.norm(seq[a:a + m] - J[:m], axis=2)
    print("  逐帧最大关节偏差：中位 %.1f 厘米，最大 %.1f 厘米" % (
        100 * float(np.median(diff)), 100 * float(diff.max())))
    # 哪个关节差最多
    per_joint = diff.mean(axis=0) * 100
    for k in np.argsort(-per_joint)[:3]:
        print("    %s 平均 %.1f 厘米" % (BONE_NAMES[k] if k < len(BONE_NAMES) else k, per_joint[k]))

    # 动作幅度：手腕相对肩的位移范围（判断手臂到底有没有挥出去）
    def reach(X):
        w = X[:, 21, :] - X[:, 16, :]
        return float(np.linalg.norm(w, axis=1).max())
    print("  手腕相对肩的最大伸展：原始 %.2f 米，装配后 %.2f 米"
          % (reach(J), reach(seq[a:a + m])))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
