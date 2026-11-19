"""对一轮评测的每一段跑动作签名，列出「看起来没做出来」的段（只作参考，不拦截）。"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import signature

ROOT = Path("/home/chuan/motion-work/outputs")
rows = []
for p in sorted(glob.glob(str(ROOT / "eval_s*" / "storyboard.json"))):
    jd = Path(p).parent
    sb = json.loads(Path(p).read_text(encoding="utf-8"))
    for seg in sb["segments"]:
        sel = seg.get("selected")
        if not sel:
            rows.append((jd.name, seg["segment_id"], seg.get("text_gen_final") or "", "未装配", None))
            continue
        n = int(str(sel).split("_")[-1])
        jp = jd / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
        if not jp.exists():
            continue
        J = np.load(jp)
        prompt = seg.get("text_gen_final") or ""
        ok, bad, det = signature.check(prompt, signature.signature(J))
        rows.append((jd.name, seg["segment_id"], prompt,
                     "无可判" if ok is None else ("通过" if ok else "可疑"), bad or det))

for r in rows:
    print("  %-10s %s %-46s %-5s %s" % (r[0], r[1], r[2][:46], r[3],
                                        "" if r[4] is None or r[3] in ("通过", "未装配") else r[4]))
bad = [r for r in rows if r[3] == "可疑"]
print()
print("  共 %d 段：通过 %d，可疑 %d，无法判 %d" % (
    len(rows), sum(1 for r in rows if r[3] == "通过"), len(bad),
    sum(1 for r in rows if r[3] == "无可判")))
