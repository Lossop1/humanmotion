"""截尾测量：动作还能认出来之前，最多能截掉多少帧。

问题
    交界差大时，过渡被拉得很长；如果末段其实在「收势」，截掉几帧能让过渡更早开始。
    但截多了动作就认不出来——所以先量截多少帧动作签名还成立。
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import signature

TRIMS = [0, 2, 4, 6, 8, 12, 16]


def main():
    rows = []
    for p in sorted(glob.glob("/home/chuan/motion-work/outputs/big_s20_r9/storyboard.json")):
        import json
        sb = json.loads(Path(p).read_text(encoding="utf-8"))
        for seg in sb["segments"]:
            sel = seg.get("selected")
            if not sel:
                continue
            n = int(str(sel).split("_")[-1])
            jp = Path(p).parent / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
            if not jp.exists():
                continue
            J = np.load(jp)
            prompt = seg.get("text_gen_final") or ""
            print("  %s %-38s 共 %d 帧" % (seg["segment_id"], prompt[:38], len(J)))
            for t in TRIMS:
                if t >= len(J) - 8:
                    break
                K = J[: len(J) - t]
                ok, bad, det = signature.check(prompt, signature.signature(K))
                print("    截 %2d 帧 → 签名 %-6s %s" % (
                    t, "通过" if ok else ("可疑" if ok is False else "无法判"),
                    "" if not bad else str([(b[0], round(det.get(b[0]) or 0, 2)) for b in bad])))
            rows.append((seg["segment_id"], len(J)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
