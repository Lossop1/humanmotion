"""语义验证覆盖率：30 句里到底有多少段被真正检查过。

输出
    1 段级汇总：可判且通过 / 可判但可疑 / 无法判（无校准量）
    2 无法判的段按提示词归类，看下一步该补哪类规则
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import signature

ROUND = sys.argv[1] if len(sys.argv) > 1 else "r15"


def main():
    tot = passed = suspect = unknown = 0
    unknown_prompts = Counter()
    rows = []
    for p in sorted(glob.glob("/home/chuan/motion-work/outputs/big_s*_%s/storyboard.json" % ROUND)):
        d = Path(p).parent
        for seg in json.loads(Path(p).read_text(encoding="utf-8"))["segments"]:
            sel = seg.get("selected")
            if not sel:
                continue
            n = int(str(sel).split("_")[-1])
            jp = d / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
            if not jp.exists():
                continue
            prompt = seg.get("text_gen_final") or ""
            ok, bad, det = signature.check(prompt, signature.signature(np.load(jp)))
            tot += 1
            if ok is None:
                unknown += 1
                unknown_prompts[prompt[:38]] += 1
            elif ok:
                passed += 1
            else:
                suspect += 1
                rows.append((d.name[:11], seg["segment_id"], prompt[:40], bad))
    print("  %s：共 %d 段" % (ROUND, tot))
    print("    可判且通过  %d（%.0f%%）" % (passed, 100.0 * passed / max(tot, 1)))
    print("    可判但可疑  %d（%.0f%%）" % (suspect, 100.0 * suspect / max(tot, 1)))
    print("    无法判      %d（%.0f%%）" % (unknown, 100.0 * unknown / max(tot, 1)))
    print()
    print("  可疑明细：")
    for r in rows:
        print("    %-11s %s %-40s %s" % r)
    print()
    print("  无法判的提示词（前 10）：")
    for prompt, c in unknown_prompts.most_common(10):
        print("    %2d 次  %s" % (c, prompt))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
