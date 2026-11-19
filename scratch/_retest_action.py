"""重测单个动作的两个后端：几何通过率 + 躯干前倾。

旧能力表是在老判据下测的，而且一个动作只测一句提示词。
这里用现行判据重测，并同时记录前倾——转身这类动作前倾大了观感就是「弓着腰转身」。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import criteria_v2, generate, to_joints

REFS = Path("/home/chuan/motion-work/outputs/_reference")
ACTION = "turn"
PROMPT = "a person turns around in place"
K = 10
PELVIS, SPINE3 = 0, 9


def lean(J):
    v = J[:, SPINE3] - J[:, PELVIS]
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return float(np.median(np.degrees(np.arccos(np.clip(v[:, 1], -1, 1)))))


def main():
    out = {}
    for model in ("momask", "mdm"):
        feats = generate.generate([PROMPT] * K, [44] * K, name=model)
        oks, leans, reasons = 0, [], []
        for F in feats:
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            ok, hard, met, why = criteria_v2.check(J, PROMPT, None, None)
            oks += 1 if ok else 0
            leans.append(lean(J))
            if not ok and why:
                reasons.append(why[0][:40])
        out[model] = {"passed": oks, "samples": K, "pass_rate": oks / K,
                      "lean_median": float(np.median(leans)),
                      "lean_max": float(max(leans)),
                      "fail_reasons": reasons[:3]}
        print("  %-8s 几何 %2d/%d  前倾中位 %5.1f 度  最大 %5.1f 度  %s"
              % (model, oks, K, np.median(leans), max(leans),
                 "；".join(sorted(set(reasons))[:2])))

    p = REFS / "capability_v4.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    entry = doc["table"].get(ACTION)
    if entry is None:
        print("  能力表里没有 %s 条目，未写入" % ACTION)
        return
    for model, r in out.items():
        entry["models"][model] = {
            "passed": r["passed"], "samples": r["samples"], "pass_rate": r["pass_rate"],
            "status": "support" if r["pass_rate"] >= 0.75 else
                      ("approximate" if r["pass_rate"] >= 0.40 else "unsupported"),
            "lean_median_deg": round(r["lean_median"], 1),
            "prompt": PROMPT, "retested": "现行判据 criteria_v5",
        }
    best = max(out, key=lambda m: out[m]["pass_rate"])
    entry["best"] = best
    entry["prompt"] = PROMPT
    entry["prompt_scope"] = "simple"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  已写回能力表：最佳后端 %s" % best)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
