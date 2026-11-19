"""判据 v5：骨长用「与参考的最大绝对误差（厘米）」，与体型无关。

理由：百分比对小骨头放大过大（锁骨 3 厘米 = 23%），
而插值拉长真正发生在长骨上，绝对误差能拉开量级。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hm import criteria_v2, job as jobmod

OUT = ROOT / "outputs" / "_capsheet"


def bone_err_cm(J, mean):
    L = jobmod.bone_lengths(J).mean(axis=0)
    return float(100 * np.abs(L - mean).max())


def main():
    ref = jobmod.load_bone_reference()
    mean = np.array(ref["mean"])
    labels = json.loads((OUT / "labels.json").read_text(encoding="utf-8"))
    LAB, SAMP = labels["labels"], labels.get("sample_labels", {})
    rows = []
    for f in sorted((OUT / "joints").glob("*.npy")):
        action, model, n = f.stem.rsplit("_", 2)
        lab = SAMP.get(str((action, model, int(n)))) or LAB.get(action, {}).get(model, "?")
        if lab not in ("geom", "pass"):
            continue
        J = np.asarray(np.load(f))
        _, _, met, _ = criteria_v2.check(J, "", None, None)
        rows.append({"label": lab, "err": bone_err_cm(J, mean),
                     "rev": met.get("rev_frac", 0.0),
                     "sep": met.get("min_separation_m", 0.0)})

    bad = np.array([r["label"] == "geom" for r in rows])
    print("  样本 %d（geom %d，pass %d）" % (len(rows), bad.sum(), (~bad).sum()))
    g = [r["err"] for r in rows if r["label"] == "geom"]
    p = [r["err"] for r in rows if r["label"] == "pass"]
    print("  最大绝对骨长误差：geom 中位 %.1f 厘米，pass 中位 %.1f 厘米"
          % (np.median(g), np.median(p)))
    print()
    print("  %-42s %s" % ("判据", "灵敏度 特异度 平衡"))
    for th in (3.0, 4.0, 5.0, 6.0, 8.0):
        for combo in (False, True):
            pred = np.array([(r["err"] > th) or (combo and (r["rev"] > 0 or r["sep"] < 0.045))
                             for r in rows])
            tp = int((pred & bad).sum()); fn = int((~pred & bad).sum())
            fp = int((pred & ~bad).sum()); tn = int((~pred & ~bad).sum())
            sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
            tag = ("骨长>%.0fcm 或 反向帧 或 穿模" % th) if combo else ("仅骨长>%.0fcm" % th)
            print("  %-42s %.2f   %.2f   %.2f   （拦%d 漏%d 误拦%d）"
                  % (tag, sens, spec, 0.5 * (sens + spec), tp, fn, fp))
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
