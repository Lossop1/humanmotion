"""组合判据 v4：波动做硬检查，反向帧、穿模保留；绝对差异降为软指标。

在 84 段标注样本上评估。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hm import criteria_v2, job as jobmod

OUT = ROOT / "outputs" / "_capsheet"


def fluct(J):
    L = jobmod.bone_lengths(J)
    m = L.mean(axis=0)
    return float((L.std(axis=0) / np.maximum(m, 1e-9)).max())


def main():
    labels = json.loads((OUT / "labels.json").read_text(encoding="utf-8"))
    LAB, SAMP = labels["labels"], labels.get("sample_labels", {})
    rows = []
    for f in sorted((OUT / "joints").glob("*.npy")):
        action, model, n = f.stem.rsplit("_", 2)
        lab = SAMP.get(str((action, model, int(n)))) or LAB.get(action, {}).get(model, "?")
        if lab not in ("geom", "pass"):
            continue
        J = np.load(f)
        ok, hard, met, why = criteria_v2.check(np.asarray(J), "", None, None)
        rows.append({"label": lab, "fluct": fluct(J),
                     "rev": met.get("rev_frac", 0.0),
                     "sep": met.get("min_separation_m", 0.0),
                     "abs": met.get("bone_rel_mean_max", 0.0)})

    bad = np.array([r["label"] == "geom" for r in rows])
    print("  样本 %d（geom %d，pass %d）" % (len(rows), bad.sum(), (~bad).sum()))
    print()

    def evaluate(fn, tag):
        pred = np.array([fn(r) for r in rows])
        tp = int((pred & bad).sum()); fn_ = int((~pred & bad).sum())
        fp = int((pred & ~bad).sum()); tn = int((~pred & ~bad).sum())
        sens = tp / max(tp + fn_, 1); spec = tn / max(tn + fp, 1)
        print("  %-40s 灵敏度 %.2f 特异度 %.2f 平衡 %.2f  （拦 %d 漏 %d 误拦 %d）"
              % (tag, sens, spec, 0.5 * (sens + spec), tp, fn_, fp))
        return sens, spec

    for th in (0.06, 0.0825, 0.10):
        evaluate(lambda r, th=th: r["fluct"] > th, "仅波动 > %.4f" % th)
    print()
    for th in (0.06, 0.0825, 0.10):
        evaluate(lambda r, th=th: r["fluct"] > th or r["rev"] > 0 or r["sep"] < 0.045,
                 "波动>%.4f 或 反向帧 或 穿模" % th)
    print()
    evaluate(lambda r: r["rev"] > 0 or r["sep"] < 0.045, "仅 反向帧 或 穿模")
    print()
    evaluate(lambda r: r["abs"] > 0.103 or r["rev"] > 0 or r["sep"] < 0.045,
             "旧版：绝对差异>10.3% 或 反向帧 或 穿模")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
