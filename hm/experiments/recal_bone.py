"""骨长判据改口径：从「与参考受试者的绝对差异」改成「序列内的波动」。

绝对差异对拟合出来的体型没有意义（同一个人的骨长可以比参考样本长 20%），
真正要抓的是「插值把骨头拉长」——那是序列内的波动。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import job as jobmod

OUT = ROOT / "outputs" / "_capsheet"


def fluct(J):
    """序列内骨长波动：逐骨 (std/mean) 的最大值，无量纲。"""
    L = jobmod.bone_lengths(J)
    m = L.mean(axis=0)
    return float((L.std(axis=0) / np.maximum(m, 1e-9)).max())


def abs_dev(J, ref_mean):
    L = jobmod.bone_lengths(J)
    return float((np.abs(L.mean(axis=0) - ref_mean) / ref_mean).max())


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def best_threshold(vals, bad, higher_is_bad=True):
    best = (0.0, None, None)
    for th in np.unique(np.round(vals, 6)):
        pred = vals > th if higher_is_bad else vals < th
        tp = int((pred & bad).sum()); fn = int((~pred & bad).sum())
        fp = int((pred & ~bad).sum()); tn = int((~pred & ~bad).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        bal = 0.5 * (sens + spec)
        if bal > best[0]:
            best = (bal, float(th), dict(tp=tp, fn=fn, fp=fp, tn=tn, sens=sens, spec=spec))
    return best


def main():
    ref = jobmod.load_bone_reference()
    ref_mean = np.array(ref["mean"])
    labels = json.loads((OUT / "labels.json").read_text(encoding="utf-8"))
    LAB = labels["labels"]
    SAMP = labels.get("sample_labels", {})

    rows = []
    for f in sorted((OUT / "joints").glob("*.npy")):
        name = f.stem
        action, model, n = name.rsplit("_", 2)
        key = str((action, model, int(n)))
        lab = SAMP.get(key)
        if lab is None:
            lab = LAB.get(action, {}).get(model, "unlabeled")
        if lab not in ("geom", "pass"):
            continue
        J = np.load(f)
        rows.append({"name": name, "label": lab,
                     "fluct": fluct(J), "abs": abs_dev(J, ref_mean)})

    bad = np.array([r["label"] == "geom" for r in rows])
    print("  样本 %d：geom %d，pass %d" % (len(rows), bad.sum(), (~bad).sum()))
    print()

    for key, name in (("abs", "旧口径：与参考受试者的绝对差异"),
                      ("fluct", "新口径：序列内骨长波动")):
        vals = np.array([r[key] for r in rows])
        a = auc([r[key] for r in rows if r["label"] == "pass"],
                [r[key] for r in rows if r["label"] == "geom"])
        bal, th, cm = best_threshold(vals, bad)
        print("  %s" % name)
        print("    geom 中位 %.4f，pass 中位 %.4f，AUC(越低越好区分) %.2f"
              % (np.median([r[key] for r in rows if r["label"] == "geom"]),
                 np.median([r[key] for r in rows if r["label"] == "pass"]), a))
        print("    最佳阈值 %.4f → 灵敏度 %.2f 特异度 %.2f 平衡 %.2f"
              % (th, cm["sens"], cm["spec"], bal))
        print()

    # 组合：新口径 + 反向帧（反向帧沿用之前的结论）
    print("  注意：反向帧判据（髋线 vs 脚尖）不受影响，仍按之前的灵敏度 0.79 / 特异度 1.00。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
