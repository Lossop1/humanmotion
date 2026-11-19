"""在人工标签上找每个指标的最佳阈值，并给出组合判据的混淆矩阵。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "_capsheet"


def best_threshold(vals, labels, higher_is_bad):
    """在候选阈值上扫，取平衡准确率最高者。"""
    order = np.unique(np.round(vals, 6))
    best = (0.5, None, 0.0)
    for th in order:
        if higher_is_bad:
            pred = vals > th
        else:
            pred = vals < th
        bad = (labels == "geom")
        tp = int((pred & bad).sum())
        fn = int((~pred & bad).sum())
        fp = int((pred & ~bad).sum())
        tn = int((~pred & ~bad).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        bal = 0.5 * (sens + spec)
        if bal > best[0]:
            best = (bal, float(th), {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                                     "sens": sens, "spec": spec})
    return best


def main():
    rows = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["label"] in ("geom", "pass")]
    labels = np.array([r["label"] for r in rows])

    specs = [
        ("bone_rel_ref", True, "骨长均值相对偏差"),
        ("bone_jitter", True, "逐帧骨长抖动"),
        ("r3_tilt_max", True, "躯干最大倾角"),
        ("r4_neg_frac", True, "根朝向与脚尖反向帧占比"),
        ("min_sep_m", False, "非相邻关节最小间距"),
        ("twist_p95", True, "骨向变化 p95"),
    ]
    print("  在 %d 段已标注样本上扫阈值（geom 失败 %d，pass %d）"
          % (len(rows), int((labels == "geom").sum()), int((labels == "pass").sum())))
    print()
    print("  %-22s %10s %8s %8s %8s" % ("指标", "阈值", "灵敏度", "特异度", "平衡准确率"))
    chosen = {}
    for key, hib, zh in specs:
        vals = np.array([r.get(key) if r.get(key) is not None else 0.0 for r in rows])
        bal, th, cm = best_threshold(vals, labels, hib)
        chosen[key] = (th, hib)
        print("  %-22s %10.4f %8.2f %8.2f %8.2f" % (zh, th, cm["sens"], cm["spec"], bal))
        if key == "r4_neg_frac":
            print("        （阈值取 0 时即为「只要有反向帧就判失败」：灵敏度 %.2f，特异度 %.2f）"
                  % (best_threshold(vals, labels, True)[2]["sens"] if th else cm["sens"],
                     best_threshold(vals, labels, True)[2]["spec"] if th else cm["spec"]))

    # 组合判据：骨长 + R4
    print()
    bth, _ = chosen["bone_rel_ref"]
    print("  组合判据：骨长偏差 > %.3f  或  存在反向帧（r4_neg_frac > 0）" % bth)
    pred = []
    for r in rows:
        bad = (r.get("bone_rel_ref") or 0) > bth
        bad = bad or (r.get("r4_neg_frac") or 0) > 0
        pred.append(bad)
    pred = np.array(pred)
    bad = labels == "geom"
    tp = int((pred & bad).sum()); fn = int((~pred & bad).sum())
    fp = int((pred & ~bad).sum()); tn = int((~pred & ~bad).sum())
    print("           几何失败 %d 段 → 拦下 %d，漏掉 %d" % (bad.sum(), tp, fn))
    print("           可用     %d 段 → 误拦 %d，放行 %d" % ((~bad).sum(), fp, tn))
    print("           灵敏度 %.2f  特异度 %.2f  平衡准确率 %.2f"
          % (tp / max(tp + fn, 1), tn / max(tn + fp, 1),
             0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))))

    # 漏掉的都是什么
    miss = [rows[i] for i in range(len(rows)) if bad[i] and not pred[i]]
    if miss:
        print("\n  漏掉的：")
        for m in miss:
            print("    %s_%s_%d  %s  骨长 %.3f  反向帧 %.2f"
                  % (m["action"], m["model"], m["sample"], m["label"],
                     m.get("bone_rel_ref") or 0, m.get("r4_neg_frac") or 0))
    wrong = [rows[i] for i in range(len(rows)) if (not bad[i]) and pred[i]]
    if wrong:
        print("\n  误拦的：")
        for m in wrong:
            print("    %s_%s_%d  骨长 %.3f  反向帧 %.2f"
                  % (m["action"], m["model"], m["sample"],
                     m.get("bone_rel_ref") or 0, m.get("r4_neg_frac") or 0))

    # 语义失败是否被抓到
    allrows = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    sem = [r for r in allrows if r["label"] == "sem"]
    print("\n  语义失败（几何判据能否抓到）：")
    for r in sem:
        caught = ((r.get("bone_rel_ref") or 0) > bth) or ((r.get("r4_neg_frac") or 0) > 0)
        print("    %s_%s_%d  骨长 %.3f  反向帧 %.2f  → %s"
              % (r["action"], r["model"], r["sample"],
                 r.get("bone_rel_ref") or 0, r.get("r4_neg_frac") or 0,
                 "误判为失败" if caught else "抓不到（几何上正常）"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
