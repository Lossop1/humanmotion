"""用关节直接算的根朝向指标（髋线 vs 脚尖）重做组合判据，并定阈值。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
OUT = ROOT / "outputs" / "_capsheet"


def hip_vs_toe(J):
    """髋线定的骨盆朝向 与 脚尖方向 的水平点积。返回 (反向帧占比, 最小点积)。"""
    dots = []
    for t in range(len(J)):
        hips = J[t, 2] - J[t, 1]
        hips[1] = 0.0
        p = np.cross(np.array([0.0, 1.0, 0.0]), hips)
        acc = np.zeros(3)
        for a, b in ((7, 10), (8, 11)):
            v = J[t, b] - J[t, a]
            v[1] = 0.0
            n = np.linalg.norm(v)
            if n > 1e-6:
                acc += v / n
        np_, nt = np.linalg.norm(p[[0, 2]]), np.linalg.norm(acc[[0, 2]])
        if np_ < 1e-6 or nt < 1e-6:
            continue
        dots.append(float(np.dot(p[[0, 2]] / np_, acc[[0, 2]] / nt)))
    d = np.array(dots) if dots else np.array([1.0])
    return float((d < 0).mean()), float(d.min())


def evaluate(rows, bone_th, rev_th):
    bad = np.array([r["label"] == "geom" for r in rows])
    pred = np.array([(r["bone_rel_ref"] > bone_th) or (r["rev_frac"] > rev_th) for r in rows])
    tp = int((pred & bad).sum()); fn = int((~pred & bad).sum())
    fp = int((pred & ~bad).sum()); tn = int((~pred & ~bad).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return 0.5 * (sens + spec), dict(tp=tp, fn=fn, fp=fp, tn=tn, sens=sens, spec=spec)


def main():
    rows = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["label"] in ("geom", "pass")]
    for r in rows:
        name = "%s_%s_%d" % (r["action"], r["model"], r["sample"])
        J = np.load(OUT / "joints" / ("%s.npy" % name))
        r["rev_frac"], r["rev_min"] = hip_vs_toe(J)

    print("  关节直接算的根朝向指标")
    geom = [r for r in rows if r["label"] == "geom"]
    good = [r for r in rows if r["label"] == "pass"]
    print("    geom %d 段：反向帧占比 中位 %.2f，最小点积 中位 %.3f"
          % (len(geom), np.median([r["rev_frac"] for r in geom]),
             np.median([r["rev_min"] for r in geom])))
    print("    pass %d 段：反向帧占比 中位 %.2f，最小点积 中位 %.3f"
          % (len(good), np.median([r["rev_frac"] for r in good]),
             np.median([r["rev_min"] for r in good])))

    print("\n  骨长阈值 × 反向帧阈值 的扫描（平衡准确率）")
    best = (0, None, None, None)
    for bt in (0.08, 0.10, 0.103, 0.12, 0.15, 0.20):
        for rt in (0.0, 0.02, 0.05, 0.10):
            bal, cm = evaluate(rows, bt, rt)
            if bal > best[0]:
                best = (bal, bt, rt, cm)
            print("    骨长>%.3f  反向帧>%.2f  →  平衡 %.2f（灵敏度 %.2f，特异度 %.2f）"
                  % (bt, rt, bal, cm["sens"], cm["spec"]))
    bal, bt, rt, cm = best
    print("\n  选定：骨长 > %.3f 或 反向帧占比 > %.2f" % (bt, rt))
    print("    灵敏度 %.2f  特异度 %.2f  平衡准确率 %.2f" % (cm["sens"], cm["spec"], bal))
    print("    混淆：几何失败 %d 拦下 %d 漏 %d；可用 %d 误拦 %d 放行 %d"
          % (cm["tp"] + cm["fn"], cm["tp"], cm["fn"], cm["fp"] + cm["tn"], cm["fp"], cm["tn"]))

    miss = [r for r in rows if r["label"] == "geom"
            and not ((r["bone_rel_ref"] > bt) or (r["rev_frac"] > rt))]
    fp = [r for r in rows if r["label"] == "pass"
          and ((r["bone_rel_ref"] > bt) or (r["rev_frac"] > rt))]
    if miss:
        print("\n  漏掉：")
        for r in miss:
            print("    %s_%s_%d 骨长 %.3f 反向帧 %.2f"
                  % (r["action"], r["model"], r["sample"], r["bone_rel_ref"], r["rev_frac"]))
    if fp:
        print("\n  误拦：")
        for r in fp:
            print("    %s_%s_%d 骨长 %.3f 反向帧 %.2f"
                  % (r["action"], r["model"], r["sample"], r["bone_rel_ref"], r["rev_frac"]))

    chosen = {"bone_th": bt, "rev_th": rt,
              "sens": round(cm["sens"], 3), "spec": round(cm["spec"], 3),
              "bal": round(bal, 3),
              "n_geom": cm["tp"] + cm["fn"], "n_pass": cm["fp"] + cm["tn"],
              "source": "96 段人工标注（2026-10-25），84 段进入二分类"}
    p = ROOT / "outputs" / "_reference" / "criteria_v2.json"
    p.write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n  阈值已写入 %s" % p)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
