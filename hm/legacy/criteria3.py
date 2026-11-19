"""在「先保证不漏坏动作」的标准下找工作点。

目标：灵敏度（拦下几何失败的比率）拉到最高，然后在这个前提下最大化特异度。
扫描四个条件的不同组合：
    骨长偏差 > t1
    反向帧占比 > t2
    非相邻关节最小间距 < t3      （穿模近似）
    躯干最大倾角 > t4
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "_capsheet"


def hip_vs_toe(J):
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
    return float((d < 0).mean())


def min_sep(J):
    best = 9.0
    for t in range(len(J)):
        p = J[t]
        for a in range(22):
            for b in range(a + 3, 22):
                d = float(np.linalg.norm(p[a] - p[b]))
                if d < best:
                    best = d
    return best


def load_rows():
    rows = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["label"] in ("geom", "pass")]
    for r in rows:
        name = "%s_%s_%d" % (r["action"], r["model"], r["sample"])
        J = np.load(OUT / "joints" / ("%s.npy" % name))
        r["rev"] = hip_vs_toe(J)
        r["sep"] = min_sep(J)
    return rows


def score(rows, t1, t2, t3, t4):
    bad = np.array([r["label"] == "geom" for r in rows])
    pred = np.array([
        (r["bone_rel_ref"] > t1) or (r["rev"] > t2)
        or (r["sep"] < t3) or (r.get("r3_tilt_max") or 0) > t4
        for r in rows])
    tp = int((pred & bad).sum()); fn = int((~pred & bad).sum())
    fp = int((pred & ~bad).sum()); tn = int((~pred & ~bad).sum())
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1), (tp, fn, fp, tn)


def main():
    rows = load_rows()
    print("  样本：geom %d，pass %d" % (sum(1 for r in rows if r["label"] == "geom"),
                                       sum(1 for r in rows if r["label"] == "pass")))
    grid1 = [0.06, 0.07, 0.08, 0.09, 0.103]
    grid2 = [0.0]
    grid3 = [9.0, 0.045, 0.0528]
    grid4 = [999, 15, 10]
    results = []
    for t1, t2, t3, t4 in itertools.product(grid1, grid2, grid3, grid4):
        sens, spec, cm = score(rows, t1, t2, t3, t4)
        results.append((sens, spec, t1, t2, t3, t4, cm))
    results.sort(key=lambda x: (-x[0], -x[1]))
    print("\n  按灵敏度优先排序的前 12 个工作点：")
    print("  %-6s %-6s %-7s %-6s %-6s %-6s %s"
          % ("灵敏度", "特异度", "骨长", "反向帧", "间距", "倾角", "混淆(拦/漏/误/放)"))
    seen = set()
    for sens, spec, t1, t2, t3, t4, cm in results:
        k = (round(sens, 2), round(spec, 2))
        if k in seen:
            continue
        seen.add(k)
        print("  %-8.2f %-8.2f %-7.3f %-6.2f %-6.3f %-6.1f %s"
              % (sens, spec, t1, t2, t3, t4, cm))
        if len(seen) >= 12:
            break

    best = results[0]
    sens, spec, t1, t2, t3, t4, cm = best
    print("\n  选定（灵敏度最高，其次特异度最高）：")
    print("    骨长 > %.3f  或  反向帧 > %.2f  或  间距 < %.3f  或  倾角 > %.1f"
          % (t1, t2, t3, t4))
    print("    灵敏度 %.2f  特异度 %.2f  平衡 %.2f"
          % (sens, spec, 0.5 * (sens + spec)))
    print("    几何失败 %d 拦下 %d 漏 %d；可用 %d 误拦 %d 放行 %d"
          % (cm[0] + cm[1], cm[0], cm[1], cm[2] + cm[3], cm[2], cm[3]))

    miss = [r for r in rows if r["label"] == "geom"
            and not ((r["bone_rel_ref"] > t1) or (r["rev"] > t2)
                     or (r["sep"] < t3) or ((r.get("r3_tilt_max") or 0) > t4))]
    print("\n  漏掉：%s" % (["%s_%s_%d" % (m["action"], m["model"], m["sample"]) for m in miss] or "无"))
    fp = [r for r in rows if r["label"] == "pass"
          and ((r["bone_rel_ref"] > t1) or (r["rev"] > t2)
               or (r["sep"] < t3) or ((r.get("r3_tilt_max") or 0) > t4))]
    print("  误拦：%s" % (["%s_%s_%d" % (m["action"], m["model"], m["sample"]) for m in fp] or "无"))

    doc = {"bone_th": t1, "rev_th": t2, "sep_th": t3, "tilt_th": t4,
           "sens": round(sens, 3), "spec": round(spec, 3),
           "bal": round(0.5 * (sens + spec), 3),
           "objective": "灵敏度优先（先保证不漏坏动作）",
           "source": "96 段人工标注，84 段进入二分类"}
    p = ROOT / "outputs" / "_reference" / "criteria_v3.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n  写入 %s" % p)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
