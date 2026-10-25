"""用已存的 96 段关节，按判据 v2 重算能力表（不重新生成）。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import criteria_v2, job as jobmod, pose_rules
from hm.capability import ACTIONS, APPROX_TH, SUPPORT_TH, classify

OUT = ROOT / "outputs" / "_capsheet"
REF = ROOT / "outputs" / "_reference"


def main():
    ref = jobmod.load_bone_reference()
    pose_th = pose_rules.load_thresholds()
    model_of = {}
    rows = []
    for key, zh, prompt, kws, frames in ACTIONS:
        for model in ("momask", "mdm"):
            passed = 0
            total = 0
            detail = []
            for i in (1, 2, 3):
                f = OUT / "joints" / ("%s_%s_%d.npy" % (key, model, i))
                if not f.exists():
                    continue
                J = np.load(f)
                ok, hard, met, why = criteria_v2.check(J, prompt, ref, pose_th)
                total += 1
                passed += int(ok)
                detail.append({"sample": i, "pass": bool(ok),
                               "bone": round(met.get("bone_rel_mean_max", 0), 3),
                               "rev": round(met.get("rev_frac", 0), 2),
                               "why": why[:2]})
            rate = passed / max(total, 1)
            rows.append({"action": key, "action_zh": zh, "prompt": prompt, "model": model,
                         "samples": total, "passed": passed, "pass_rate": round(rate, 3),
                         "status": classify(rate, total), "details": detail})
            print("  %-12s %-8s %d/%d = %.0f%%  %s"
                  % (key, model, passed, total, 100 * rate, classify(rate, total)), flush=True)

    table = {}
    for r in rows:
        table.setdefault(r["action"], {"action_zh": r["action_zh"], "prompt": r["prompt"],
                                       "models": {}})
        table[r["action"]]["models"][r["model"]] = {
            "samples": r["samples"], "passed": r["passed"],
            "pass_rate": r["pass_rate"], "status": r["status"]}
    for k, v in table.items():
        cands = [(m, d) for m, d in v["models"].items() if d["samples"] > 0]
        best = max(cands, key=lambda x: (x[1]["pass_rate"], -x[1]["samples"]))
        v["best"] = best[0]
        v["best_pass_rate"] = best[1]["pass_rate"]
        v["status"] = best[1]["status"]

    doc = {
        "version": "capability_v2",
        "created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "criteria": "criteria_v2（骨长>10.3% 或 反向帧>0 判失败；阈值由 96 段人工标注校准）",
        "thresholds": {"supported": SUPPORT_TH, "approximate": APPROX_TH},
        "table": table, "raw": rows,
    }
    p = REF / "capability_v2.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    old = json.loads((REF / "capability_v1.json").read_text(encoding="utf-8"))["table"] \
        if (REF / "capability_v1.json").exists() else None
    print("\n  %-12s %-22s %-22s" % ("动作", "旧判据（骨长5%/包络）", "新判据（骨长10.3%/反向帧）"))
    for k, v in table.items():
        o = old.get(k) if old else None
        os_ = "%s %.0f%%" % (o["status"], 100 * (o.get("best_pass_rate") or 0)) if o else "—"
        ns_ = "%s %.0f%%" % (v["status"], 100 * (v.get("best_pass_rate") or 0))
        flag = "  ← 变化" if o and o["status"] != v["status"] else ""
        print("  %-12s %-24s %-24s%s" % (v["action_zh"], os_, ns_, flag))
    print("\n  写入 %s" % p)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
