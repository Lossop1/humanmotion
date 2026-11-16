"""把提示词库扩到动作全集：每个动作试三种写法，用几何判据打分。

为什么
    库现在只有 8 个动作，A06 只能在这 8 个上替换措辞。
    措辞对结果影响很大（punch 0/4→4/4），所以把库里覆盖到实际会用到的动作，
    规划器落到「测过且稳」写法的机会就大。

三种写法
    canonical  能力测试用的原写法
    stripped   去掉修饰尾巴（slowly / with the right hand / a few steps…）
    in place   没有方向词时补 in place，避免模型乱拐
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import actions, criteria_v2, generate, to_joints
from hm.prompt_variants import variants

REF = Path("/home/chuan/motion-work/outputs/_reference")
K = 4


def score(text, frames, model="momask"):
    feats = generate.generate([text] * K, [frames] * K, name=model)
    passed = 0
    for F in feats:
        J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
        ok, _h, _m, _w = criteria_v2.check(J, text, None, None)
        passed += 1 if ok else 0
    return {"passed": passed, "samples": K, "rate": passed / K}


def main(limit=None):
    bank = {}
    p = REF / "prompt_bank.json"
    if p.exists():
        bank = json.loads(p.read_text(encoding="utf-8"))
    todo = actions.ACTIONS_ALL[:limit] if limit else actions.ACTIONS_ALL
    rows = []
    for key, zh, prompt, frames in todo:
        entry = bank.setdefault(key, {"action_zh": zh, "phrasings": []})
        have = {r["text"] for r in entry["phrasings"]}
        for text in variants(prompt):
            if text in have:
                continue
            r = score(text, frames)
            entry["phrasings"].append({"text": text, "models": {"momask": r}})
            print("    %-12s %-46s %d/%d" % (key, text[:46], r["passed"], r["samples"]), flush=True)
        best = max(entry["phrasings"], key=lambda r: r["models"]["momask"]["rate"])
        entry["best"] = best["text"]
        entry["best_rate"] = best["models"]["momask"]["rate"]
        entry["tested_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rows.append((key, entry["best_rate"], len(entry["phrasings"])))
    p.write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")
    good = [r for r in rows if r[1] >= 0.75]
    print()
    print("  动作 %d 个：最优写法通过率 ≥75%% 的 %d 个，<50%% 的 %d 个"
          % (len(rows), len(good), len([r for r in rows if r[1] < 0.5])))
    print("  通过率最低的 6 个：%s" % [(k, round(r, 2)) for k, r, _ in sorted(rows, key=lambda x: x[1])[:6]])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
