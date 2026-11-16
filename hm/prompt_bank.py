"""提示词库：同一动作换几种写法，用实测通过率挑出该用哪句。

为什么这属于我们的方法
    生成模型不重研，但它吃哪句话、吃得怎么样，是规划层能决定的事。
    坐下就是一个例子：能力表里 1.00 的那句是「坐下再站起」，
    单动作的五种写法 0/20。所以「用哪句」必须测出来，不能由 planner 自由发挥。

做法
    每个动作给 3 种简单写法，各生成 k 个样本，跑几何判据，记通过率。
    产物 outputs/_reference/prompt_bank.json：
        {动作: {phrasings: [{text, passed, samples}], best, best_rate, tested_at}}
    A06 组提示词时会先查这个库：有更稳的写法就用库里的。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import criteria_v2, generate, to_joints

REF = Path("/home/chuan/motion-work/outputs/_reference")
K = 4

CASES = {
    "sit": ("a person sits down", [
        "a person sits down",
        "a person sits on a chair",
        "a person lowers into a chair",
    ], 52),
    "pick": ("a person picks something up", [
        "a person picks something up from the floor",
        "a person bends down and picks something up",
        "a person reaches down to pick something up",
    ], 52),
    "bend": ("a person bends forward", [
        "a person bends forward",
        "a person bends over forward",
        "a person bows forward",
    ], 44),
    "jump": ("a person jumps", [
        "a person jumps up",
        "a person jumps up once in place",
        "a person hops up",
    ], 40),
    "wave": ("a person waves", [
        "a person waves hello",
        "a person waves with the right hand",
        "a person waves goodbye",
    ], 44),
    "walk_back": ("a person walks backward", [
        "a person walks backward",
        "a person steps backward",
        "a person walks back",
    ], 56),
    "kick": ("a person kicks", [
        "a person kicks forward",
        "a person kicks with the right leg",
        "a person does a front kick",
    ], 40),
    "punch": ("a person punches", [
        "a person punches forward",
        "a person throws a punch",
        "a person punches with the right hand",
    ], 40),
}


def main(models=("momask",)):
    bank = {}
    if (REF / "prompt_bank.json").exists():
        bank = json.loads((REF / "prompt_bank.json").read_text(encoding="utf-8"))
    for action, (_label, phrasings, frames) in CASES.items():
        entry = bank.setdefault(action, {"phrasings": []})
        for text in phrasings:
            row = {"text": text, "models": {}}
            for model in models:
                feats = generate.generate([text] * K, [frames] * K, name=model)
                passed = 0
                for F in feats:
                    J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
                    ok, _h, _m, _w = criteria_v2.check(J, text, None, None)
                    passed += 1 if ok else 0
                row["models"][model] = {"passed": passed, "samples": K,
                                        "rate": passed / K}
            entry["phrasings"].append(row)
            print("    %-9s %-44s %s" % (action, text,
                  " ".join("%s %d/%d" % (m, r["passed"], r["samples"])
                           for m, r in row["models"].items())), flush=True)
        best = max(entry["phrasings"], key=lambda r: r["models"][models[0]]["rate"])
        entry["best"] = best["text"]
        entry["best_rate"] = best["models"][models[0]]["rate"]
        entry["tested_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print("    → %s 最优：%r（%d/%d）" % (action, entry["best"],
              best["models"][models[0]]["passed"], best["models"][models[0]]["samples"]))
    (REF / "prompt_bank.json").write_text(json.dumps(bank, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("  已写 %s" % (REF / "prompt_bank.json"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
