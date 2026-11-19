"""按训练数据的实际措辞重写提示词，重测那批失败动作。

改动依据：从 87384 条描述里统计出的高频句式（见 mine_phrasings 输出）。
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
from hm import criteria_v2, generate, job as jobmod, pose_rules, to_joints

REF_DIR = ROOT + "/outputs/_reference"

# (键, 中文, 旧提示词, 新提示词, 帧数, 数据里的高频写法)
CASES = [
    ("sit", "坐下", "a person sits down on a chair",
     "a person sits down and stands back up", 52,
     "sits down and stands back up（18 次）"),
    ("kneel", "跪下", "a person kneels down on one knee",
     "a person kneels down on both knees", 44,
     "kneels down on both knees（8 次）"),
    ("crouch", "蹲下", "a person crouches down",
     "a person squats down and stands back up", 52,
     "squats down and stands back up（4 次）；crouch 在数据里几乎不出现"),
    ("lie", "躺下", "a person lies down on the floor",
     "a person lies down on the floor", 44,
     "无高频写法，保持原样作对照"),
    ("nod", "点头", "a person nods the head",
     "a person nods their head", 32,
     "nods their head（数据里仅 20 条）"),
    ("scratch", "挠头", "a person scratches the head with the right hand",
     "a person scratches their head with their right hand", 36,
     "scratches their head with their right hand（10 次）"),
    ("turn_left", "向左转", "a person turns left in place",
     "a person walks forward and turns left", 48,
     "walks forward, turns left（4 次）"),
    ("turn_right", "向右转", "a person turns right in place",
     "a person walks forward and turns right", 48,
     "walks forward, turns right（4 次）"),
    ("spin", "转圈", "a person spins around in place",
     "a person walks in a circle", 48,
     "walks in a circle（32 次）"),
    ("crawl", "爬行", "a person crawls forward on the floor",
     "a person gets on all fours and crawls forward", 56,
     "gets on all fours and crawls（6 次）"),
    ("stretch", "伸展", "a person stretches both arms up",
     "a person stretches their arms above their head", 40,
     "stretching arms above head（4 次）"),
    ("jump_fwd", "向前跳", "a person jumps forward",
     "a person jumps forward once", 36,
     "jumps forward once（18 次）"),
    ("kick", "踢腿", "a person kicks with the right leg",
     "a person kicks with their right leg", 36,
     "kicks with their left leg（6 次）"),
    ("dance", "跳舞", "a person dances in place",
     "a person dances", 60,
     "dances / is dancing"),
]


def check(prompt, frames, k, ref, pose_th, model="momask"):
    feats = generate.generate([prompt] * k, [frames] * k, name=model)
    ok, why_all, rel = 0, [], []
    for F in feats:
        J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
        passed, hard, met, why = criteria_v2.check(J, prompt, ref, pose_th)
        ok += int(passed)
        why_all.append(why[:2])
        rel.append(round(met.get("bone_rel_mean_max") or 0, 3))
    return ok, len(feats), why_all, rel


def main():
    ref = jobmod.load_bone_reference()
    pose_th = pose_rules.load_thresholds()
    out = []
    print("  %-10s %-6s %-6s  %s" % ("动作", "旧", "新", "新提示词"))
    for key, zh, old_p, new_p, frames, ev in CASES:
        o1, n1, w1, r1 = check(old_p, frames, 3, ref, pose_th)
        o2, n2, w2, r2 = check(new_p, frames, 3, ref, pose_th)
        out.append({"key": key, "zh": zh, "old_prompt": old_p, "new_prompt": new_p,
                    "old_pass": "%d/%d" % (o1, n1), "new_pass": "%d/%d" % (o2, n2),
                    "old_rate": o1 / n1, "new_rate": o2 / n2,
                    "evidence": ev, "new_why": w2, "new_bone": r2})
        flag = "  ↑" if o2 > o1 else ("" if o2 == o1 else "  ↓")
        print("  %-10s %d/%d    %d/%d   %s%s" % (zh, o1, n1, o2, n2, new_p[:52], flag), flush=True)

    p = REF_DIR + "/prompt_rewrite_test.json"
    json.dump({"criteria": "criteria_v3", "model": "momask", "k": 3, "cases": out},
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n  写入 %s" % p)
    better = [x for x in out if x["new_rate"] > x["old_rate"]]
    print("  %d/%d 个动作在换写法后通过率提高" % (len(better), len(out)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
