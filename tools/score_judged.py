"""人工判定统计：给一张判定表（clip → 对/部分/错）出汇总。

为什么单独一个脚本
    判定是自由文本（note 里写意见），把「空 = 对」这类解读固定成代码会误判；
    所以判读结果单独存成 JSON（人可读、可复核），这份脚本只负责统计与分类汇总。

用法
    python -m tools.score_judged --verdicts 判定.json [--categories 分类.json]
判定文件格式
    {"clip_01.mp4": "对", "clip_02.mp4": "部分", ...}
分类文件（可选）
    {"clip_02.mp4": ["方向/参照系"], ...}
"""
import argparse
import json
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--categories", default=None)
    a = ap.parse_args()
    v = json.load(open(a.verdicts, encoding="utf-8"))
    cnt = Counter(v.values())
    n = sum(cnt.values())
    print("  == 人工判定（%d 条） ==" % n)
    for k in ("对", "部分", "错"):
        print("    %-3s %2d 条（%.0f%%）" % (k, cnt[k], 100.0 * cnt[k] / max(n, 1)))
    print("    对+部分：%d 条（%.0f%%）" % (cnt["对"] + cnt["部分"],
          100.0 * (cnt["对"] + cnt["部分"]) / max(n, 1)))
    if a.categories:
        cats = json.load(open(a.categories, encoding="utf-8"))
        c = Counter()
        for clip, cs in cats.items():
            for x in cs:
                c[x] += 1
        print()
        print("  == 问题类别 ==")
        for name, k in c.most_common():
            clips = [x.replace(".mp4", "") for x, cs in cats.items() if name in cs]
            print("    %-12s %2d 条：%s" % (name, k, "、".join(clips)))
    bad = [c for c, x in v.items() if x == "错"]
    part = [c for c, x in v.items() if x == "部分"]
    if bad:
        print()
        print("  == 判「错」= %d 条 ==" % len(bad))
        for c in bad:
            print("    " + c)
    if part:
        print("  == 判「部分」= %d 条 ==" % len(part))
        for c in part:
            print("    " + c)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
