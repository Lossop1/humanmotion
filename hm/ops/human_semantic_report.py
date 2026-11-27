"""人工语义评测的判读与统计。

数据来源
    videos/blind_review/_review/sheet.csv（使用者填写：note 写意见，空的表示没问题）
    videos/blind_review/_review/answer_key.json（clip → job/类别）

判读规则（使用者要求）
    note 为空            → 对
    note 只抱怨措辞/疑问  → 对（带注）
    note 指出缺动作/方向错/跳变/字幕不符 → 部分 或 错
    整句明显不是那个动作    → 错

为什么要脚本：人工判定是论文里最重要的一张表，手抄一次就够了，之后统计必须可重跑。
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "videos" / "blind_review" / "_review"

# 逐条判读（依据使用者的 note 原文，见 sheet.csv）
# 取值：对 / 部分 / 错
VERDICT = {
    "clip_01.mp4": "对",       # 空
    "clip_02.mp4": "对",       # 空
    "clip_03.mp4": "对",       # 空（note 只是说句子本身含糊）
    "clip_04.mp4": "部分",     # 只有跑过去，没有走回来
    "clip_05.mp4": "部分",     # 只有弯腰；2、3 秒有跳变
    "clip_06.mp4": "对",       # 空
    "clip_07.mp4": "对",       # 空
    "clip_08.mp4": "部分",     # 3s 附近画面不连续
    "clip_09.mp4": "部分",     # 3s 附近画面不连续
    "clip_10.mp4": "部分",     # 停住做成了双手张开
    "clip_11.mp4": "部分",     # 转身不慢、浮空转身；走开变成下楼梯
    "clip_12.mp4": "部分",     # 只有弯腰，没捡东西
    "clip_13.mp4": "部分",     # 左右参照系不明
    "clip_14.mp4": "部分",     # 参照系 + 多出后退
    "clip_15.mp4": "错",       # 后退变成前进
    "clip_16.mp4": "部分",     # 少了弯腰；起身像太极起手
    "clip_17.mp4": "错",       # 说后退只往前走了两步，且 2–3s 扭曲
    "clip_18.mp4": "部分",     # 13–14s 转身扭曲
    "clip_19.mp4": "部分",     # 没有挥手只有摆臂
    "clip_20.mp4": "部分",     # 先站 2 秒；出现两个「减速停下」
    "clip_21.mp4": "部分",     # 张望做成了向左走
    "clip_22.mp4": "部分",     # 字幕只有蹲下；3s 不连续
    "clip_23.mp4": "错",       # 原地踏步做成往前走
    "clip_24.mp4": "错",       # 拳头是手刀；1s 跳变；没有后退
    "clip_25.mp4": "部分",     # 编排问题：调一下→转身→膝盖弯
    "clip_26.mp4": "部分",     # 2–3s 跳变
    "clip_27.mp4": "对",       # 空
    "clip_28.mp4": "部分",     # 转身没播完就结束
    "clip_29.mp4": "对",       # 空（走开=向前走，可接受）
    "clip_30.mp4": "部分",     # 脚没离地；后退还是前进
    "clip_31.mp4": "错",       # 转 90° 后像倒放
    "clip_32.mp4": "对",       # 空
    "clip_33.mp4": "部分",     # 3s 不连续
    "clip_34.mp4": "部分",     # 一圈 vs 180° 语义不对齐
    "clip_35.mp4": "错",       # 看不出拿杯子喝水，只是伸手弯腰
    "clip_36.mp4": "部分",     # 3s 不连续
    "clip_37.mp4": "对",       # 空
    "clip_38.mp4": "对",       # 空
    "clip_39.mp4": "对",       # 挥手对，看不出「再见」但不苛求
}

# 问题类别（按 note 归纳，一条可多类）
CATEGORY = {
    "clip_04.mp4": ["动作缺失"],
    "clip_05.mp4": ["动作缺失", "跳变/不连续"],
    "clip_08.mp4": ["跳变/不连续"],
    "clip_09.mp4": ["跳变/不连续"],
    "clip_10.mp4": ["字幕/动作不符"],
    "clip_11.mp4": ["方向/参照系", "字幕/动作不符"],
    "clip_12.mp4": ["动作缺失"],
    "clip_13.mp4": ["方向/参照系"],
    "clip_14.mp4": ["方向/参照系"],
    "clip_15.mp4": ["方向/参照系"],
    "clip_16.mp4": ["动作缺失", "动作风格错"],
    "clip_17.mp4": ["方向/参照系", "跳变/不连续"],
    "clip_18.mp4": ["跳变/不连续"],
    "clip_19.mp4": ["动作缺失"],
    "clip_20.mp4": ["编排/重复"],
    "clip_21.mp4": ["动作缺失", "字幕/动作不符"],
    "clip_22.mp4": ["跳变/不连续", "字幕/动作不符"],
    "clip_23.mp4": ["方向/参照系"],
    "clip_24.mp4": ["动作缺失", "跳变/不连续", "方向/参照系"],
    "clip_25.mp4": ["编排/重复"],
    "clip_26.mp4": ["跳变/不连续"],
    "clip_28.mp4": ["截断"],
    "clip_30.mp4": ["方向/参照系", "动作缺失"],
    "clip_31.mp4": ["动作风格错"],
    "clip_33.mp4": ["跳变/不连续"],
    "clip_34.mp4": ["字幕/动作不符"],
    "clip_35.mp4": ["动作缺失"],
    "clip_36.mp4": ["跳变/不连续"],
}


def main():
    key = json.loads((BASE / "answer_key.json").read_text(encoding="utf-8"))
    items = sorted(VERDICT)
    cnt = Counter(VERDICT[c] for c in items)
    n = len(items)
    print("  == 人工语义评测（%d 条） ==" % n)
    for k in ("对", "部分", "错"):
        print("    %-3s %2d 条（%.0f%%）" % (k, cnt[k], 100.0 * cnt[k] / n))
    print("    对 + 部分：%d 条（%.0f%%）" % (cnt["对"] + cnt["部分"], 100.0 * (cnt["对"] + cnt["部分"]) / n))
    print()
    cats = Counter()
    for c, cs in CATEGORY.items():
        for x in cs:
            cats[x] += 1
    print("  == 问题类别（按 note 归纳，一条可多类） ==")
    for name, k in cats.most_common():
        clips = [c.replace(".mp4", "") for c, cs in CATEGORY.items() if name in cs]
        print("    %-12s %2d 条：%s" % (name, k, "、".join(clips)))
    print()
    print("  == 按句子集拆分 ==")
    for setname in ("std", "fresh"):
        sub = [c for c in items if key.get(c, {}).get("set") == setname]
        cc = Counter(VERDICT[c] for c in sub)
        print("    %-6s %d 条：对 %d、部分 %d、错 %d" % (setname, len(sub), cc["对"], cc["部分"], cc["错"]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
