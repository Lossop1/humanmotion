"""把 G16–G20 正式写进目标树，并核对写入是否真的发生。

教训
    之前几次「文档已更新」的打印是无条件执行的，实际插入锚点写错（标题是
    "### G10 评测　【未达成】"），四处内容根本没进文档。这里改成：写完重新读，逐条断言。
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

P = r"D:\my-projects\humanmotion\repo\目标树.md"
s = open(P, encoding="utf-8", newline="").read()

BLOCKS = [
    ("G16", """### G16 唯一入口与冒烟测试　【达成】

```
判据  python -m hm.main "句子" 一条命令出片；出口码 0/1/2 = 正常/有问题/没出片
实测  单句 21.4 秒出片、退出码 0；--check 三句全部通过；--batch 五句成表
手段  之前有两套入口（main 与 run_full）各自维护流程，现在 main 只做参数与汇报
叶子  hm/main.py；hm/run_full.py run()
```

"""),
    ("G17", """### G17 帧数是杠杆　【达成】

```
来历  「raises one hand up in place」批处理 4/4 全败，单独测 44 帧却 16/16 通过
量化  同一提示词：40 帧 1/4（骨盆朝向与脚尖反向，100% 帧点积 -1.00）；44 帧 4/4
做法  重试时片段加长 8 帧（上限 80），与加倍候选同时进行
效果  批处理去掉段减少，但暴露了更大的交界（此前「丢段」掩盖了它）
叶子  hm/run_full.py 重试段
```

"""),
    ("G18", """### G18 交界修复（补候选）　【部分】

```
做法  选择后对姿态差 >90 度的相邻段，为后一段再生成 6–8 个候选（编号 91 起，
      不覆盖原候选），挑与前一端最贴合的通过者
实测  四对坏交界：93→87、104→95、157→151、112→112
结论  收益 5–10 度 → 这些交界是两侧姿态本来就差得远，不是抽样运气
叶子  hm/run_job.py repair_junctions
```

"""),
    ("G19", """### G19 合并相邻短段　【达成】

```
做法  两段都不超过 56 帧时合成一段用复合提示词生成（默认开启，HM_MERGE_SHORT=0 关）
实测  上限 44：有交界的句子 22 → 18，中位 45 → 28 度
      上限 56：有交界的句子 18 → 7，四分位 22–28，最差 124 → 101 度
前提  复合提示词本身能过：「停下→举手」「跳→转身」各 6/6 几何通过且两动作都出现
代价  合并段不能逐动作做签名检查；日志保留合并来源
叶子  hm/run_job.py merge_short_segments
```

"""),
    ("G20", """### G20 三动作合链　【否证】

```
实测  「jumps, turns around and walks forward」几何 6/6、跳起 5/6，但转身 0/6——转身被吃掉
      「stretches, bends forward and stands up」几何 5/6
结论  三动作合链会丢动作，而合并段无法逐动作检查，风险大于收益；合并只做两段
```

"""),
]

anchor = "### G10 评测"
missing = [k for k, _ in BLOCKS if ("### %s " % k) not in s]
if not missing:
    print("  四条已在文档里，无需写入")
else:
    idx = s.index(anchor)
    add = "".join(text for key, text in BLOCKS if ("### %s " % key) in missing)
    s = s[:idx] + add + s[idx:]
    open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("  已写入：%s" % "、".join(missing))

# 复核：重新读盘，逐条断言
s2 = open(P, encoding="utf-8", newline="").read()
for key, _ in BLOCKS:
    head = "### %s " % key
    ok = head in s2
    print("  %s %s" % ("OK " if ok else "缺失", key))
print("  文档行数：%d" % len(s2.splitlines()))
