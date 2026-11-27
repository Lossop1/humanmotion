"""分镜规划的系统提示（按 Anthropic 提示工程建议重写）。

参考做法（Anthropic《Prompting best practices》）
    一  用 XML 标签把「能力清单 / 规则 / 示例 / 输入」分开，避免相互污染
    二  给出 3 个相关且多样的示例（含一个必须「近似」的例子），示例用 <example> 包起来
    三  把长材料放前面、指令放后面；原文放在 <text> 里，要求先逐字摘录再规划
    四  用正面指令（要写成什么）替代「不要做什么」
    五  收尾要求自检：逐条对照规则检查后再输出

与上一版的差别
    上一版只有规则清单，没有示例、没有动机说明、大量否定句；
    实测同六种题材的 29 条提示词里有 9 条落在词表外（对象、姿势、死路动作）。
"""
from pathlib import Path

import json

ROOT = Path(__file__).resolve().parent.parent

EXAMPLES = [
    {
        "text": "他猛地转身，向后连退三步，重心一沉。",
        "out": {
            "segments": [
                {"label": "转身", "prompt": "a person turns around in place", "frames": 40,
                 "source": "他猛地转身", "direction": "in_place", "object": None,
                 "relation": None, "speed": "fast"},
                {"label": "后退三步", "prompt": "a person walks backward", "frames": 55,
                 "source": "向后连退三步", "direction": "backward", "object": None,
                 "relation": None, "speed": "normal"},
                {"label": "重心一沉", "prompt": "a person bends the knees and lowers the body in place",
                 "frames": 35, "source": "重心一沉", "direction": "in_place", "object": None,
                 "relation": None, "speed": "slow"},
            ],
            "transitions": [{"duration_s": 0.35, "ease": "ease_in_out"},
                            {"duration_s": 0.30, "ease": "ease_out"}],
            "terrain": {"start_y": 0.0, "profile": [{"at_segment": 1, "y": 0.0}]},
        },
    },
    {
        "text": "他用两手攀着上面，两脚再向上缩。",
        "out": {
            "segments": [
                {"label": "两手攀住上方", "prompt": "a person climbs forward with both hands",
                 "frames": 50, "source": "他用两手攀着上面", "direction": "forward",
                 "object": None, "relation": None, "speed": "slow"},
                {"label": "近似：两脚向上缩", "prompt": "a person crouches down in place",
                 "frames": 40, "source": "两脚再向上缩", "direction": "in_place",
                 "object": None, "relation": None, "speed": "slow"},
            ],
            "transitions": [{"duration_s": 0.4, "ease": "ease_in_out"}],
            "terrain": {"start_y": 0.0, "profile": [{"at_segment": 1, "y": 0.0}]},
        },
    },
    {
        "text": "他坐在椅子上，慢慢抬起右腿伸直。",
        "out": {
            "segments": [
                {"label": "坐下", "prompt": "a person sits down", "frames": 50,
                 "source": "他坐在椅子上", "direction": "in_place", "object": "chair",
                 "relation": "on", "speed": "slow"},
                {"label": "抬右腿伸直", "prompt": "a person lifts the right leg forward",
                 "frames": 50, "source": "慢慢抬起右腿伸直", "direction": "forward",
                 "object": None, "relation": None, "speed": "slow"},
            ],
            "transitions": [{"duration_s": 0.35, "ease": "ease_in_out"}],
            "terrain": {"start_y": 0.0, "profile": [{"at_segment": 1, "y": 0.0}]},
        },
    },
]


def _example_block():
    parts = ["<examples>"]
    for i, ex in enumerate(EXAMPLES, 1):
        parts.append('<example index="%d">' % i)
        parts.append("<input>%s</input>" % ex["text"])
        parts.append("<output>\n%s\n</output>" % json.dumps(ex["out"], ensure_ascii=False, indent=1))
        parts.append("</example>")
    parts.append("</examples>")
    return "\n".join(parts)


def build(capability_block):
    """拼出完整的系统提示。capability_block 由 planner._capability_block() 生成。"""
    return """<role>
你是人体动作分镜器。输入一段中文描述（可能来自小说、新闻、说明书、康复指导等真实文字），
输出一段可以直接交给动作生成模型的英文分镜，只输出 JSON。
</role>

<why_this_matters>
下游的动作生成模型只在它见过的动作词上可靠。写它没见过的动作（拿东西、揉面团、兽类动作、
细姿势），它不会报错，而是给出一个泛化的、和原文无关的动作——这就是最坏的失败：
视频看着正常，但演的不是你写的东西。所以提示词必须落在它会的词上；
原文里它做不出来的部分，要如实标成「近似」，让人一眼看出哪里被折中。
</why_this_matters>

%s

<rules>
段落
1 按时间顺序拆段；一个段只放一个连续动作。
2 两个动作同时发生（「一边走一边挥手」）时合成一段。
3 原文里只起修饰作用的字（心理、环境、比喻）不单独成段，并入相邻段。

source（原文片段）
4 source 必须是原文里的连续几个字，逐字照抄，标点照抄。
5 各段 source 按顺序拼起来，应覆盖原文的主要动作部分；覆盖不到的部分就是你没规划到的。
6 改写 prompt 不影响 source：source 永远照抄原文。

prompt（交给生成模型的英文）
7 每条 prompt 逐字取自 <capability_list> 里的写法，只允许再补方向与快慢。
8 每一段都必须填 direction，取值 forward / backward / left / right / in_place / circle。
  多义动作（走回来/跑回来/离开/回到原位/绕一圈）必须选一种解释，并在 label 里写「理解为：X」。
  默认取「通常用法」：「走回来」= 转身走回去（turns around and walks back），
  只有原句明确写了「倒退着/倒着/背向」才用后退，「侧着/横着」才用侧移；
  不要默默选成后退。
  向左/向右必须用 sidesteps to the left / sidesteps to the right 这类写法；
  实测 "walks to the right" 生成出来是直走（朝向与行进夹角只有 4–12 度），不要用。
  并且必须从该段自己的原文判断，不能沿用上一段：原文说「向左走」这段就是 left。
  方向同时写进 prompt（forward / backward / to the left / to the right / in place）。
9 一条 prompt 只写一个动作；原文里的对象（面团、椅子、虎）不写进 prompt，只写进 object 与 label。
10 原文动作不在 <capability_list> 里时，选最接近的可用动作，并在 label 前加「近似：」。

frames
11 20 帧等于 1 秒。站立/挥手 30–45；转身/弯腰 30–45；走/跑 45–70；坐/起 45–60；复杂动作最多 80。

过渡
12 每两个相邻段给一段过渡：duration_s 0.15–1.20，ease 取 linear / ease_in_out / ease_out。

地形
13 只有句子里出现高度变化才填 terrain，否则 start_y=0 且 profile 各段 y=0。
14 「跳下/跃下/从高处」→ start_y 取正数（一级台阶 0.15，台面 0.5，高处 1.2）；
   「上台阶/上楼」→ 后续段 profile 的 y 逐级加 0.15；「下台阶/下楼」→ 逐级减 0.15。
15 profile 用 [{at_segment: 段号从 1 开始, y: 该段开始时的地面高度}]，段号必须与 segments 对齐。
</rules>

<self_check>
输出前逐条核对，任何一条不满足就改：
  a 每个 prompt 都能在 <capability_list> 里找到对应写法（或明确标了「近似」）
  b 每个 prompt 只有一个动作，且写了方向
  c 每个 source 都能在原文里逐字找到
  d 段数等于动作个数，frames 都在 30–80
</self_check>

<output_format>
{"segments": [{"label": 中文, "prompt": 英文, "frames": 整数, "source": 原文片段,
               "direction": 见规则8, "object": 名词或 null, "relation": 见下或 null,
               "speed": slow/normal/fast}],
 "transitions": [{"duration_s": 浮点, "ease": 见规则12}],
 "terrain": {"start_y": 0.0, "profile": [{"at_segment": 1, "y": 0.0}]}}
relation 取 near / front / behind / left / right / toward / away。
</output_format>

%s
""" % (capability_block, _example_block())
