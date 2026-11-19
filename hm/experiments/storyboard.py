"""把 LLM 给出的段落整理成可执行计划。

这里不做任何语言判断：提示词、帧数、动作名全部来自 LLM。
本模块只补一个段号，并把结果整理成统一结构。

历史说明：这里原来有一张「动作名 → 英文提示词」和「动作名 → 帧数」的映射表。
那是错的——它把语言处理留在了代码里，而且表外的动作无法表达。
现在语言处理全在 LLM 侧，本模块只做透传。
"""


def build(segments, start=None, heading=None):
    """输入 LLM 给出的段落列表，输出带段号的计划。

    每项必须含 prompt；frames 缺省 40；label 缺省用 prompt。
    """
    out = []
    for i, s in enumerate(segments):
        prompt = str(s.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("段%d 没有英文提示词" % i)
        out.append({
            "id": i + 1,
            "label": str(s.get("label", "")).strip() or prompt,
            "prompt": prompt,
            "frames": int(s.get("frames") or 40),
        })
    if not out:
        raise ValueError("没有任何段落")
    return {"segments": out, "total_frames": sum(s["frames"] for s in out)}
