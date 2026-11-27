"""多义动作：同一句中文在不同场景指哪种动作，先声明再执行。

为什么需要
    「走回来」至少有三种合理理解：
      ① 转身往回走（转身 180°，再走向出发点）
      ② 后退着回来（面朝原方向，倒着退回来）
      ③ 侧着挪回来（侧移）
    使用者指出的正是这件事：动作本身能解释成「回来」，所以不能拿一种理解去判对错；
    真正的问题是「系统默默选了哪一种、有没有说出来」。

用法
    from hm.ambiguity import lookup, describe
    lookup("一个人快速跑过去，然后慢慢走回来")   # -> [(词, [解释...])]
    check_interpretation(prompt, sig)            # 声明了什么解释、与动作是否相符
"""
import re

# 词 → [(解释名, 该解释对应的提示词要素, 可用判据)]
AMBIGUOUS = {
    "走回来": [
        ("转身往回走", "turns around and walks back", {"yaw_deg": ("min", 120.0)}),
        ("后退着回来", "walks backward", {"backward_m": ("min", 0.15)}),
        ("侧着挪回来", "sidesteps back", {"facing_travel_deg": ("min", 60.0)}),
    ],
    "跑回来": [
        ("转身跑回来", "turns around and runs back", {"yaw_deg": ("min", 120.0)}),
        ("倒退着跑", "runs backward", {"backward_m": ("min", 0.15)}),
    ],
    "离开": [
        ("向前走开", "walks forward away", {"disp_m": ("min", 0.6)}),
        ("原路退开", "walks backward away", {"backward_m": ("min", 0.15)}),
    ],
    "回到原位": [
        ("转身走回原位", "turns around and walks back to the start", {"yaw_deg": ("min", 120.0)}),
        ("后退回原位", "walks backward to the start", {"backward_m": ("min", 0.15)}),
    ],
    "绕一圈": [
        ("绕圈走", "walks in a circle", {"yaw_deg": ("min", 200.0)}),
        ("原地转一圈", "turns around in place once", {"yaw_deg": ("min", 300.0)}),
    ],
}


def lookup(text):
    """返回句子里命中哪些多义词、各有哪些解释。"""
    s = str(text or "")
    return [(w, opts) for w, opts in AMBIGUOUS.items() if w in s]


def interpretations(text):
    """把命中的多义词展开成一个解释列表（用于给规划器看）。"""
    out = []
    for w, opts in lookup(text):
        for name, _hint, _rule in opts:
            out.append("%s → %s" % (w, name))
    return out


def declared(label):
    """从规划给出的标签里读出「理解为：X」，没写就返回 None。"""
    m = re.search(r"理解[为成][:：]?\s*([^）)]+)", str(label or ""))
    return m.group(1).strip() if m else None


def check_interpretation(prompt, sig):
    """按「提示词实际做了什么」推断解释，并检查是否可判定。

    返回 (解释名 or None, 与哪些判据相符/不符)
    """
    p = str(prompt or "").lower()
    if "turn" in p and "back" in p:
        name, rules = "转身往回走", {"yaw_deg": ("min", 120.0)}
    elif "backward" in p:
        name, rules = "后退着回来", {"backward_m": ("min", 0.15)}
    elif "sidestep" in p and "back" in p:
        name, rules = "侧着挪回来", {"facing_travel_deg": ("min", 60.0)}
    elif "circle" in p:
        name, rules = "绕圈走", {"yaw_deg": ("min", 200.0)}
    else:
        return None, []
    bad = []
    for key, (kind, thr) in rules.items():
        v = sig.get(key)
        if v is None:
            bad.append("%s 无数据" % key)
        elif kind == "min" and v < thr:
            bad.append("%s=%.1f < %.1f" % (key, v, thr))
        elif kind == "max" and v > thr:
            bad.append("%s=%.1f > %.1f" % (key, v, thr))
    return name, bad
