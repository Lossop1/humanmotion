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
# 列表第一项 = 通常用法（默认解释）。使用者定：标定要确定通常用法，
# 例如「走回来」通常指「转身走回去」，不是「后退着回来」。
# 只有原句带明确修饰（倒退着、侧着、原地…）时才偏离通常用法。
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


# 原句里出现这些修饰时，才偏离通常用法
HINTS = {
    "后退着回来": ("倒退", "后退", "倒着", "背向"),
    "侧着挪回来": ("侧着", "横着", "侧移"),
    "转身往回走": ("转身", "回头"),
    "转身跑回来": ("转身", "回头"),
    "倒退着跑": ("倒退", "倒着", "背向"),
    "原路退开": ("倒退", "后退", "倒着"),
}


def usual(term):
    """该词的通常用法（列表第一项）。"""
    opts = AMBIGUOUS.get(term) or []
    return opts[0][0] if opts else None


def pick(text, term=None):
    """按原句修饰选解释：有明确修饰就用它，否则用通常用法。

    返回 (解释名, 依据)，依据形如「通常用法」或「原句有『倒退』」。
    """
    s = str(text or "")
    terms = [term] if term else [w for w, _ in lookup(s)]
    out = []
    for w in terms:
        opts = AMBIGUOUS.get(w) or []
        for name, _hint, _rule in opts:
            hits = [h for h in HINTS.get(name, ()) if h in s]
            if hits:
                out.append((w, name, "原句有『%s』" % hits[0]))
                break
        else:
            if opts:
                out.append((w, opts[0][0], "通常用法"))
    return out


# 声明用的中文可能和表里的写法不一样（实测规划写「背向行进方向后退返回」），
# 这里按关键词归一到表内的标准名，避免把「同一件事的两种说法」判成不一致。
KEYS = {
    "转身往回走": ("转身", "回头", "转过身"),
    "后退着回来": ("后退", "倒退", "背向", "倒着"),
    "侧着挪回来": ("侧", "横"),
    "转身跑回来": ("转身", "回头"),
    "倒退着跑": ("倒退", "倒着", "背向"),
    "向前走开": ("向前", "走开", "离开"),
    "原路退开": ("后退", "倒退", "退开"),
    "转身走回原位": ("转身", "走回"),
    "后退回原位": ("后退", "倒退"),
    "绕圈走": ("绕", "圈"),
    "原地转一圈": ("原地", "转一圈"),
}


def canonical(term, declared):
    """把任意中文声明归一成表内标准名；认不出来就原样返回。"""
    text = str(declared or "")
    opts = [n for n, _h, _r in (AMBIGUOUS.get(term) or [])]
    if not text:
        return None
    if text in opts:
        return text
    for name in opts:
        if any(k in text for k in KEYS.get(name, ())):
            return name
    return text
