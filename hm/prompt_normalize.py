"""限定版措辞规范化：只做两条实测支持、且不改变语义的改写。

为什么只做两条
    以前放开改写（提示词库替换、库变体）三次都改错了语义（crawl→push、stands up→stands still、
    stand up→pick）。所以这里只保留「补出肢体」这一类：出拳用手、踢腿用脚，是动作本身的含义，
    不是换动作。

依据
    出拳：负对照最大 0.22；«punches with the right hand» 最小 0.53（6/6）、
          «throws a right punch» 最小 0.52（6/6）；而 «punches forward» 有不点手的样本只到 0.17。
"""
import re

# (正则, 说明, 改写函数)
SUBJECT_PATTERNS = [
    (r"\bpunches forward\b", "出拳补出用哪只手",
     lambda m: "punches forward with the right hand"),
    (r"\bthrows a punch\b", "出拳补出用哪只手",
     lambda m: "throws a right punch"),
]

# 「减速并停下」这类：带 in place 时实测几乎停不住（64/72 帧 0/6、80 帧 2/6），
# 去掉 in place 后 64 帧就 6/6 —— 语义不变，属于纯句式问题
STOP_IN_PLACE = re.compile(r"\b(slows? down and stops?|stops? walking and stands? still)\s+in place\b",
                           re.I)

LIMB = re.compile(r"\b(hand|handed|arm|fist|left|right|leg|knee|foot)\b", re.I)


def normalize(prompt):
    """返回 (新提示词, 改动说明或 None)。只在原句没有点名肢体时才补。"""
    text = str(prompt)
    if LIMB.search(text):
        return text, None
    m_stop = STOP_IN_PLACE.search(text)
    if m_stop:
        return STOP_IN_PLACE.sub(lambda m: m.group(1), text), "停住句去掉 in place"

    for pattern, why, fn in SUBJECT_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            new = re.sub(pattern, fn, text, count=1, flags=re.I)
            # 只允许「原动词仍在、且新增的是肢体词」的改动
            verb_ok = re.search(r"punch", new, flags=re.I) is not None
            if verb_ok and LIMB.search(new):
                return new, why
            return text, None
    return text, None


# 中文动作 → 提示词里应出现的关键词（用于检查「字幕说做了、提示词里没有」）
ACTION_LEX = [
    ("跑", ("run", "runs", "running", "sprint", "jog")),
    # 顺序即优先级：踏步/侧移要排在「走」前面，否则描述里 step/march 会被当成走
    ("踏步", ("march", "marches", "marching", "stomp", "stomps")),
    ("侧移", ("sidestep", "sidesteps", "sidestepping")),
    ("走", ("walk", "walks", "walking")),
    ("挥手", ("wave", "waves", "waving")),
    ("举手", ("raise", "raises", "raising", "lift", "lifts")),
    ("出拳", ("punch", "punches", "punching")),
    ("踢", ("kick", "kicks", "kicking")),
    ("跳", ("jump", "jumps", "jumping", "hop", "hops")),
    ("转身", ("turn", "turns", "turning")),
    ("停下", ("stop", "stops", "stands still", "slow")),
    ("坐", ("sit", "sits", "sitting", "seat")),
    ("蹲", ("crouch", "squat", "kneel")),
    ("弯腰", ("bend", "bends", "bending", "bow", "bowing")),
    ("推", ("push", "pushes", "pushing")),
    ("鞠躬", ("bow", "bowing")),
]


def missing_actions(source_text, prompt):
    """source 里提到、但 prompt 里没有对应动作词的项。

    实测 clip_19「一边走一边挥手」：prompt 只有 walks forward，挥手整段丢了；
    clip_12「弯腰捡起地上的东西」：prompt 只有 bend，捡起没有。
    这类段要继续生成也可以，但字幕必须标出「缺」，不能假装做了。
    """
    src = str(source_text or "")
    en = str(prompt or "").lower()
    missing = []
    for zh, words in ACTION_LEX:
        if zh in src and not any(w in en for w in words):
            missing.append(zh)
    return missing


def describe_zh(prompt):
    """把英文提示词翻成「它实际要做什么」的中文短标签。

    字幕要取实际提示词，而不是规划原句：实测规划说「一边走一边挥手」，
    但提示词只有 walks forward，字幕却写「边走边挥手」——那是撒谎。
    这里只做词表映射，不做翻译，宁可粗糙也不假装懂语义。
    """
    en = str(prompt or "").lower()
    out = []
    for zh, words in ACTION_LEX:
        if any(w in en for w in words):
            out.append(zh)
    return out


SPEED_WORD = {"slow": "slowly", "fast": "quickly"}


def apply_speed_word(prompt, speed):
    """把 slow/fast 写进提示词（slowly / quickly）。

    实测问题：使用者说「慢慢转身」的转身并不慢。根因是速度只做了帧数伸缩，
    提示词里没有 slowly，模型就按自己节奏生成。这里只加副词，不动动作词。
    """
    text = str(prompt or "").strip()
    word = SPEED_WORD.get(str(speed or "").strip().lower())
    if not word or not text:
        return text
    low = text.lower()
    # 已有任何速度词就不再添加，否则会出现 "runs forward fast quickly"
    if any(w in low for w in ("slowly", "quickly", "slow", "fast")):
        return text
    # 已经以 in place / around 结尾的，副词放句末更自然
    return "%s %s" % (text, word)
