"""不依赖大模型的兜底规划：按中文动作词表切句并给出英文提示词。

为什么要有
    规划这一步要调用外部大模型。它一旦不可用（欠费、断网、限流），整条管线就停住，
    而后面所有步骤其实都不需要外部服务。这里给一个确定性的兜底：
    用 hm/actions.py 的中文动作名做最长匹配，切句规则简单，宁可保守也不乱猜。

边界
    兜底只认动作表里出现过的说法；认不出就返回 None，由上层明确报错，不假装成功。
"""
import re

from hm import actions

# 句子切分：先按连接词与标点断开
SPLIT = re.compile(r"[，,。；;、]|然后|接着|之后|最后|再|并且|同时")

# 额外同义说法（中文 → 动作键）
SYNONYMS = {
    "走开": "walk", "走两步": "walk", "向前走": "walk", "往前走": "walk", "走过去": "walk",
    "后退": "walk_back", "往后退": "walk_back", "退后": "walk_back",
    "跑": "run", "跑开": "run", "快跑": "sprint", "冲刺": "sprint", "小跑": "jog",
    "停下": "stand", "停住": "stand", "站住": "stand", "站定": "stand", "停下来": "stand",
    "举手": "raise_hands", "抬起手": "raise_hands", "挥手": "wave_hello", "打招呼": "wave_hello", "再见": "wave_goodbye",
    "转身": "turn", "转个身": "turn", "转过身": "turn", "转两圈": "spin", "转一圈": "spin",
    "跳": "jump", "跳一下": "jump", "跳两下": "jump", "蹦": "jump",
    "蹲下": "crouch", "蹲": "crouch", "弯腰": "bend", "鞠躬": "bow",
    "坐下": "sit", "坐着": "sit", "站起来": "stand_up", "起身": "stand_up", "站起来": "stand_up",
    "踢腿": "kick", "踢": "kick", "出拳": "punch", "打一拳": "punch", "挥拳": "punch",
    "鼓掌": "clap", "抱拳": "fist_salute", "敬礼": "salute",
    "伸懒腰": "stretch", "伸个懒腰": "stretch", "张开双臂": "open_arms",
}


def _table():
    by_key = {}
    for key, zh, prompt, frames in actions.ACTIONS_ALL:
        by_key[key] = (zh, prompt, frames)
    return by_key


def _match_chunk(chunk, by_key):
    """在一个短句里找动作；返回 [(key, prompt, frames)] 或 []。"""
    hits = []
    # 同义词优先（更长更具体的先匹配）
    for zh in sorted(SYNONYMS, key=len, reverse=True):
        if zh in chunk:
            key = SYNONYMS[zh]
            if key in by_key:
                hits.append((key, by_key[key][1], by_key[key][2]))
                chunk = chunk.replace(zh, " ")
    # 动作表里的中文名
    for key, zh, prompt, frames in actions.ACTIONS_ALL:
        if zh and zh in chunk:
            hits.append((key, prompt, frames))
            chunk = chunk.replace(zh, " ")
    # 去重（同一个 key 只留一次）
    seen, out = set(), []
    for key, prompt, frames in hits:
        if key in seen:
            continue
        seen.add(key)
        out.append((key, prompt, frames))
    return out


def plan(sentence):
    """返回 (segments, transitions, terrain) 或 (None, None, None)。"""
    by_key = _table()
    segs = []
    for chunk in SPLIT.split(str(sentence)):
        chunk = chunk.strip()
        if not chunk:
            continue
        for key, prompt, frames in _match_chunk(chunk, by_key):
            segs.append({"key": key, "label": key, "prompt": prompt,
                         "frames": int(frames), "source": chunk,
                         "direction": None, "object": None, "relation": None,
                         "speed": None})
    if not segs:
        return None, None, None
    # 「停下」这类没有独立动作词的句子：动作表里有 stop 吗？
    return segs, [], None
