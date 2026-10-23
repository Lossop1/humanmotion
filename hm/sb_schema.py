"""分镜 Schema：A02 的字段约定与 A04 的校验。

字段按说明书 A02 给出的 JSON 固定；校验失败即停止，不靠下游模型猜。
"""
import json

# 每段允许的字段与类型
SEGMENT_SCHEMA = {
    "segment_id": str,
    "order": int,
    "action": str,
    "source_text": str,
    "object": (str, type(None)),
    "relation": (str, type(None)),
    "duration_frames": (int, type(None)),
    "speed": (str, type(None)),
    "facing": (str, type(None)),
    "spatial_target": (dict, type(None)),
}

REQUIRED = ("segment_id", "order", "action", "source_text")

SPEEDS = ("slow", "normal", "fast", None)
RELATIONS = ("near", "front", "behind", "left", "right", "toward", "away", None)


def blank_segment(seg_id, order, action, source_text):
    """生成一个字段齐全的空段，缺的填 null（不填假值）。"""
    return {
        "segment_id": seg_id,
        "order": order,
        "action": action,
        "source_text": source_text,
        "object": None,
        "relation": None,
        "duration_frames": None,
        "speed": None,
        "facing": None,
        "spatial_target": None,
    }


def validate(storyboard):
    """返回 (ok, errors, warnings)。"""
    errors, warnings = [], []
    segs = storyboard.get("segments")
    if not isinstance(segs, list) or not segs:
        return False, ["segments 为空或不是数组"], warnings

    seen_ids, seen_orders = set(), []
    for i, s in enumerate(segs):
        tag = "第%d段" % (i + 1)
        if not isinstance(s, dict):
            errors.append("%s 不是对象" % tag)
            continue
        for k in REQUIRED:
            if k not in s or s[k] in (None, ""):
                errors.append("%s 缺少必填字段 %s" % (tag, k))
        for k, t in SEGMENT_SCHEMA.items():
            if k in s and s[k] is not None and not isinstance(s[k], t):
                errors.append("%s 字段 %s 类型不对：%r" % (tag, k, s[k]))
        sid = s.get("segment_id")
        if sid in seen_ids:
            errors.append("%s 编号重复：%s" % (tag, sid))
        seen_ids.add(sid)
        if isinstance(s.get("order"), int):
            seen_orders.append(s["order"])
        # 只做约定内的检查，不擅自补空间目标
        if s.get("duration_frames") is not None:
            if not isinstance(s["duration_frames"], int) or s["duration_frames"] <= 0:
                errors.append("%s duration_frames 必须是正整数或 null" % tag)
        if s.get("speed") not in SPEEDS:
            warnings.append("%s speed 不在枚举内：%r" % (tag, s.get("speed")))
        if s.get("relation") not in RELATIONS:
            warnings.append("%s relation 不在枚举内：%r" % (tag, s.get("relation")))
        if s.get("spatial_target") is not None:
            st = s["spatial_target"]
            if not isinstance(st, dict) or "x" not in st or "z" not in st:
                errors.append("%s spatial_target 需要 x/z 字段" % tag)

    if seen_orders:
        if sorted(seen_orders) != list(range(1, len(seen_orders) + 1)):
            errors.append("order 不连续：%s" % seen_orders)
    return (len(errors) == 0), errors, warnings


def to_json(storyboard):
    return json.dumps(storyboard, ensure_ascii=False, indent=1)
