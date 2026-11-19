"""读 JSON：失败时给默认值，但只吞「读不到 / 内容坏了」这两类，并把后一类说出来。

为什么要有
    原来到处是 `try: json.loads(...) except Exception: continue`。这样写会把程序自己的错误
    （拼错属性、类型不对）和「文件还没生成」混在一起吞掉，出了问题只看到结果悄悄变差。
    这里只处理 OSError（缺文件、没权限）与 ValueError（含 JSONDecodeError），其余照常抛出。
"""
import json
import sys
from pathlib import Path


def load_json(path, default=None, *, want=None, warn_on_corrupt=True):
    """读取 JSON 文件。

    path           文件路径
    default        读不到或内容坏了时返回的值
    want           期望的顶层类型（如 dict）；类型不符按「内容坏了」处理
    warn_on_corrupt  文件存在但解析失败（或类型不符）时往 stderr 打一行警告；缺文件不警告
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    except OSError as e:
        _warn(warn_on_corrupt, "无法读取 %s：%s" % (p, e))
        return default
    try:
        data = json.loads(text)
    except ValueError as e:
        _warn(warn_on_corrupt, "JSON 损坏 %s：%s" % (p, str(e)[:60]))
        return default
    if want is not None and not isinstance(data, want):
        _warn(warn_on_corrupt, "JSON 类型不符 %s：期望 %s，实际 %s" % (p, want.__name__, type(data).__name__))
        return default
    return data


def _warn(enabled, msg):
    if enabled:
        print("  警告：" + msg, file=sys.stderr)
