"""判据一致性报告：用「提示词」当伪标签，量每条规则的漏放与误报。

为什么能用提示词当标签
    提示词是我们自己下给生成模型的指令（"walks forward"、"turns around in place"）；
    如果模型照做了，规则就应该命中。提示词与规则不一致时，要么模型没照做，要么规则太松/太紧。
    所以这里的数字只能说明「规则与提示词是否自洽」，不能等同于人工标注的灵敏度/特异度——
    真正的灵敏度要人判（标定包在做这件事）。

用法
    python -m hm.ops.rule_report --glob "big_s*_r26"      # 标准集
    python -m hm.ops.rule_report --glob "frs_*_r1"        # 新句集
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def segments(pattern):
    out = []
    for d in sorted(ROOT.joinpath("outputs").glob(pattern)):
        sp = d / "storyboard.json"
        if not sp.exists():
            continue
        for seg in json.loads(sp.read_text(encoding="utf-8"))["segments"]:
            sel = seg.get("selected")
            if not sel:
                continue
            n = int(str(sel).split("_")[-1])
            jp = d / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
            if not jp.exists():
                continue
            out.append((d.name, seg.get("text_gen_final") or "", np.load(jp)))
    return out


def _pass(sig, rules):
    """一组规则全部成立才算这个动作做出来（与 signature.check 的 AND 语义一致）。"""
    for _pat, field, th, _why, direction in rules:
        v = sig.get(field)
        if v is None:
            return None
        if direction == "min" and v < th:
            return False
        if direction == "max" and v > th:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="big_s*_r26")
    a = ap.parse_args()
    from hm import signature

    segs = segments(a.glob)
    if not segs:
        print("  没有匹配的任务：outputs/%s" % a.glob)
        return
    # 同一个正则的多条规则属于同一个动作，合并成一组（AND）
    groups = {}
    for rule in signature.RULES:
        groups.setdefault(rule[0], []).append(rule)
    print("  段数 %d（%s）" % (len(segs), a.glob))
    print()
    print("  | 动作（按正则） | 该动作段数 | 漏放 | 非该动作段数 | 误报 | 特异度 |")
    print("  |---|---|---|---|---|---|")
    for pat, rules in groups.items():
        rx = re.compile(pat, re.I)
        pos = [(J, s) for _job, s, J in segs if rx.search(s)]
        neg = [(J, s) for _job, s, J in segs if not rx.search(s)]
        miss = sum(1 for J, _s in pos if _pass(signature.signature(J), rules) is False)
        fired = sum(1 for J, _s in neg if _pass(signature.signature(J), rules) is True)
        spec = "%.0f%%" % (100.0 * (len(neg) - fired) / max(len(neg), 1)) if neg else "—"
        print("  | %s | %d | %d | %d | %d | %s |"
              % (pat[:30], len(pos), miss, len(neg), fired, spec))
    print()
    print("  漏放 = 提示词要求这个动作但整组条件没通过；误报 = 别的动作却通过了整组条件。")
    print("  提示词当伪标签：这条只说明「规则与提示词是否自洽」，不是人工灵敏度/特异度。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
