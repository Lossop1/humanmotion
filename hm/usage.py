"""用量统计：把 outputs/_usage.jsonl 汇总成报告要有的一行数。

用法
    python -m hm.usage
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs" / "_usage.jsonl"


def load():
    if not LOG.exists():
        return []
    return [json.loads(ln) for ln in LOG.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main():
    rows = load()
    if not rows:
        print("  还没有调用记录（跑一次带规划的句子就会有）")
        return
    pt = [r.get("usage", {}).get("prompt_tokens") for r in rows if r.get("usage")]
    ct = [r.get("usage", {}).get("completion_tokens") for r in rows if r.get("usage")]
    print("  LLM 调用 %d 次" % len(rows))
    print("  提示字符 中位 %d｜回复字符 中位 %d"
          % (st.median([r["prompt_chars"] for r in rows]), st.median([r["reply_chars"] for r in rows])))
    if pt:
        print("  prompt tokens 合计 %d，completion tokens 合计 %d" % (sum(pt), sum(ct)))
    else:
        print("  接口没有返回 usage，token 数不可得（只有字符数）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()