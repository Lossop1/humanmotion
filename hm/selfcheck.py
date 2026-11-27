"""提交前自检：一条命令跑守门 + 汇总最新评测。

用法
    python -m hm.selfcheck            # 守门六项 + 最近一轮评测摘要
    python -m hm.selfcheck --fast     # 守门跳过慢项

出口码
    0  守门全过且评测摘要无明显问题
    1  有问题（逐条列出）
"""
import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs" / "_eval_big"


def latest_round():
    files = sorted(EVAL.glob("round*.json"), key=lambda p: int(p.stem.replace("round", "")))
    return files[-1] if files else None


def summarize_eval(p):
    rows = json.loads(p.read_text(encoding="utf-8"))
    made = sum(1 for r in rows if r.get("ok"))
    gaps = [g for r in rows if r.get("junctions") for g in r["junctions"]]
    ratios = [x for r in rows if r.get("ratios") for x in r["ratios"]]
    print("  评测文件  %s（%d 句）" % (p.name, len(rows)))
    print("  出片      %d/%d" % (made, len(rows)))
    if gaps:
        print("  姿态差    中位 %.0f 度，最大 %.0f 度（只记录）" % (st.median(gaps), max(gaps)))
    if ratios:
        ok = sum(1 for x in ratios if 0.7 <= x <= 1.4)
        print("  速率比    中位 %.2f，最小 %.2f，最大 %.2f；合格 %d/%d"
              % (st.median(ratios), min(ratios), max(ratios), ok, len(ratios)))
        return made == len(rows) and ok == len(ratios)
    print("  速率比    该轮未记录（旧格式）")
    return made == len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    print("  == 守门 ==")
    cmd = [sys.executable, "-m", "hm.guard"] + (["--fast"] if a.fast else [])
    pr = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print((pr.stdout or "").rstrip())
    if pr.returncode != 0:
        print()
        print("  自检结论：守门未通过")
        sys.exit(1)

    print()
    print("  == 评测摘要 ==")
    p = latest_round()
    if p is None:
        print("  还没有评测结果（先跑 python -m hm.cli eval --round N）")
        sys.exit(0)
    ok = summarize_eval(p)
    print()
    print("  自检结论：%s" % ("通过" if ok else "评测里有不合格项"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
