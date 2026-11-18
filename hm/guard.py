"""改动守门：改完代码或文档之后跑一次，专抓这几轮反复犯的两类错。

抓什么
    一、同一功能两套实现 / 改了却没生效 → 跑完整句子，看出口码与关键日志，
        而不是相信代码看起来对。
    二、文档写了却没落盘 → 读回文件逐条断言标题存在。

检查项
    1  能出片            「一个人向前走」出口码 0
    2  多段能出片        「走、停、举手、跳、转身、走开」几何通过、无丢段
    3  做不了的句子要报清楚 「蹲下、站起来」出口码 2 且提示「做不了」
    4  文档章节存在      G16–G20 标题都在
    5  文档没有测试残留  「测试标记」不应出现

用法
    python -m hm.guard          # 全部跑
    python -m hm.guard --fast   # 跳过第 2 项（最慢）
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "目标树.md"
REQUIRED_HEADINGS = ["### G16 ", "### G17 ", "### G18 ", "### G19 ", "### G20 "]

CASES = [
    ("single", "一个人向前走", 0, []),
    ("multi", "一个人向前走，停下，举手，跳一下，转身，最后走开", None, []),
    ("dead_end", "一个人爬行", 2, ["做不了"]),
]

PLAN_CASE = "一个人向前走，停下，举手，跳一下，转身，最后走开"


def check_merged_actions(job_id="guard_multi"):
    """合并段必须两个动作都做出来——这是语义承诺，要有回归保护。"""
    import json as _json

    import numpy as np
    from hm import signature

    d = ROOT / "outputs" / job_id
    sb_p = d / "storyboard.json"
    if not sb_p.exists():
        return False, "没有 %s 的结果可查" % job_id
    sb = _json.loads(sb_p.read_text(encoding="utf-8"))
    checked = missing = 0
    detail = []
    for seg in sb.get("segments", []):
        if not seg.get("merged_from") or not seg.get("selected"):
            continue
        n = int(str(seg["selected"]).split("_")[-1])
        jp = d / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
        if not jp.exists():
            continue
        checked += 1
        sig = signature.signature(np.load(jp))
        fails = []
        for part in (seg.get("text_gen_final") or "").split(" and "):
            part = part.strip()
            body = part if part.lower().startswith("a person") else "a person " + part
            ok, _bad, _det = signature.check(body, sig)
            if ok is False:
                fails.append(part[:24])
        if fails:
            missing += 1
            detail.append("%s 缺 %s" % (seg["segment_id"], "/".join(fails)))
    if checked == 0:
        return False, "没有合并段可查（合并逻辑可能没生效）"
    return missing == 0, "合并段 %d 个，动作缺失 %d 个%s" % (
        checked, missing, "" if not detail else "（%s）" % "；".join(detail[:2]))


def check_fresh(job_id=None):
    """新句集（未参与调参）上的可疑数必须为 0，否则判为过拟合或回归。"""
    pr = subprocess.run([sys.executable, "-m", "hm.capability_report", "--set", "fresh"],
                        cwd=str(ROOT), capture_output=True, text=True)
    out = (pr.stdout or "") + (pr.stderr or "")
    import re as _re
    m = _re.search(r"其中可疑\s+(\d+)", out)
    if not m:
        return False, "没读到新句集的统计"
    sus = int(m.group(1))
    m2 = _re.search(r"语义覆盖（可判/总）\s+(\S+)", out)
    cov = m2.group(1) if m2 else "?"
    return sus == 0, "新句集可疑 %d（覆盖 %s）" % (sus, cov)


def check_plan_consistency(job_id="guard_plan"):
    """预测的合并数与实际合并段数对不上时，运行日志里必须有解释。"""
    pr = subprocess.run([sys.executable, "-m", "hm.main", PLAN_CASE,
                         "--compare", "--k", "4", "--job-id", job_id],
                        cwd=str(ROOT), capture_output=True, text=True)
    out = (pr.stdout or "") + (pr.stderr or "")
    import re
    m = re.search(r"预测合并 (\d+) 处，实际合并段 (\d+) 个", out)
    if not m:
        return False, "没找到对照行"
    pred, act = int(m.group(1)), int(m.group(2))
    if pred == act:
        return True, "预测 %d 处合并，实际 %d 个，一致" % (pred, act)
    log = ROOT / "outputs" / (job_id + "_run") / "run.log"
    explained = log.exists() and ("退回未合并" in log.read_text(encoding="utf-8", errors="replace"))
    return explained, "预测 %d 处、实际 %d 个；%s" % (
        pred, act, "日志有解释" if explained else "日志没有解释")


def run_cli(sentence, job_id):
    pr = subprocess.run([sys.executable, "-m", "hm.main", sentence, "--job-id", job_id],
                        cwd=str(ROOT), capture_output=True, text=True)
    return pr.returncode, (pr.stdout or "") + (pr.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    failures = []

    # 清掉上次遗留的任务目录：不清会把旧日志与旧状态带进来，造成假失败/假通过
    import shutil
    for _d in (ROOT / "outputs").glob("guard_*"):
        if _d.is_dir():
            shutil.rmtree(_d, ignore_errors=True)

    for name, sentence, want_code, want_words in CASES:
        if a.fast and name == "multi":
            continue
        code, out = run_cli(sentence, "guard_%s" % name)
        ok = True
        if want_code is not None and code != want_code:
            ok = False
            failures.append("%s：出口码 %d，应为 %d" % (name, code, want_code))
        for w in want_words:
            if w not in out:
                ok = False
                failures.append("%s：输出里没有「%s」" % (name, w))
        if name == "multi":
            if "几何      通过" not in out:
                ok = False
                failures.append("multi：几何未通过")
            if "去掉的段  -" in out or "去掉的段  无" in out:
                pass
            elif "去掉的段" in out:
                seg = out.split("去掉的段")[1].splitlines()[0].strip()
                if seg not in ("无", "-"):
                    ok = False
                    failures.append("multi：有段被去掉（%s）" % seg)
        print("  %-9s %s  出口码 %d" % (name, "通过" if ok else "失败", code), flush=True)

    if not DOC.exists():
        print("  %-9s 跳过（本机无文档副本，文档只在本地仓库维护）" % "doc")
        text = None
    else:
        text = DOC.read_text(encoding="utf-8")
    miss = [] if text is None else [h.strip() for h in REQUIRED_HEADINGS if h not in text]
    if miss:
        failures.append("文档缺章节：%s" % "、".join(miss))
    if text is not None and "测试标记" in text:
        failures.append("文档里有测试残留")
    if text is not None:
        print("  %-9s %s  %d 行" % ("doc", "通过" if not miss and "测试标记" not in text else "失败",
                                    len(text.splitlines())))

    if not a.fast:
        ok, msg = check_merged_actions()
        if not ok:
            failures.append("merged：%s" % msg)
        print("  %-9s %s  %s" % ("merged", "通过" if ok else "失败", msg))

        ok, msg = check_fresh()
        if not ok:
            failures.append("fresh：%s" % msg)
        print("  %-9s %s  %s" % ("fresh", "通过" if ok else "失败", msg))

        ok, msg = check_plan_consistency()
        if not ok:
            failures.append("plan：%s" % msg)
        print("  %-9s %s  %s" % ("plan", "通过" if ok else "失败", msg))

    print()
    if failures:
        print("  守门未通过：")
        for f in failures:
            print("    - %s" % f)
        sys.exit(1)
    print("  守门通过：能出片、多段通过、做不了的句子报得清楚%s"
          % ("、文档章节齐全" if text is not None else ""))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
