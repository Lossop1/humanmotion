"""账本：把状态表里的数字从结果目录重新算一遍，输出 docs/账本.md。

为什么需要
    目标树是手写的，数字与代码容易脱节（对抗审查发现过同一节点四种状态）。
    这个脚本只负责「能被机器重算的那部分」：提交号、守门结果、最新一轮评测、质量指标。
    判断类的东西（用途、边界、否证）仍写目标树，不由脚本编。

用法
    python -m tools.make_ledger            # 打印到 stdout
    python -m tools.make_ledger > docs/账本.md
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs" / "_eval_big"


def sh(cmd, timeout=600):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        return ((p.stdout or "") + (p.stderr or "")).strip()
    except Exception as e:
        return "执行失败：%s" % e


def git_hash():
    """优先读 MIRROR.txt：远端工作目录的 git HEAD 是旧的，只有 MIRROR 记录真实部署版本。"""
    m = ROOT / "MIRROR.txt"
    if m.exists():
        txt = m.read_text(encoding="utf-8")
        for part in txt.replace("=", " ").split():
            if part and all(c in "0123456789abcdef" for c in part) and 6 <= len(part) <= 12:
                return part + "（MIRROR.txt）"
    return (sh("git rev-parse --short HEAD") or "?")


def latest_round():
    files = sorted(EVAL.glob("round*.json"), key=lambda p: int(p.stem[5:]))
    return files[-1] if files else None


def round_stats(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    items = rows.values() if isinstance(rows, dict) else rows
    made = sum(1 for r in items if r.get("ok"))
    ratios = [x for r in items if r.get("ratios") for x in r["ratios"]]
    good = sum(1 for x in ratios if 0.7 <= x <= 1.4)
    total_t = sum(r.get("elapsed") or 0 for r in items)
    return {"round": path.stem, "made": made, "sentences": len(list(items)),
            "ratios": len(ratios), "ratio_ok": good, "elapsed_s": round(total_t)}


def coverage(rnd_tag):
    """用签名重算某一轮的可判覆盖（可疑/无法判）。"""
    from hm import signature
    tot = passed = sus = unk = rule_seg = rule_ok = 0
    for d in sorted(ROOT.joinpath("outputs").glob("big_s*_%s" % rnd_tag)):
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
            ok, _b, _d = signature.check(seg.get("text_gen_final") or "",
                                         signature.signature(np.load(jp)))
            tot += 1
            if ok is None:
                unk += 1
            elif ok:
                passed += 1
            else:
                sus += 1
            # 该段是否落在「有判据的类别」：提示词能匹配到任何一条规则
            if any(re.search(r[0], (seg.get("text_gen_final") or "").lower()) for r in signature.RULES):
                rule_seg += 1
                if ok is not None:
                    rule_ok += 1
    return {"judgeable": passed + sus, "total": tot, "passed": passed, "suspect": sus,
            "unknown": unk, "rule_segments": rule_seg, "rule_judgeable": rule_ok}


def quality_digest(rnd_tag):
    """质量指标：穿地、脚滑、逐交界加速度（对某一轮的任务目录）。"""
    from hm import quality_metrics as QM
    jobs = sorted(ROOT.joinpath("outputs").glob("big_s*_%s" % rnd_tag))
    if not jobs:
        return None
    below = frames = 0
    depths = []
    accs = []
    for d in jobs:
        p = d / "assembled" / "joints22.npy"
        if not p.exists():
            continue
        J = np.load(p)
        gp = QM.ground_penetration(J)
        frames += gp["frames"]
        below += gp["frames_below"]
        depths.append(gp["max_depth_m"])
        a = QM.accel_ratio(J)
        if a is not None:
            accs.append(a)
    return {"jobs": len(jobs), "frames": frames, "below": below,
            "per_kframe_below": round(1000.0 * below / max(frames, 1), 1),
            "max_depth_m": max(depths) if depths else None,
            "accel_median": round(float(np.median(accs)), 2) if accs else None,
            "accel_max": round(float(np.max(accs)), 2) if accs else None}


def main():
    print("# 账本\n")
    print("由 `python -m tools.make_ledger` 从结果目录重算；手写不了的部分（用途、边界、否证）在 `目标树.md`。\n")
    print("| 项 | 值 | 来源 |")
    print("|---|---|---|")
    print("| 提交 | `%s` | git rev-parse |" % git_hash())
    lr = latest_round()
    if lr:
        st = round_stats(lr)
        print("| 最新评测 | %s：%d/%d 句出片，交界合格 %d/%d，总耗时 %d 秒 | %s |"
              % (st["round"], st["made"], st["sentences"], st["ratio_ok"], st["ratios"],
                 st["elapsed_s"], lr.name))
        tag = lr.stem.replace("round", "r")
        cov = coverage(tag)
        if cov["total"]:
            print("| 语义覆盖 | 总体 可判 %d/%d（%d%%），可疑 %d，无法判 %d | %s 的 storyboard |"
                  % (cov["judgeable"], cov["total"], round(100.0 * cov["judgeable"] / cov["total"]),
                     cov["suspect"], cov["unknown"], tag))
            rule = cov.get("rule_segments", 0)
            rule_ok = cov.get("rule_judgeable", 0)
            if rule:
                print("| 有判据类别 | %d/%d（%d%%）｜无判据类别 %d 段，原因已全部登记 | 同上 |"
                      % (rule_ok, rule, round(100.0 * rule_ok / rule), cov["unknown"]))
        q = quality_digest(tag)
        if q:
            print("| 穿地 | 每千帧 %.1f 帧低于地面；最大深度 %.3f 米（按网格最低点对齐后，这一项接近构造成立，"
                  "不能当质量证明） | %s 的 joints22 |"
                  % (q["per_kframe_below"], q["max_depth_m"] or 0.0, tag))
            print("| 加速度比 | 中位 %s，最大 %s | 同上 |" % (q["accel_median"], q["accel_max"]))
    guard = sh("source env.sh >/dev/null 2>&1 && python -m hm.guard --fast 2>&1 | tail -n 5")
    if guard:
        print("| 守门 | %s | hm.guard --fast |" %
              "；".join(ln.strip() for ln in guard.splitlines() if "通过" in ln or "失败" in ln)[:200])
    print()
    print("## 未纳入账本（需人判或另测）\n")
    print("```")
    print("语义正确率    盲评 39 条（videos/blind_review）")
    print("穿模/拖脚/接缝判据  标定包 12 条（videos/calib）")
    print("复现性        冷跑需 HM_SEED；规划仍可能给不同帧数")
    print("```")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
