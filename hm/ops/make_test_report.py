"""生成测试报告（Markdown）：所有数字现场重算，不手抄。

用法（在远端 GPU 机器上跑，数据都在那里）
    python -m hm.ops.make_test_report > docs/测试报告.md

包含
    1 测试环境
    2 单元测试与包结构
    3 集成守门（七项）
    4 五臂消融
    5 换后端演练
    6 判据校准与覆盖
    7 能力表
    8 未验证项（如实列出）
"""
import datetime
import json
import platform
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "eval" if (ROOT / "outputs" / "eval").exists() else ROOT / "outputs"
EVAL = ROOT / "outputs" / "_eval_big"


def sh(cmd, timeout=1800):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return "执行失败：%s" % e


def env_section():
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU"
    return [
        "| 项目 | 值 |", "|---|---|",
        "| 操作系统 | %s |" % platform.platform(),
        "| Python | %s |" % platform.python_version(),
        "| PyTorch | %s |" % torch.__version__,
        "| GPU | %s |" % gpu,
        "| CUDA 可用 | %s |" % torch.cuda.is_available(),
        "| 报告生成时间 | %s |" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    ]


def unit_section():
    out = sh("%s -m pytest tests -q" % sys.executable, timeout=900)
    m = re.search(r"(\d+) passed", out)
    n = m.group(1) if m else "（未取到汇总行）"
    tail = "\n".join(out.strip().splitlines()[-3:])
    return ["通过用例：%s" % n, "", "```", tail, "```"]


def guard_section():
    out = sh("%s -m hm.guard" % sys.executable)
    lines = [ln for ln in out.splitlines() if re.search(r"通过|失败|守门", ln)]
    return ["```", "\n".join(lines), "```"]


ARMS = [(20, "主臂：大模型 + k=6 + 合并 + 交界修复（基线）"),
        (22, "k=1：每段只生成 1 个候选"),
        (23, "不合并相邻短段"),
        (24, "不做交界修复"),
        (21, "规则兜底：不使用大模型规划")]


def ablation_section():
    from hm import signature

    def one(rnd):
        p = EVAL / ("round%d.json" % rnd)
        if not p.exists():
            return None
        rows = json.loads(p.read_text(encoding="utf-8"))
        items = rows.values() if isinstance(rows, dict) else rows
        made = sum(1 for r in items if r.get("ok"))
        ratios = [x for r in items if r.get("ratios") for x in r["ratios"]]
        good = sum(1 for x in ratios if 0.7 <= x <= 1.4)
        total_t = sum(r["elapsed"] for r in items if r.get("elapsed"))
        tot = sel = sus = unk = 0
        for d in sorted(ROOT.joinpath("outputs").glob("big_s*_r%d" % rnd)):
            sp = d / "storyboard.json"
            if not sp.exists():
                continue
            for seg in json.loads(sp.read_text(encoding="utf-8"))["segments"]:
                tot += 1
                s0 = seg.get("selected")
                if not s0:
                    continue
                sel += 1
                n = int(str(s0).split("_")[-1])
                jp = d / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
                if not jp.exists():
                    continue
                ok, _b, _d = signature.check(seg.get("text_gen_final") or "",
                                             signature.signature(np.load(jp)))
                if ok is None:
                    unk += 1
                elif not ok:
                    sus += 1
        return dict(made=made, tot=tot, sel=sel, sus=sus, unk=unk,
                    good=good, ratios=len(ratios), t=total_t)

    out = ["| 臂 | 出片 | 段（规划/出片） | 可疑 / 无法判 | 交界速率比合格 | 总耗时 |",
           "|---|---|---|---|---|---|"]
    for rnd, name in ARMS:
        r = one(rnd)
        if not r:
            out.append("| %s | 缺 round%d.json | | | | |" % (name, rnd))
            continue
        out.append("| %s | %d/30 | %d/%d | %d/%d | %d/%d | %.0f 秒 |"
                   % (name, r["made"], r["tot"], r["sel"], r["sus"], r["unk"],
                      r["good"], r["ratios"], r["t"]))
    out += ["",
            "读表注意：耗时不可横比（k=1 与兜底臂是新提示词、缓存未命中，主臂命中缓存）；",
            "「不合并」臂不是纯单一变量——它同时多触发几次重试（帧数 +8）。"]
    return out


def swap_section():
    rows = []
    for m in ("momask", "mdm"):
        d = ROOT / "outputs" / ("swap_%s" % m)
        bj_p = d / "assembled" / "bounds.json"
        if not bj_p.exists():
            continue
        bj = json.loads(bj_p.read_text(encoding="utf-8"))
        gaps = [j["pose_gap_deg"] for j in bj.get("junctions", [])]
        rows.append("| %s | %d 段 / %s 帧 | %s |" % (
            m, len(bj.get("segments", [])), bj.get("total_frames"),
            "/".join("%.0f 度" % g for g in gaps) or "无"))
    if not rows:
        return ["本次未找到换后端演练的结果（先跑 HM_FORCE_BACKEND=mdm 的一句）。"]
    return ["| 后端 | 结果 | 交界 |", "|---|---|---|"] + rows


def coverage_section():
    from hm import signature
    tot = passed = suspect = unknown = 0
    rnd = 20
    for d in sorted(ROOT.joinpath("outputs").glob("big_s*_r%d" % rnd)):
        sp = d / "storyboard.json"
        if not sp.exists():
            continue
        for seg in json.loads(sp.read_text(encoding="utf-8"))["segments"]:
            s0 = seg.get("selected")
            if not s0:
                continue
            n = int(str(s0).split("_")[-1])
            jp = d / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
            if not jp.exists():
                continue
            ok, _b, _d = signature.check(seg.get("text_gen_final") or "",
                                         signature.signature(np.load(jp)))
            tot += 1
            if ok is None:
                unknown += 1
            elif ok:
                passed += 1
            else:
                suspect += 1
    return ["轮%d 的 %d 段：通过 %d、可疑 %d、无法判 %d（可判覆盖 %.0f%%）"
            % (rnd, tot, passed, suspect, unknown, 100.0 * (passed + suspect) / max(tot, 1))]


def main():
    blocks = []
    blocks.append("# 测试报告\n")
    blocks.append("本报告的数字全部由 `hm/ops/make_test_report.py` 现场重算，不手抄。\n")
    blocks.append("## 1 测试环境\n")
    blocks += env_section()
    blocks.append("\n## 2 单元测试与包结构\n")
    blocks += unit_section()
    blocks.append("\n## 3 集成守门（七项）\n")
    blocks += guard_section()
    blocks.append("\n## 4 五臂消融（同一批 30 句，只动一个杠杆）\n")
    blocks += ablation_section()
    blocks.append("\n## 5 换后端演练\n")
    blocks += swap_section()
    blocks.append("\n## 6 判据校准与语义覆盖\n")
    blocks += coverage_section()
    blocks.append("\n校准表可用 `python -m hm.calibrate_signature` 重生成；"
                  "判据量登记见目标树 G29。\n")
    blocks.append("\n## 7 未验证项（如实列出）\n")
    blocks.append("1 语义正确率的人判（盲评 39 个片段）尚未完成；")
    blocks.append("2 并发/多用户未测；")
    blocks.append("3 真实设备部署（非本机 GPU）未测；")
    blocks.append("4 长文（>300 字）未做覆盖率统计。")
    print("\n".join(blocks))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
