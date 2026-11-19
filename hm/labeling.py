"""G10 人工标注：导出盲标表 → 人填 → 与判据结论对账，算新口径的数字。

用法
    python -m hm.labeling export <任务根目录> -o labels.csv   # 每段一行，不含判据结论
    python -m hm.labeling score  <任务根目录> labels.csv      # 对账并打印报告

任务目录约定（run_full 的产物）
    <根>/<job>/storyboard.json   每段 source_text、text_gen_final
    <根>/<job>/semantics.json    判据结论：通过 / 可疑 / 无法判 / 未选中
    <根>/<job>/final_cropped.mp4 标注用视频（无 [OK]/[?] 角标，避免影响判断）

为什么盲标
    判据结论若出现在表里，人会倾向于同意它，测出来的"一致率"就失真。
    表里只放原文片段和提示词；判据结论在 score 时才按 (job, seg) 并进来。

人判取值：是 / 否 / 部分（也接受 y/n/p）。留空的行不计入。
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from hm.jsonio import load_json

COLS = ["job", "seg", "order", "source_text", "prompt", "video", "人判(是/否/部分)", "备注"]
LABEL_COL = "人判(是/否/部分)"
_NORM = {"是": "是", "y": "是", "yes": "是", "对": "是",
         "否": "否", "n": "否", "no": "否", "错": "否",
         "部分": "部分", "p": "部分"}


def seg_short(segment_id):
    """semantics.json 用 segment_id 的末三位当键（见 hm.main.verify_segments）。"""
    return str(segment_id)[-3:]


def export_rows(root):
    rows = []
    for jd in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        sb = load_json(jd / "storyboard.json", want=dict)
        if not sb:
            continue
        video = jd / "final_cropped.mp4"
        for i, s in enumerate(sb.get("segments", []), 1):
            if not s.get("selected"):
                continue   # 没有被选中的段不会出现在视频里，没东西可标
            rows.append({"job": jd.name, "seg": seg_short(s.get("segment_id", "")), "order": i,
                         "source_text": s.get("source_text") or s.get("source") or "",
                         "prompt": s.get("text_gen_final") or s.get("prompt") or "",
                         "video": str(video) if video.exists() else "",
                         LABEL_COL: "", "备注": ""})
    return rows


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:   # BOM：Excel 直接打开不乱码
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def read_labels(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            raw = (r.get(LABEL_COL) or "").strip().lower()
            if not raw:
                continue
            if raw not in _NORM:
                raise ValueError("第 %s 行人判取值无法识别：%r（只接受 是/否/部分）" % (r.get("job"), raw))
            out[(r["job"], r["seg"])] = {"label": _NORM[raw], "action": r.get("prompt", ""),
                                         "source_text": r.get("source_text", "")}
    return out


def score(root, labels):
    """返回统计字典。status 来自 semantics.json，按 (job, seg) 对账。"""
    status = {}
    for jd in Path(root).iterdir():
        sem = load_json(jd / "semantics.json", default=[], want=list)
        for e in sem:
            status[(jd.name, e.get("segment"))] = e.get("status", "?")
    by = defaultdict(Counter)
    missing = []
    for k, v in labels.items():
        st = status.get(k)
        if st is None:
            missing.append(k)
            continue
        by[st][v["label"]] += 1
    n = sum(sum(c.values()) for c in by.values())
    total = Counter()
    for c in by.values():
        total.update(c)

    def rate(c, good=("是",)):
        t = sum(c.values())
        return (sum(c[g] for g in good) / t) if t else None

    passed = by.get("通过", Counter())
    suspect = by.get("可疑", Counter())
    return {
        "n": n, "missing": missing, "by_status": {k: dict(v) for k, v in by.items()},
        "correct_rate": rate(total), "correct_or_partial_rate": rate(total, ("是", "部分")),
        # 漏放：判据说通过，人判为"否"（目标树 G7：≤10%）
        "leak_rate": (passed["否"] / sum(passed.values())) if passed else None,
        "leak_n": (passed["否"], sum(passed.values())),
        # 误报：判据说可疑，人判为"是"
        "false_alarm_rate": (suspect["是"] / sum(suspect.values())) if suspect else None,
        "false_alarm_n": (suspect["是"], sum(suspect.values())),
        "unjudgeable_share": (sum(by.get("无法判", Counter()).values()) / n) if n else None,
    }


def _pct(x):
    return "—" if x is None else "%.1f%%" % (100 * x)


def format_report(r):
    L = ["人工标注对账（n=%d）" % r["n"],
         "  人判正确率（是）        %s" % _pct(r["correct_rate"]),
         "  人判正确或部分          %s" % _pct(r["correct_or_partial_rate"]),
         "  语义漏放（通过→人判否） %s（%d/%d，目标 ≤10%%）" % ((_pct(r["leak_rate"]),) + tuple(r["leak_n"])),
         "  误报（可疑→人判是）     %s（%d/%d）" % ((_pct(r["false_alarm_rate"]),) + tuple(r["false_alarm_n"])),
         "  无法判占比              %s" % _pct(r["unjudgeable_share"]),
         "  分状态人判分布："]
    for st, c in sorted(r["by_status"].items()):
        L.append("    %-6s %s" % (st, "  ".join("%s=%d" % kv for kv in sorted(c.items()))))
    if r["missing"]:
        L.append("  注意：%d 条标注在 semantics.json 里找不到对应段，未计入：%s"
                 % (len(r["missing"]), r["missing"][:5]))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("root")
    e.add_argument("-o", "--out", default="labels.csv")
    s = sub.add_parser("score")
    s.add_argument("root")
    s.add_argument("labels")
    s.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        rows = export_rows(a.root)
        write_csv(a.out, rows)
        print("导出 %d 段 → %s" % (len(rows), a.out))
        return 0
    r = score(a.root, read_labels(a.labels))
    print(json.dumps(r, ensure_ascii=False, indent=1) if a.json else format_report(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
