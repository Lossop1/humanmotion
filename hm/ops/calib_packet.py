"""标定包：把需要人判的片段整理成一张表（穿模 / 拖脚 / 接缝）。

为什么需要人判
    穿模、拖脚、接缝这三件事机器只能给出度量值，说不出「人看不看得出来」；
    判据阈值必须落在人工标注上（看到 / 没看到），否则就是自证。

用法
    python -m hm.ops.calib_packet --build     # 生成 outputs/_calib
    python -m hm.ops.calib_packet --score outputs/_calib/sheet.csv
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CAL = OUT / "_calib"

QUESTIONS = {
    "penetration": ("有没有明显穿模（身体部位互相插进去、或脚陷进地面）？", ["有", "轻微", "没有"]),
    "footslide": ("脚踩在地上时，有没有明显横向滑动（拖脚）？", ["有", "轻微", "没有"]),
    "seam": ("两段衔接的地方，有没有明显顿挫或突跳？", ["明显", "轻微", "没有"]),
}

# 选片依据写在这里，便于复核：每条都标了机器指标
SPEC = [
    ("big_s17_r20", "penetration", "关节最低 −0.861 米（标准集最大穿地）"),
    ("big_s19_r20", "penetration", "关节穿地 0.019 米（中位水平）"),
    ("big_s30_r20", "penetration", "关节穿地 0.005 米（几乎无）"),
    ("big_s08_r20", "penetration", "网格穿地 0.065 米（网格法偏大）"),
    ("fs_inplace", "footslide", "接触帧 0（双脚没落地）"),
    ("fs_walk", "footslide", "脚滑中位 9.6 mm/帧"),
    ("fs_run", "footslide", "脚滑中位 4.2 mm/帧"),
    ("big_s17_r20", "footslide", "脚滑 p90 28.1 mm/帧（最大）"),
    ("big_s03_r20", "seam", "交界加速度比 1.72、姿态差 85.8°（最猛）"),
    ("big_s26_r20", "seam", "交界加速度比 0.18（最顿）"),
    ("big_s08_r20", "seam", "交界加速度比 1.23、姿态差 11.9°（中间水平）"),
    ("big_s19_r20", "seam", "交界 25 帧过渡、姿态差 79.2°（长过渡）"),
]


def build(limit=None):
    if not CAL.exists():
        CAL.mkdir(parents=True)
    clips = CAL / "clips"
    if clips.exists():
        shutil.rmtree(clips)
    clips.mkdir(parents=True)
    rows = []
    key = {}
    spec = SPEC[:limit] if limit else SPEC
    for i, (job, cat, why) in enumerate(spec, 1):
        src = OUT / job / "render" / "final_captioned.mp4"
        if not src.exists():
            src = OUT / job / "render" / "final.mp4"
        if not src.exists():
            print("  跳过（没有视频）：%s" % job)
            continue
        clip = "clip_%02d.mp4" % i
        shutil.copyfile(src, clips / clip)
        q, opts = QUESTIONS[cat]
        rows.append({"clip": clip, "question": q, "answer": "", "note": "", "options": "/".join(opts)})
        key[clip] = {"job": job, "category": cat, "metric": why}
        print("  %s ← %-14s [%s] %s" % (clip, job, cat, why))
    with (CAL / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "question", "answer", "note", "options"])
        w.writeheader()
        w.writerows(rows)
    (CAL / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (CAL / "README.txt").write_text(
        "标定说明\n\n"
        "1 看 clips 里的视频，逐行在 sheet.csv 的 answer 列填「有 / 轻微 / 没有」（接缝那几行填 明显 / 轻微 / 没有）。\n"
        "2 note 列选填：写一句你看到的现象（例：右手插进胸口、脚在地上滑行、衔接处顿了一下）。\n"
        "3 填完运行：python -m hm.ops.calib_packet --score outputs/_calib/sheet.csv\n"
        "   它会给出每一类的「人判分布 vs 机器指标」，并告诉我阈值该定在哪。\n"
        "4 answer_key.json 里是每条片子的机器指标与来源，判完再看。\n", encoding="utf-8")
    print()
    print("  已生成 %d 条：%s" % (len(key), CAL))


def score(sheet_path):
    key = json.loads((CAL / "answer_key.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(Path(sheet_path).open(encoding="utf-8-sig")))
    bycat = {}
    for r in rows:
        c = (r["clip"] or "").strip()
        c = c if c.endswith(".mp4") else c + ".mp4"
        a = (r.get("answer") or "").strip()
        if c not in key or a not in ("有", "轻微", "没有", "明显"):
            continue
        cat = key[c]["category"]
        bycat.setdefault(cat, []).append((a, key[c]["metric"]))
    if not bycat:
        print("  sheet 还没填（answer 列填 有/轻微/没有）")
        return
    for cat, items in bycat.items():
        print("  == %s ==" % cat)
        for a, m in items:
            print("    人判 %-4s ← %s" % (a, m))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", default=None)
    a = ap.parse_args()
    if a.build:
        build(limit=a.limit or None)
    elif a.score:
        score(a.score)
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
