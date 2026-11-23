"""盲评包：把成片随机改名，只给你看视频，不看句子。

为什么盲评
    看片时如果先看到「应该演什么」，人会被解释带走；先看片再判，得到的才是真实可用性。
    判完再用答案键算分，避免影响判断。

用法
    python -m tools.blind_review --build            # 生成 outputs/_review
    python -m tools.blind_review --score sheet.csv  # 判完算分

产物
    outputs/_review/clips/clip_NN.mp4    随机顺序、匿名文件名
    outputs/_review/sheet.csv            三列：clip,verdict(对/部分/错),note
    outputs/_review/answer_key.json      答案键（clip → 句子与任务目录，判完再看）
"""
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
REVIEW = OUT / "_review"
DIR_OVERRIDE = None


def _review_dir():
    return Path(DIR_OVERRIDE) if DIR_OVERRIDE else REVIEW

# 两批：标准集（参与过调参）与新句集（没参与过）；盲评时混在一起
BATCHES = [("std", "r20", "big_s%02d_%s"), ("fresh", "r1", "frs_f%02d_%s")]


def _sentences():
    import eval_big
    return {"std": eval_big.SENTENCES, "fresh": eval_big.sentences("fresh")}


def build(seed=20261121, limit=None):
    sents = _sentences()
    items = []
    for setname, tag, pattern in BATCHES:
        for sid, text in sents[setname]:
            job = pattern % (int(sid[1:]), tag)
            mp4 = OUT / job / "render" / "final_captioned.mp4"
            if mp4.exists():
                items.append({"set": setname, "sid": sid, "sentence": text, "job": job, "src": mp4})
    if limit:
        items = items[:limit]
    rng = random.Random(seed)
    rng.shuffle(items)
    if REVIEW.exists():
        shutil.rmtree(REVIEW)
    (REVIEW / "clips").mkdir(parents=True)
    key = {}
    for i, it in enumerate(items, 1):
        clip = "clip_%02d.mp4" % i
        shutil.copyfile(it["src"], REVIEW / "clips" / clip)
        key[clip] = {"set": it["set"], "sid": it["sid"], "sentence": it["sentence"], "job": it["job"]}
        print("  %s ← %s （%s）" % (clip, it["job"], "标准集" if it["set"] == "std" else "新句集"))
    (REVIEW / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    with (_review_dir() / "sheet.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "sentence", "verdict", "note"])
        for clip, info in key.items():
            w.writerow([clip, info["sentence"], "", ""])
    readme = REVIEW / "README.txt"
    readme.write_text(
        "盲评说明\n\n"
        "1 打开 clips/ 里的视频，按顺序看，每个只看一遍（可回看一次）。\n"
        "2 只看画面，不要猜这是哪句；如果画面里明显演了某个连续动作，记「对」。\n"
        "3 在 sheet.csv 的 verdict 列填：对 / 部分 / 错\n"
        "   对   = 画面里的动作与「一句完整的话」相符，衔接自然\n"
        "   部分 = 能看出主要动作，但缺动作或多动作混在一起 / 衔接别扭\n"
        "   错   = 看不出这句话在说什么\n"
        "4 note 列可写一句原因（例如「挥手没抬起来」「中间停顿」）。\n"
        "5 填完运行：python -m tools.blind_review --score outputs/_review/sheet.csv\n"
        "6 answer_key.json 在判完之前不要打开。\n", encoding="utf-8")
    print()
    print("  已生成 %d 个片段：%s" % (len(key), REVIEW))
    print("  判完再算分：python -m tools.blind_review --score %s" % (REVIEW / "sheet.csv"))


def score(sheet_path):
    key = json.loads((_review_dir() / "answer_key.json").read_text(encoding="utf-8"))
    # Excel 存出来的 CSV 常带 BOM，统一用 utf-8-sig 读
    # Excel 在中文 Windows 上可能存成 GBK，按顺序试几种编码
    _rows = None
    for _enc in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            with Path(sheet_path).open(encoding=_enc, newline="") as _f:
                _rows = list(csv.DictReader(_f))
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if _rows is None:
        print("  读不出 sheet（编码问题）：请用记事本另存为 UTF-8，或在 Excel 里选「CSV UTF-8」")
        return
    rows = _rows
    def _norm(c):
        c = (c or "").strip()
        return c if c.endswith(".mp4") else c + ".mp4"
    got = [(_norm(r["clip"]), (r["verdict"] or "").strip()) for r in rows]
    filled = [(c, v) for c, v in got if v in ("对", "部分", "错")]
    if not filled:
        print("  sheet 还没有判定，先在 verdict 列填 对/部分/错")
        return
    missing = len(got) - len(filled)
    if missing:
        print("  提醒：还有 %d 条没填（总 %d 条），下面的比例只按已填的算" % (missing, len(got)))
    def rate(name, items):
        n = len(items)
        ok = sum(1 for c, v in items if v == "对")
        half = sum(1 for c, v in items if v == "部分")
        print("  %-8s %2d 条：对 %d（%.0f%%）、部分 %d、错 %d"
              % (name, n, ok, 100.0 * ok / max(n, 1), half, n - ok - half))
        return ok, half, n
    print()
    print("  == 盲评结果 ==")
    all_ok = all_half = all_n = 0
    for setname in ("std", "fresh"):
        items = [(c, v) for c, v in filled if key[c]["set"] == setname]
        if items:
            ok, half, n = rate("标准集" if setname == "std" else "新句集", items)
            all_ok += ok; all_half += half; all_n += n
    rate("合计", filled)
    print()
    print("  == 判为「错」的 ==")
    for c, v in filled:
        if v == "错":
            print("    %s %-6s %s（%s）" % (c, key[c]["sid"], key[c]["sentence"], key[c]["job"]))
    print()
    print("  == 判为「部分」的 ==")
    for c, v in filled:
        if v == "部分":
            print("    %s %-6s %s（%s）" % (c, key[c]["sid"], key[c]["sentence"], key[c]["job"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dir", default=None, help="盲评包目录（默认 outputs/_review）")
    a = ap.parse_args()
    if a.dir:
        globals()["DIR_OVERRIDE"] = a.dir
    if a.build:
        build(limit=a.limit or None)
    elif a.score:
        score(a.score)
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
