"""唯一入口：一句中文 → 一条视频与分镜。

用法
    python -m hm.main "一个人先向前走，然后转身，最后挥手"
    python -m hm.main "..." --k 6 --job-id myjob
    python -m hm.main --check          # 冒烟测试：三句各跑一遍，只看能不能出片

出口码
    0  出片且几何通过
    1  出片但有问题（有段被去掉 / 几何不通过）
    2  没出片

为什么要有这一层
    之前入口有两套（main 与 run_full），行为不一致，各自还维护一套选择逻辑。
    现在 main 只做参数解析与结果汇报，真正的流程一律走 run_full。
"""
import argparse
import sys
import time
from pathlib import Path

from hm import run_full

CHECK_SENTENCES = [
    "一个人向前走",
    "一个人先伸个懒腰，再弯腰，最后站起来",
    "一个人向前走，停下，举手，跳一下，转身，最后走开",
]


def run_one(sentence, job_id=None, k=6):
    t0 = time.time()
    mp4, bj = run_full.run(sentence, k=k, job_id=job_id)
    dt = time.time() - t0
    if not mp4:
        sb_p = Path(job_dir) / "storyboard.json"
        reasons = []
        if sb_p.exists():
            try:
                import json as _json
                sb = _json.loads(sb_p.read_text(encoding="utf-8"))
                reasons = [s.get("support_reason", "") for s in sb.get("segments", [])]
            except Exception:
                reasons = []
        if reasons and all(r and ("不支持" in r or "0/4" in r or "复合" in r) for r in reasons):
            print("  这句话做不了：所有动作都属于已知做不出来的类别")
            for r in reasons:
                print("    %s" % r)
        else:
            print("  没有出片：所有段都没通过检查")
        return 2, dict(ok=False, dropped=[], mp4=None)
    job_dir = Path(mp4).parent.parent
    dropped = []
    log = job_dir / "run.log"
    if log.exists():
        for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "取舍：" in ln:
                dropped.append(ln.split("取舍：")[1].split(" ")[0])
    ok = bool(bj.get("assembled_ok"))
    arts = [("视频", job_dir / "render" / "final.mp4"),
            ("裁切", job_dir / "render" / "final_cropped.mp4"),
            ("字幕", job_dir / "render" / "final_captioned.mp4"),
            ("分镜", job_dir / "分镜.png")]
    print()
    print("  原句      %s" % sentence)
    print("  任务目录  %s" % job_dir)
    print("  段/帧     %d 段 / %s 帧" % (len(bj.get("segments", [])), bj.get("total_frames")))
    print("  几何      %s" % ("通过" if ok else "不通过"))
    print("  去掉的段  %s" % ("、".join(dropped) if dropped else "无"))
    print("  交界      %s" % (["%.0f 度" % j["pose_gap_deg"] for j in bj.get("junctions", [])] or "无"))
    print("  耗时      %.1f 秒" % dt)
    for name, p in arts:
        if p.exists():
            print("  产物 %-4s %s" % (name, p))
    return (0 if (mp4 and ok and not dropped) else 1), dict(ok=ok, dropped=dropped,
                                                            mp4=str(mp4), dir=str(job_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence", nargs="?", default=None)
    ap.add_argument("--k", type=int, default=6, help="每段候选数")
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--check", action="store_true", help="冒烟测试三句")
    a = ap.parse_args()

    if a.check:
        worst = 0
        for i, s in enumerate(CHECK_SENTENCES, 1):
            print("  == 冒烟 %d/%d ==" % (i, len(CHECK_SENTENCES)))
            code, _ = run_one(s, job_id="smoke_%d" % i, k=a.k)
            worst = max(worst, code)
        print()
        print("  冒烟结果：%s" % ("全部通过" if worst == 0 else "有问题，退出码 %d" % worst))
        sys.exit(worst)

    if not a.sentence:
        ap.print_help()
        sys.exit(2)
    code, _ = run_one(a.sentence, job_id=a.job_id, k=a.k)
    sys.exit(code)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
