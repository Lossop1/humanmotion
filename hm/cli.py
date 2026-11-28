"""项目唯一的命令行入口。

为什么要有这个文件
    使用者明确指出：入口只能规定在 readme，标定放在一个文件夹，别不断自创脚本。
    之前每个任务一个 hm/ops/xxx.py（19 个）加顶层几个脚本，用户根本记不住该跑哪个。
    现在所有对外操作都收在这里，README 只写 `python -m hm.cli --help` 与几个子命令。

子命令
    run        一句中文 → 出片（等价 hm.main）
    guard      守门自检（等价 hm.guard）
    selfcheck  单元级自检（等价 hm.selfcheck）
    eval       30 句标准集评测
    pack       生成人工判定包：verify（8 条全项）/ open（只含未解决项）/ random（随机句集）
    measure    单任务度量：quality / mesh / visual-jump
    report     文档类：capability / coverage / rules / ledger / test / semantic
    score      判定统计（传入判定 JSON）

用法
    python -m hm.cli --help
    python -m hm.cli run "一个人向左走" --job-id demo --k 3
    python -m hm.cli pack open
"""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_module(mod, argv):
    """以子进程等价的方式运行另一个模块（走它的 main），不重复实现。"""
    import subprocess
    return subprocess.call([sys.executable, "-m", mod] + argv, cwd=str(ROOT))


def _run_op(name, argv):
    """跑 hm.ops.<name>（原 hm/ops/*.py，已收进包里，不再对外暴露脚本路径）。"""
    import subprocess
    return subprocess.call([sys.executable, "-m", "hm.ops." + name] + argv, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(prog="hm.cli", description="人体动作生成项目唯一命令行入口")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="一句中文 → 出片")
    p.add_argument("sentence")
    p.add_argument("--job-id", default=None)
    p.add_argument("--k", default=None)
    p.add_argument("--rest", nargs=argparse.REMAINDER)

    sub.add_parser("guard", help="守门自检（八项）")
    sub.add_parser("selfcheck", help="单元级自检")

    p = sub.add_parser("eval", help="30 句标准集评测")
    p.add_argument("--round", default=None)

    p = sub.add_parser("pack", help="生成人工判定包")
    p.add_argument("kind", choices=["verify", "open", "random"])
    p.add_argument("--n", default="20")
    p.add_argument("--seed", default="20261127")
    p.add_argument("--k", default="3")

    p = sub.add_parser("measure", help="单任务度量")
    p.add_argument("kind", choices=["quality", "mesh", "visual-jump", "coverage"])
    p.add_argument("--job", default=None)
    p.add_argument("--dir", default=None)

    p = sub.add_parser("report", help="文档/汇总类报告")
    p.add_argument("kind", choices=["capability", "rules", "ledger", "test", "semantic",
                                    "transition-review"])
    p.add_argument("--rest", nargs=argparse.REMAINDER)

    p = sub.add_parser("score", help="判定统计")
    p.add_argument("--verdicts", required=True)
    p.add_argument("--categories", default=None)

    a = ap.parse_args()

    if a.cmd == "run":
        argv = [a.sentence]
        if a.job_id:
            argv += ["--job-id", a.job_id]
        if a.k is not None:
            argv += ["--k", str(a.k)]
        argv += (a.rest or [])
        return _run_module("hm.main", argv)
    if a.cmd == "guard":
        return _run_module("hm.guard", [])
    if a.cmd == "selfcheck":
        return _run_module("hm.selfcheck", [])
    if a.cmd == "eval":
        argv = ["--round", a.round] if a.round else []
        return _run_op("eval_big", argv)
    if a.cmd == "pack":
        if a.kind == "verify":
            return _run_op("make_verify_pack", ["--k", str(a.k)])
        if a.kind == "open":
            return _run_op("make_open_pack", ["--k", str(a.k)])
        return _run_op("run_random_set", ["--n", str(a.n), "--seed", str(a.seed), "--k", str(a.k)])
    if a.cmd == "measure":
        if a.kind == "quality":
            return _run_op("measure_quality", (["--glob", "outputs/%s/*" % a.job] if a.job else []))
        if a.kind == "mesh":
            return _run_op("measure_mesh", (["--glob", "outputs/%s/*" % a.job] if a.job else []))
        if a.kind == "visual-jump":
            return _run_op("visual_jump_report", (["--dir", a.dir] if a.dir else []))
        return _run_op("coverage_report", [])
    if a.cmd == "report":
        table = {"capability": "capability_report", "rules": "rule_report", "ledger": "make_ledger",
                 "test": "make_test_report", "semantic": "human_semantic_report",
                     "transition-review": "transition_review"}
        if a.kind == "capability":
            return _run_module("hm.capability_report", [])
        return _run_op(table[a.kind], a.rest or [])
    if a.cmd == "score":
        argv = ["--verdicts", a.verdicts]
        if a.categories:
            argv += ["--categories", a.categories]
        return _run_op("score_judged", argv)
    return 1


if __name__ == "__main__":
    sys.exit(main())
