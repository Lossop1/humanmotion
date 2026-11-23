"""唯一入口：一句中文 → 一条视频与分镜。

用法
    python -m hm.main "一个人先向前走，然后转身，最后挥手"
    python -m hm.main "..." --k 6 --job-id myjob
    python -m hm.main --explain "一句话"   # 只做规划，说明这句话能保证到什么程度
    python -m hm.main --check             # 冒烟测试：三句各跑一遍
    python -m hm.main --batch 文件        # 批量：每行一句

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


def verify_segments(job_dir, sb):
    """逐段做语义验证：可判且通过 / 可疑 / 无法判。返回 [(段号, 状态, 说明)]。"""
    import numpy as _np

    from hm import signature
    out = []
    for seg in sb.get("segments", []):
        sel = seg.get("selected")
        if not sel:
            out.append((seg["segment_id"][-3:], "未选中", ""))
            continue
        n = int(str(sel).split("_")[-1])
        jp = Path(job_dir) / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
        if not jp.exists():
            out.append((seg["segment_id"][-3:], "未选中", ""))
            continue
        prompt = seg.get("text_gen_final") or ""
        sig = signature.signature(_np.load(jp))
        # 合并段要按拆开的两个动作分别查：任一子动作「无法判」时，整段只能算「未验证」，
        # 不能算「通过」——否则一个没做出来的挥手会被 merged 的另一半掩盖过去。
        # 规划层自己标注的「近似：…」同样不算通过：这说明原文动作做不出来
        _label = (seg.get("label") or "")
        if _label.startswith("近似"):
            out.append((seg["segment_id"][-3:], "未验证", _label[:40]))
            continue
        _dead = signature.dead_end_in(prompt)
        if _dead:
            out.append((seg["segment_id"][-3:], "未验证", "含做不出来的动作：%s（已近似）" % _dead))
            continue
        # 近似代替（原文动作做不出来）同样算未验证，不能报「通过」
        if seg.get("approx_from_unsupported"):
            out.append((seg["segment_id"][-3:], "未验证", "原文动作做不出来，已用近似动作代替"))
            continue
        if seg.get("merged_from"):
            parts = [p.strip() for p in prompt.split(" and ") if p.strip()]
            statuses = []
            for part in parts:
                body = part if part.lower().startswith("a person") else "a person " + part
                ok, bad, _det = signature.check(body, sig)
                statuses.append((part[:24], ok, bad))
            if any(ok is False for _p, ok, _b in statuses):
                bad_names = "、".join(b[0] for _p, ok, b in statuses if ok is False)
                out.append((seg["segment_id"][-3:], "可疑", bad_names))
            elif any(ok is None for _p, ok, _b in statuses):
                miss = "、".join(p for p, ok, _b in statuses if ok is None)
                out.append((seg["segment_id"][-3:], "未验证", "子动作无法判：" + miss))
            else:
                out.append((seg["segment_id"][-3:], "通过", ""))
        else:
            ok, bad, _det = signature.check(prompt, sig)
            if ok is None:
                out.append((seg["segment_id"][-3:], "无法判", "无校准量"))
            elif ok:
                out.append((seg["segment_id"][-3:], "通过", ""))
            else:
                out.append((seg["segment_id"][-3:], "可疑", "、".join(b[0] for b in bad)))
    return out


def run_one(sentence, job_id=None, k=6):
    t0 = time.time()
    mp4, bj, job_dir_str = run_full.run(sentence, k=k, job_id=job_id)
    dt = time.time() - t0
    job_dir = Path(job_dir_str)

    if not mp4:
        # 先分清是「这句话做不了」还是「都没通过检查」，两者对使用者意义不同
        reasons = []
        sb_p = job_dir / "storyboard.json"
        if sb_p.exists():
            try:
                import json as _json
                sb = _json.loads(sb_p.read_text(encoding="utf-8"))
                reasons = [seg.get("support_reason") or "" for seg in sb.get("segments", [])]
            except Exception:
                reasons = []
        if reasons and all(r and ("不支持" in r or "0/4" in r or "复合" in r) for r in reasons):
            print("  这句话做不了：所有动作都属于已知做不出来的类别")
            for r in reasons:
                print("    %s" % r)
        else:
            print("  没有出片：所有段都没通过检查")
        return 2, dict(ok=False, dropped=[], mp4=None, dir=str(job_dir))

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
    try:
        import json as _json2
        sb2 = _json2.loads((job_dir / "storyboard.json").read_text(encoding="utf-8"))
        status = verify_segments(job_dir, sb2)
        passed = sum(1 for _s, st, _m in status if st == "通过")
        unknown = [s for s, st, _m in status if st in ("无法判", "未验证")]
        suspect = [s for s, st, m in status if st == "可疑"]
        print("  语义验证  %d 段通过 / %d 段未验证 / %d 段可疑（共 %d 段）%s%s"
              % (passed, len(unknown), len(suspect), len(status),
                 "；未验证 %s" % "/".join(unknown) if unknown else "",
                 "；可疑 %s" % "/".join(suspect) if suspect else ""))
        _approx_n = sum(1 for seg in sb2.get("segments", []) if seg.get("approx_from_unsupported"))
        _trim = sum(int(seg.get("trim_tail") or 0) for seg in sb2.get("segments", []))
        if _approx_n:
            print("  近似代替  %d 段：原文动作做不出来，用最接近的动作代替（不算通过）" % _approx_n)
        if _trim:
            print("  截尾      %d 帧（交界过大时截掉前段末尾，已在成片体现）" % _trim)
        (job_dir / "semantics.json").write_text(_json2.dumps(
            [dict(segment=s, status=st, detail=m) for s, st, m in status],
            ensure_ascii=False, indent=1), encoding="utf-8")
        # 分镜在 semantics.json 之前生成，标不上状态；这里重画一次
        try:
            import contextlib as _cl
            import io as _io2
            import job_report as _jr
            with _cl.redirect_stdout(_io2.StringIO()):
                _jr.report(str(job_dir))
        except Exception:
            pass
    except Exception as _e:
        print("  语义验证  跳过（%s）" % str(_e)[:40])
    for name, p in arts:
        if p.exists():
            print("  产物 %-4s %s" % (name, p))
    # 有可疑段就不再算「正常」：可疑意味着这一段做不出它该做的动作
    _suspect_n = sum(1 for _s, st, _m in status if st == "可疑") if "status" in dir() else 0
    _bad = _suspect_n + _approx_n
    return (0 if (mp4 and ok and not dropped and not _bad) else 1), dict(ok=bool(ok), dropped=dropped,
                                                            mp4=str(mp4), dir=str(job_dir))


def explain(sentence, job_id=None):
    """说明这句话能保证到什么程度：分段、可支持性、能判什么、哪些会合并。"""
    import contextlib
    import io as _io
    import re as _re

    from hm import signature

    _buf = _io.StringIO()
    with contextlib.redirect_stdout(_buf):
        job, sb = run_full.plan_only(sentence, job_id=job_id or "explain")
    segs = sb["segments"]

    # 合并对：相邻两段都不超过 56 帧就会合并
    pairs, i = [], 0
    while i + 1 < len(segs):
        a = int(segs[i].get("duration_frames") or 40)
        b = int(segs[i + 1].get("duration_frames") or 40)
        if a <= 56 and b <= 56:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    first = {a for a, _ in pairs}
    second = {b for _, b in pairs}

    print()
    print("  原句      %s" % sentence)
    print("  规划      %d 段（其中 %d 处会合并 → 成片约 %d 段）"
          % (len(segs), len(pairs), len(segs) - len(pairs)))
    print()
    print("  段    帧  支持      能判什么                              提示词")
    dead = judgeable = 0
    for idx, seg in enumerate(segs):
        prompt = seg.get("text_gen_final") or ""
        status = seg.get("support_status") or "?"
        fields = [f for pat, f, _th, _w, _k in signature.RULES
                  if _re.search(pat, prompt.lower())]
        if status == "unsupported":
            dead += 1
        if fields:
            judgeable += 1
        note = ""
        if idx in first:
            note = "  ⇄ 与下一段合并"
        elif idx in second:
            note = "  ⇧ 与上一段合并"
        print("  %-5s %3s  %-8s %-38s %s%s"
              % (seg["segment_id"][-3:], seg.get("duration_frames"),
                 "不支持" if status == "unsupported" else "可支持",
                 "/".join(fields) if fields else
                 ("无法判（%s）" % signature.why_unjudgeable(prompt)
                  if signature.why_unjudgeable(prompt) else "无法判（无校准量）"),
                 prompt[:36], note))
    print()
    if dead == len(segs):
        print("  结论  这句话做不了：所有段都属于已知做不出来的动作")
    elif dead:
        print("  结论  有 %d 段做不出来会被去掉；其余 %d 段出片" % (dead, len(segs) - dead))
    else:
        print("  结论  全部段都有生成先例；%d 段可用校准量检查，%d 段只能做几何检查"
              % (judgeable, len(segs) - judgeable))
    print("  任务目录  %s" % job.dir)
    return 0


def compare(sentence, k=6, job_id=None):
    """预测对照：--explain 说的与实际跑出来的并排看。

    预测与交付对不上，说明说明书写错了（例如合并规则、可支持性判定），这比视频难看更严重。
    """
    import contextlib
    import io as _io
    import json as _json
    import re as _re

    from hm import signature

    base = job_id or "cmp"
    _buf = _io.StringIO()
    with contextlib.redirect_stdout(_buf):
        pjob, psb = run_full.plan_only(sentence, job_id=base + "_plan")
    pred = {}
    for i, seg in enumerate(psb["segments"]):
        prompt = seg.get("text_gen_final") or ""
        fields = [f for pat, f, _th, _w, _k in signature.RULES if _re.search(pat, prompt.lower())]
        pred[i] = dict(prompt=prompt, support=seg.get("support_status"),
                       fields=fields, frames=seg.get("duration_frames"))
    pairs, i = [], 0
    while i + 1 < len(psb["segments"]):
        a = int(psb["segments"][i].get("duration_frames") or 40)
        b = int(psb["segments"][i + 1].get("duration_frames") or 40)
        if a <= 56 and b <= 56:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    print()
    print("  原句          %s" % sentence)
    print("  预测          规划 %d 段，%d 处合并 → 成片约 %d 段；可判 %d 段"
          % (len(psb["segments"]), len(pairs), len(psb["segments"]) - len(pairs),
             sum(1 for v in pred.values() if v["fields"])))

    code, info = run_one(sentence, job_id=base + "_run", k=k)
    d = Path(info.get("dir", ""))
    rows = []
    sb2 = _json.loads((d / "storyboard.json").read_text(encoding="utf-8"))
    bj = _json.loads((d / "assembled" / "bounds.json").read_text(encoding="utf-8")) \
        if (d / "assembled" / "bounds.json").exists() else {}
    for seg in sb2["segments"]:
        sel = seg.get("selected")
        verdict = "未选中"
        if sel:
            n = int(str(sel).split("_")[-1])
            jp = d / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
            if jp.exists():
                ok, bad, _det = signature.check(seg.get("text_gen_final") or "",
                                                signature.signature(__import__("numpy").load(jp)))
                verdict = "无法判" if ok is None else ("通过" if ok else "可疑")
        rows.append((seg["segment_id"][-3:], (seg.get("text_gen_final") or "")[:34],
                     bool(seg.get("merged_from")), verdict))
    print("  实际          成片 %d 段（其中合并段 %d 个），出口码 %d"
          % (len(bj.get("segments", [])), sum(1 for r in rows if r[2]), code))
    print()
    print("  段   合并  预测能判                        实际判定")
    for sid, prompt, is_merged, verdict in rows:
        fields = [f for pat, f, _th, _w, _k in signature.RULES if _re.search(pat, prompt.lower())]
        print("  %-4s %-4s  %-30s %s" % (sid, "是" if is_merged else "-",
              "/".join(fields) if fields else "无法判（无校准量）", verdict))
    print()
    print("  对照          预测合并 %d 处，实际合并段 %d 个"
          % (len(pairs), sum(1 for r in rows if r[2])))
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence", nargs="?", default=None)
    ap.add_argument("--k", type=int, default=6, help="每段候选数")
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--check", action="store_true", help="冒烟测试三句")
    ap.add_argument("--batch", default=None, help="批量文件：每行一句")
    ap.add_argument("--explain", action="store_true",
                    help="只做规划，说明这句话能做到什么程度")
    ap.add_argument("--compare", action="store_true",
                    help="跑完整流程，并把预测与实际结果并排输出")
    a = ap.parse_args()

    if a.compare:
        if not a.sentence:
            ap.print_help()
            sys.exit(2)
        sys.exit(compare(a.sentence, k=a.k, job_id=a.job_id))

    if a.explain:
        if not a.sentence:
            ap.print_help()
            sys.exit(2)
        sys.exit(explain(a.sentence, a.job_id))

    if a.check:
        worst = 0
        for i, s in enumerate(CHECK_SENTENCES, 1):
            print("  == 冒烟 %d/%d ==" % (i, len(CHECK_SENTENCES)))
            code, _ = run_one(s, job_id="smoke_%d" % i, k=a.k)
            worst = max(worst, code)
        print()
        print("  冒烟结果：%s" % ("全部通过" if worst == 0 else "有问题，退出码 %d" % worst))
        sys.exit(worst)

    if a.batch:
        f = Path(a.batch)
        if not f.exists():
            print("  找不到批量文件：%s" % f)
            sys.exit(2)
        lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        print("  批量 %d 句" % len(lines))
        rows, worst = [], 0
        for i, s_text in enumerate(lines, 1):
            code, info = run_one(s_text, job_id="batch_%02d" % i, k=a.k)
            worst = max(worst, code)
            bj_p = Path(info.get("dir", "")) / "assembled" / "bounds.json"
            gaps = []
            if bj_p.exists():
                import json as _json
                gaps = [j["pose_gap_deg"] for j in _json.loads(
                    bj_p.read_text(encoding="utf-8")).get("junctions", [])]
            rows.append((i, s_text, code, info.get("dropped") or [], gaps))
        print()
        print("  序号 出口 去掉段 交界              句子")
        for i, s_text, code, dropped, gaps in rows:
            print("  %-4d %-4d %-6s %-18s %s" % (i, code, "、".join(dropped) or "-",
                  " ".join("%.0f" % g for g in gaps) or "-", s_text[:38]))
        print()
        print("  批量结果：%d 句，最差出口码 %d（0 正常 / 1 有问题 / 2 没出片）" % (len(rows), worst))
        sys.exit(worst)

    if not a.sentence:
        ap.print_help()
        sys.exit(2)
    code, _ = run_one(a.sentence, job_id=a.job_id, k=a.k)
    sys.exit(code)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
