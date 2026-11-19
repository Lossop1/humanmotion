"""把动作签名挪到「选择候选」这一步，不动 A08 的记录结构。

之前尝试在 A08 里写入字段，改到一半容易把记录结构弄坏。
选择阶段本来就要逐个读候选的关节文件，在那里算签名最省事，也不改数据格式。
"""
import ast
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
p = r"D:\my-projects\humanmotion\repo\hm\run_job.py"
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")

# 1) 撤掉 A08 里无效的插入
s = s.replace('''            from hm import signature as _sig
            _s = _sig.signature(J)
            _sok, _sbad, _sdet = _sig.check(seg.get("text_gen") or "", _s)
''', "")

# 2) 日志恢复成原来的形状
s = re.sub(r'job\.log\("A08-A10 %s 检查完成：.*?\)\)\n',
           'job.log("A08-A10 %s 检查完成：%d 个候选，%d 个通过硬检查"\n'
           '                % (seg["segment_id"], len(results), sum(1 for r in results if r["decision"] == "pass")))\n',
           s, count=1, flags=re.S)

# 3) 选择阶段加签名优先
old = '''        ok = [c for c in cands if c["decision"] == "pass"]
        _sig_ok = [c for c in ok if c.get("signature_ok") is True]
        if _sig_ok:
            ok = _sig_ok'''
new = '''        ok = [c for c in cands if c["decision"] == "pass"]
        # 动作签名：几何通过的候选里，优先挑「确实做出来这个动作」的
        try:
            from hm import signature as _sig
            _prompt = seg.get("text_gen_final") or seg.get("text_gen") or ""

            def _sig_pass(c):
                n = int(c["candidate_id"].split("_")[-1])
                J = np.load(job.dir / "segments" / seg["segment_id"] /
                            ("candidate_%d" % n) / "joints22.npy")
                verdict, bad, det = _sig.check(_prompt, _sig.signature(J))
                return verdict

            _good = [c for c in ok if _sig_pass(c) is True]
            if _good and len(_good) < len(ok):
                job.log("A12 %s 动作签名：%d/%d 个候选确实做出了该动作，优先从中挑"
                        % (seg["segment_id"], len(_good), len(ok)))
            if _good:
                ok = _good
        except Exception as _e:
            job.log("A12 %s 动作签名跳过：%s" % (seg["segment_id"], str(_e)[:80]))'''
if old in s:
    s = s.replace(old, new, 1)
elif "_sig_pass(c)" not in s:
    raise SystemExit("未匹配选择入口，当前片段：\n" + s[s.find("ok = [c for c in cands"):][:400])

open(p, "w", encoding="utf-8", newline="\n").write(s)
ast.parse(s)
print("  签名改为选择阶段判定，A08 恢复原状")
