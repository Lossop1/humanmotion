"""按行号打补丁：签名改为排序项，全段失败默认丢弃。

上一次靠空白字符匹配失败，这次只认行号与关键字，避免缩进猜错。
"""
import ast
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------- run_job.py ----------
p = r"D:\my-projects\humanmotion\repo\hm\run_job.py"
lines = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n").split("\n")

i0 = next(k for k, l in enumerate(lines) if "_good = [c for c in ok if _sig_pass(c) is True]" in l)
i1 = next(k for k, l in enumerate(lines) if k > i0 and l.strip() == "ok = _good")
ind = lines[i0][:len(lines[i0]) - len(lines[i0].lstrip())]
new_block = [
    ind + "_sig_of = {c['candidate_id']: _sig_pass(c) for c in ok}",
    ind + "_n_good = sum(1 for v in _sig_of.values() if v is True)",
    ind + "if _n_good < len(ok):",
    ind + "    job.log('A12 %s 动作签名（只作排序参考）：%d/%d 个候选看起来做出了该动作'",
    ind + "            % (seg['segment_id'], _n_good, len(ok)))",
]
lines[i0:i1 + 1] = new_block

# 排序键：先看签名，再看原来的三项
j0 = next(k for k, l in enumerate(lines) if l.strip() == "return (m.get(\"bone_length_jitter_m\") or 9e9,")
ind_j = lines[j0][:len(lines[j0]) - len(lines[j0].lstrip())]
lines[j0:j0 + 3] = [
    ind_j + "_ok = _sig_of.get(c[\"candidate_id\"]) if \"_sig_of\" in dir() else None",
    ind_j + "return (0 if _ok is True else 1,",
    ind_j + "        m.get(\"bone_length_jitter_m\") or 9e9,",
    ind_j + "        m.get(\"pose_rate_max\") or 9e9,",
    ind_j + "        m.get(\"root_path_length\") or 9e9)",
]
s = "\n".join(lines)
s = s.replace("排序依据：骨长偏差、单帧姿态变化、路径长度",
              "排序依据：动作签名、骨长偏差、单帧姿态变化、路径长度")
open(p, "w", encoding="utf-8", newline="\n").write(s)
ast.parse(s)
print("  run_job：签名改为排序项")

# ---------- finish_job.py ----------
q = r"D:\my-projects\humanmotion\repo\hm\finish_job.py"
ls = open(q, encoding="utf-8", newline="").read().replace("\r\n", "\n").split("\n")
k0 = next(k for k, l in enumerate(ls) if l.strip() == "segs = segment_candidates(job, sb)")
ind2 = ls[k0][:len(ls[k0]) - len(ls[k0].lstrip())]
if "HM_DEGRADE" not in "\n".join(ls):
    block = [
        ind2 + "import os as _os",
        ind2 + "policy = _os.environ.get(\"HM_DEGRADE\", \"drop\")   # drop|clip|abort",
        ind2 + "if any(not c for _, c in segs):",
        ind2 + "    bad = [s[\"segment_id\"] for s, c in segs if not c]",
        ind2 + "    if policy == \"abort\":",
        ind2 + "        raise SystemExit(\"这些段没有通过判据的候选：%s\" % bad)",
        ind2 + "    if policy == \"drop\":",
        ind2 + "        job.log(\"取舍：%s 没有通过判据的候选，从成片里去掉（HM_DEGRADE=drop）\"",
        ind2 + "                % \"、\".join(bad))",
        ind2 + "        segs = [(s, c) for s, c in segs if c]",
    ]
    ls[k0 + 1:k0 + 1] = block
    t = "\n".join(ls)
    open(q, "w", encoding="utf-8", newline="\n").write(t)
    ast.parse(t)
    print("  finish_job：默认丢弃失败段（HM_DEGRADE=drop|clip|abort）")
else:
    print("  finish_job 已有策略")
