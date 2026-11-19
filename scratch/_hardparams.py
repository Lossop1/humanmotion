"""把「硬参数」的角色分清楚，改两处代码。

一、动作签名从「筛掉」改为「排序优先」
    位移 0.6 米、手过肩 5 厘米、下蹲 18 厘米这些数只是观察来的量，
    没有按动作、体型、后端校准过，不能当门禁。
    它只在候选之间比较时起作用：都合格的话优先挑「看起来做出来了」的那个；
    全都不合格也不拦，记录数字让人看。

二、全段失败时的默认行为从「降级用最差的候选」改为「丢掉这一段」
    实测：坐下六段候选几何全不过，降级用的那条反向帧占 51%，
    画面是坏的。产品上宁可少一段，也不放一段坏动作，并在日志与产物里说清楚。
    可用 HM_DEGRADE=clip 回到老行为，或 abort 直接中止。
"""
import ast
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------- 一、签名改为排序 ----------
p = r"D:\my-projects\humanmotion\repo\hm\run_job.py"
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")

old = '''              _good = [c for c in ok if _sig_pass(c) is True]
              if _good and len(_good) < len(ok):
                  job.log("A12 %s 动作签名：%d/%d 个候选确实做出了该动作，优先从中挑"
                          % (seg["segment_id"], len(_good), len(ok)))
              if _good:
                  ok = _good'''
new = '''              _sig_of = {c["candidate_id"]: _sig_pass(c) for c in ok}
              _n_good = sum(1 for v in _sig_of.values() if v is True)
              if _n_good < len(ok):
                  job.log("A12 %s 动作签名（只作排序参考）：%d/%d 个候选看起来做出了该动作"
                          % (seg["segment_id"], _n_good, len(ok)))'''
if old in s:
    s = s.replace(old, new, 1)
else:
    raise SystemExit("未匹配签名筛选段")

old2 = '''          # 软排序：骨长偏差优先，其次单帧姿态变化，再其次路径长度
          def key(c):
              m = c["metrics"]
              return (m.get("bone_length_jitter_m") or 9e9,
                      m.get("pose_rate_max") or 9e9,
                      m.get("root_path_length") or 9e9)'''
new2 = '''          # 软排序：先看动作签名（只是偏好，不是门禁），再看骨长偏差、
          # 单帧姿态变化、路径长度。全部是相对比较，没有绝对阈值拦人。
          def key(c):
              m = c["metrics"]
              _ok = locals().get("_sig_of", {}).get(c["candidate_id"])
              return (0 if _ok is True else 1,
                      m.get("bone_length_jitter_m") or 9e9,
                      m.get("pose_rate_max") or 9e9,
                      m.get("root_path_length") or 9e9)'''
assert old2 in s, "未匹配排序键"
s = s.replace(old2, new2, 1)
s = s.replace('job.log("A12 %s 选中 %s（排序依据：骨长偏差、单帧姿态变化、路径长度）"',
              'job.log("A12 %s 选中 %s（排序依据：动作签名、骨长偏差、单帧姿态变化、路径长度）"')
open(p, "w", encoding="utf-8", newline="\n").write(s)
ast.parse(s)
print("  签名改为排序项（不再是筛选）")

# ---------- 二、全段失败默认丢弃 ----------
q = r"D:\my-projects\humanmotion\repo\hm\finish_job.py"
t = open(q, encoding="utf-8", newline="").read().replace("\r\n", "\n")

if "HM_DEGRADE" in t:
    print("  降级策略已存在")
else:
    old3 = '''    segs = segment_candidates(job, sb)
    if any(not c for _, c in segs):'''
    new3 = '''    segs = segment_candidates(job, sb)
    import os as _os
    policy = _os.environ.get("HM_DEGRADE", "drop")
    if policy == "abort" and any(not c for _, c in segs):
        bad = [s["segment_id"] for s, c in segs if not c]
        raise SystemExit("这些段没有任何通过判据的候选：%s（HM_DEGRADE=abort）" % bad)
    if policy == "drop" and any(not c for _, c in segs):
        bad = [s["segment_id"] for s, c in segs if not c]
        job.log("取舍：%s 没有通过判据的候选，从成片里去掉（HM_DEGRADE=drop）" % "、".join(bad))
        segs = [(s, c) for s, c in segs if c]
    if any(not c for _, c in segs):'''
    assert old3 in t, "未匹配无候选分支入口"
    t = t.replace(old3, new3, 1)
    open(q, "w", encoding="utf-8", newline="\n").write(t)
    ast.parse(t)
    print("  全段失败默认丢弃（HM_DEGRADE=drop|clip|abort）")
