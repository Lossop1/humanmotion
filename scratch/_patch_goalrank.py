"""选择时按动作偏好排序：拳类看前伸与腕速，通用看站距。

不做绝对门禁，只在同一批候选里做相对比较（min-max 归一），
因为「多前伸才算一拳」取决于动作、模型、体型，定死数字就是上一轮的错。
"""
import ast
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 1) 签名里补三个量
p = r"D:\my-projects\humanmotion\repo\hm\signature.py"
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "punch_reach" in s:
    print("  签名已有出拳量")
else:
    s = s.replace('''    return {
        "disp_m": float(horizon.max()),''',
'''    fwd = J[0, 10] - J[0, 7]
    fwd[1] = 0.0
    nf = float(np.linalg.norm(fwd))
    fwd = fwd / nf if nf > 1e-9 else np.array([0.0, 0.0, 1.0])
    rw = J[:, 21, :] - J[:, 16, :]          # 右手腕相对右肩
    reach = float((rw @ fwd).max())
    wspeed = float((np.linalg.norm(np.diff(J[:, 21, :], axis=0), axis=1) * 20).max()) if T > 1 else 0.0
    stance = float(np.mean(np.linalg.norm(J[:, 7, :] - J[:, 8, :], axis=1)))
    return {
        "punch_reach": reach,
        "wrist_speed": wspeed,
        "stance_width": stance,
        "disp_m": float(horizon.max()),''')
    open(p, "w", encoding="utf-8", newline="\n").write(s)
    ast.parse(s)
    print("  签名：加入 punch_reach / wrist_speed / stance_width")

# 2) 选择时按动作偏好做批内相对排序
q = r"D:\my-projects\humanmotion\repo\hm\run_job.py"
t = open(q, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "_goal_rank" in t:
    print("  选择已有动作偏好")
else:
    old = '''              _sig_of = {c["candidate_id"]: _sig_pass(c) for c in ok}'''
    new = '''              _sig_of = {c["candidate_id"]: _sig_pass(c) for c in ok}

              def _goal_rank(cands):
                  """按动作偏好做批内排序（0 最好）。只用相对比较，不用绝对阈值。"""
                  import re as _re
                  prompt = (seg.get("text_gen_final") or "").lower()
                  vals = {}
                  for _c in cands:
                      _n = int(_c["candidate_id"].split("_")[-1])
                      _J = np.load(job.dir / "segments" / seg["segment_id"] /
                                   ("candidate_%d" % _n) / "joints22.npy")
                      vals[_c["candidate_id"]] = _sig.signature(_J)
                  if _re.search(r"punch|kick|throw", prompt):
                      keyf = "punch_reach"
                  elif _re.search(r"jump|hop", prompt):
                      keyf = "pelvis_up"
                  else:
                      keyf = None
                  def _rank(c):
                      v = vals[c["candidate_id"]]
                      score = 0.0
                      if keyf:
                          arr = [vals[x["candidate_id"]][keyf] for x in cands]
                          lo, hi = min(arr), max(arr)
                          score -= 2.0 * ((v[keyf] - lo) / (hi - lo) if hi > lo else 0.0)
                      # 通用：站距越接近该批中位越好
                      st = [vals[x["candidate_id"]]["stance_width"] for x in cands]
                      mid = sorted(st)[len(st) // 2]
                      score += abs(v["stance_width"] - mid)
                      return score
                  return _rank, vals
              _gr, _sig_vals = _goal_rank(ok)'''
    assert old in t, "未匹配签名块"
    t = t.replace(old, new, 1)
    old2 = '''          def key(c):
              m = c["metrics"]
              _ok = _sig_of.get(c["candidate_id"]) if "_sig_of" in dir() else None'''
    new2 = '''          def key(c):
              m = c["metrics"]
              _ok = _sig_of.get(c["candidate_id"]) if "_sig_of" in dir() else None
              _g = _gr(c) if "_gr" in dir() else 0.0'''
    assert old2 in t, "未匹配排序键"
    t = t.replace(old2, new2, 1)
    old3 = '''              return (0 if _ok is True else 1,
                      m.get("bone_length_jitter_m") or 9e9,'''
    new3 = '''              return (0 if _ok is True else 1,
                      _g,
                      m.get("bone_length_jitter_m") or 9e9,'''
    assert old3 in t, "未匹配排序元组"
    t = t.replace(old3, new3, 1)
    open(q, "w", encoding="utf-8", newline="\n").write(t)
    ast.parse(t)
    print("  选择：加入动作偏好（拳类看前伸，通用看站距中位）")
