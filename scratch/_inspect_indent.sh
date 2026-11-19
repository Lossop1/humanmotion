#!/bin/bash
cd /home/chuan/motion-work
echo "=== A12 相关行的原始缩进 ==="
grep -n "_good = \[c for c in ok" hm/run_job.py | cat -A | head -n 2
grep -n "def key(c):" -A 6 hm/run_job.py | cat -A | head -n 10
echo
echo "=== finish_job 无候选分支 ==="
grep -n "segs = segment_candidates" -A 4 hm/finish_job.py | cat -A | head -n 8
