#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
cache: content-addressed plan/generation/fit/render caches. Same sentence reruns in 5s
instead of 120s, and the planner is pinned to temperature 0 so downstream stages see
identical prompts. Stage timing now separates what we own (k, routing, algorithms,
scheduling, render settings) from what hardware owns (raw model inference speed).
EOF
git add hm/cache.py hm/planner.py hm/run_job.py hm/assemble_params.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
