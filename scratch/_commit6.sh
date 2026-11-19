#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
backends: add a registry so capability knowledge travels with each backend.
A05 now routes per segment from the capability table (turn goes to mdm at 0.80 vs
0.50 on momask), rejects unregistered backends instead of silently falling back,
and only pays a slow backend when its pass rate is at least 0.15 higher.
EOF
git add hm/backends.py hm/run_job.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
