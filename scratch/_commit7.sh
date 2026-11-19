#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
routing: drop the cost-based switching rule (speed is a hardware property, not ours to
route on). Capability table becomes optional: with a table we pick the best backend per
action, without one we just generate. Also default k 6 so the DP can pick closer poses.
EOF
git add hm/backends.py hm/run_full.py hm/run_job.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
