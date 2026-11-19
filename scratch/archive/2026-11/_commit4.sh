#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
fit: estimate betas from the target bone lengths before the analytic IK,
which cuts joint error from 2.7-2.9cm to 1.6-1.8cm and bone residual to 0.9cm
EOF
git add hm/shape.py hm/fit.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
