#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
parameters: separate physical invariants (hard gates), calibrated criteria (only in the
scope they were calibrated on) and uncalibrated observations (record and rank, never
gate). The action signature becomes a ranking term; a segment with no passing candidate
is dropped by default instead of degrading to a broken clip.
EOF
git add hm/run_job.py hm/finish_job.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
