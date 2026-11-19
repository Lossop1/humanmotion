#!/bin/bash
set -e
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
reliability: A03 reads capability_v4 table instead of a hardcoded word list,
and a segment with no passing candidate now degrades to the least-bad one
instead of aborting the whole sentence
EOF
git add hm/run_job.py hm/finish_job.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
