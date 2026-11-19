#!/bin/bash
set -e
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
planner: emit verbatim source fragments plus direction/object/relation/speed;
A02 uses them for source_text and reports coverage after stripping fillers and punctuation
EOF
git add hm/planner.py hm/run_job.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
