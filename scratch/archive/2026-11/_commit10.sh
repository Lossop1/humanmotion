#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
turn: recalibrate on the user-reported pair. The MDM turn had a 44 deg/frame spike at
the incoming junction (13.6x the previous segment) and leaned 8.9 deg while moving 0.97m
backward; the MoMask turn has neither. Routing now picks MoMask for turn (10/10 vs 8/10
under current criteria), and both symptoms become measured checks: junction_jerk() plus
lean/backward terms in the action signature.
EOF
git add hm/gates.py hm/signature.py hm/assemble_params.py outputs/_reference/capability_v4.json
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
