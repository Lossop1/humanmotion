#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
criteria: make thresholds a property of the backend, not of the pipeline.
Physical invariants stay global; quality thresholds are calibrated per backend from a
fixed probe set (momask 29.5 deg/frame vs mdm 48.5 on the same probe), and absolute
physical quantities are used only for ranking. Backends now declare a contract
(hml263, 20fps, metre, 22 joints) and are rejected if it does not hold.
EOF
git add hm/backends.py hm/gates.py hm/assemble_params.py hm/criteria_backend.py hm/calibrate_backend.py outputs/_reference/criteria_momask.json outputs/_reference/criteria_mdm.json
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
