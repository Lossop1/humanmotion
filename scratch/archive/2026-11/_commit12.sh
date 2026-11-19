#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
prompts: add a measured prompt bank. Eight actions times three phrasings, scored with the
geometry criteria: "punches forward" 4/4 vs "throws a punch" 0/4, "kicks forward" 3/4 vs
"does a front kick" 1/4. A06 now substitutes the bank's best phrasing when the direction
agrees and the tested rate is at least 0.75; the direction guard stops a forward walk from
being swapped for a backward one. Sitting stays at 0/12 across phrasings: a genuine limit.
EOF
git add hm/run_job.py hm/prompt_bank.py outputs/_reference/prompt_bank.json
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
