#!/bin/bash
set -e
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
cat > /tmp/msg.txt <<'EOF'
delivery: render at 1080 with 16 samples, crop to the figure and downscale to 720,
then burn the storyboard caption with x264 crf16; A19 emits cropped, captioned and
storyboard artifacts for every job. Figure pixels go from 291 to 385 (and 1080 source).
EOF
git add hm/render_blender.py hm/assemble_params.py frame_crop.py frame_kit.py annotate_video.py job_report.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
