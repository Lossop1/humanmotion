#!/bin/bash
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
echo "  ffmpeg: $(which ffmpeg || echo 缺)"
ls -l outputs/job_shape/render/final.mp4 outputs/job_shape/render/final_captioned.mp4 2>/dev/null \
  | while read -r perm links owner group size rest; do echo "  $size 字节  $rest"; done
echo "  流信息："
for f in outputs/job_shape/render/final.mp4 outputs/job_shape/render/final_captioned.mp4; do
  echo -n "    $f  "
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,bit_rate -of default=nw=1:nk=1 "$f" 2>/dev/null | tr '\n' ' '
  echo
done
