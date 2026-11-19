#!/bin/bash
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
python - <<'PYEOF'
from hm import config
for k, v in config.MODEL_PATHS.items():
    print("  %-12s %s" % (k, v))
PYEOF
echo
echo "  有哪些 pipeline 目录："
ls motius/motius/pipelines/ | tr '\n' ' '
echo
echo
echo "  mdm 相关 pipeline 的 infer 签名："
for f in motius/motius/pipelines/mdm/pipeline.py motius/motius/pipelines/motionlcm/pipeline.py; do
  [ -f "$f" ] && { echo "  --- $f"; grep -n "def " "$f" | head -n 10; }
done
