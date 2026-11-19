#!/bin/bash
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
python - <<'PYEOF'
import os
import motius
print("  motius:", os.path.dirname(motius.__file__))
from motius.pipelines.auto import Pipeline
print("  infer_t2m 签名:", __import__("inspect").signature(Pipeline.infer_t2m))
print("  Pipeline 方法:", [m for m in dir(Pipeline) if not m.startswith("_")])
PYEOF
echo "  含 inpaint 关键字的文件："
grep -rln "inpaint\|inpaint\|prefix" deps/motius 2>/dev/null | head -n 8
