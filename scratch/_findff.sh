#!/bin/bash
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
echo "  python 里找自带 ffmpeg："
python - <<'PYEOF'
import shutil
print("    conda bin:", shutil.which("ffmpeg") or "无")
for mod in ("imageio_ffmpeg", "moviepy", "av", "ffmpeg"):
    try:
        m = __import__(mod)
        extra = ""
        if mod == "imageio_ffmpeg":
            extra = m.get_ffmpeg_exe()
        print("    %-14s 可用 %s" % (mod, extra))
    except Exception as e:
        print("    %-14s 不可用" % mod)
PYEOF
echo "  系统里找："
ls /usr/bin/ffmpeg* /opt/*/bin/ffmpeg 2>/dev/null || echo "    没有"
