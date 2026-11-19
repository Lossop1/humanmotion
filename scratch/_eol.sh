#!/bin/bash
set -e
cd /home/chuan/motion-work
python - <<'PYEOF'
import pathlib
n = 0
for p in pathlib.Path("hm").glob("*.py"):
    b = p.read_bytes()
    if b"\r\n" in b:
        p.write_bytes(b.replace(b"\r\n", b"\n"))
        n += 1
print("  converted", n, "files to LF")
PYEOF
cat > /tmp/msg.txt <<'EOF'
normalise python line endings to LF on the compute host too
EOF
git add hm/*.py
git -c user.name=chuan -c user.email=chuan@local commit -q -F /tmp/msg.txt || echo "  nothing to commit"
git bundle create /home/chuan/motion-work.bundle --all >/dev/null 2>&1
git log --oneline | head -n 1
