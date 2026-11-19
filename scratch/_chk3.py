import glob
import json
from pathlib import Path

ROOT = Path("/home/chuan/motion-work")
for sid in ("S5", "S7"):
    job = ROOT / "outputs" / ("eval_%s" % sid.lower())
    print("== %s ==" % sid)
    log = job / "run.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "没有可用候选" in line or "A13" in line:
                print("  " + line.strip()[-110:])
    for f in sorted(glob.glob(str(job / "segments" / "*" / "candidate_*" / "metrics.json")))[:3]:
        d = json.load(open(f, encoding="utf-8"))
        print("  %s/%s  %s" % (f.split("/")[-4], f.split("/")[-2],
                               (d.get("failure_reasons") or ["通过"])[0][:60]))
