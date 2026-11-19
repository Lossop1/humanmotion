import glob
import json

for f in sorted(glob.glob("outputs/job_ik2/segments/seg_003/candidate_*/metrics.json"))[:6]:
    d = json.load(open(f, encoding="utf-8"))
    print("  %s  hard=%s  reasons=%s" % (
        f.split("/")[-2], d.get("hard_checks"), d.get("failure_reasons")))
