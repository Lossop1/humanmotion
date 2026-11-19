"""给能力表补「提示词范围」字段。

发现的矛盾
    能力表 sit 条目记的是 "a person sits down and stands back up"（坐下再站起），测试 10/10 通过；
    但实际用的单动作措辞 "a person sits down" 五种写法全部几何不过（0/20）。
    也就是说表里的「支持」只对那一句复合提示词成立，不能外推到同名的单动作。

做法
    给每个条目加 prompt_scope：
        simple     单动作措辞，可直接用于分镜段
        composite  复合动作（含 and），只对原句成立，路由时按 unverified 处理
    路由与提示词复用都读这个字段，避免把复合提示词塞进单段。
"""
import json
from pathlib import Path

P = Path("/home/chuan/motion-work/outputs/_reference/capability_v4.json")
doc = json.loads(P.read_text(encoding="utf-8"))

changed = 0
for key, v in doc["table"].items():
    prompt = (v.get("prompt") or "").lower()
    scope = "composite" if " and " in (" %s " % prompt) or "," in prompt else "simple"
    if v.get("prompt_scope") != scope:
        v["prompt_scope"] = scope
        changed += 1

doc.setdefault("notes", []).append(
    "prompt_scope 由 capability 生成时补：composite 的条目只对原句成立，路由按 unverified 处理")
P.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

comp = [k for k, v in doc["table"].items() if v.get("prompt_scope") == "composite"]
print("  已给 %d 个条目写 prompt_scope，其中复合动作 %d 个：%s"
      % (changed, len(comp), ", ".join(comp[:10])))
