"""测试公共设置：保证测试不碰真实缓存、不读真实密钥、不联网。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    from hm import cache
    monkeypatch.setattr(cache, "ROOT", tmp_path / "cache")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("HM_PLAN_FALLBACK", raising=False)
    monkeypatch.delenv("HM_NO_CACHE", raising=False)
