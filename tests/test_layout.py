"""包结构守门：不执行任何模块，只做静态检查。

整理代码（移动模块）时最容易出的两类错：导入路径失效、核心代码反向依赖旧代码。
很多模块没有 main 保护、导入即运行，所以这里用 AST，不用 import。
"""
import ast
import py_compile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HM = REPO / "hm"
PY_FILES = sorted(HM.rglob("*.py")) + sorted(REPO.glob("*.py")) + sorted((REPO / "tests").glob("*.py"))


def _module_exists(dotted):
    p = REPO.joinpath(*dotted.split("."))
    return p.with_suffix(".py").exists() or (p / "__init__.py").exists()


def _hm_imports(path):
    """返回 [(module, name_or_None)]，只含 hm 内部导入。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and (
                n.module == "hm" or n.module.startswith("hm.")):
            out.extend((n.module, a.name) for a in n.names)
        elif isinstance(n, ast.Import):
            out.extend((a.name, None) for a in n.names if a.name.startswith("hm."))
    return out


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.relative_to(REPO).as_posix())
def test_compiles_and_hm_imports_resolve(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)
    for module, name in _hm_imports(path):
        assert _module_exists(module), "%s 导入了不存在的模块 %s" % (path.name, module)
        if name and name != "*":
            pkg = REPO.joinpath(*module.split("."))
            if (pkg / "__init__.py").exists():
                init = (pkg / "__init__.py").read_text(encoding="utf-8")
                assert _module_exists(module + "." + name) or name in init, \
                    "%s: %s 里没有 %s" % (path.name, module, name)


def test_core_does_not_depend_on_legacy_or_experiments():
    """核心流程不许依赖 legacy / experiments；依赖方向只能反过来。"""
    bad = []
    for p in HM.glob("*.py"):
        for module, _ in _hm_imports(p):
            if module.startswith(("hm.legacy", "hm.experiments")):
                bad.append("%s -> %s" % (p.name, module))
    assert not bad, bad


def test_blender_scripts_stay_where_callers_expect():
    """render_blender / render_skeleton 由多处按路径调用，移动它们要同时改调用方。"""
    assert (HM / "render_blender.py").exists()
    assert (HM / "render_skeleton.py").exists()


def test_legacy_callers_locate_render_scripts():
    """legacy 里按 parents[1] 找渲染脚本，目录层级变了这里要报错。"""
    for p in (HM / "legacy").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        for script in ("render_blender.py", "render_skeleton.py"):
            if script in text:
                assert 'parents[1] / "%s"' % script in text, "%s 没有用 parents[1] 定位 %s" % (p.name, script)
