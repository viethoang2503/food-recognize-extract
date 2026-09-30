import json
import subprocess
import sys

import pytest

from helpers import REPO_ROOT

NOTEBOOKS = [("build_notebook_m2.py", "02_milestone2.ipynb"), ("build_notebook_vlm.py", "03_vlm.ipynb"),
             ("build_notebook_demo.py", "04_demo.ipynb")]


@pytest.mark.parametrize("builder,name", NOTEBOOKS)
def test_notebook_is_generated_and_up_to_date(tmp_path, builder, name):
    out = tmp_path / name
    subprocess.run([sys.executable, str(REPO_ROOT / "tools" / builder), "--out", str(out)], check=True)
    assert out.read_text(encoding="utf-8") == (REPO_ROOT / "notebooks" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("builder,name", NOTEBOOKS)
def test_notebook_cells_compile_and_imports_resolve(builder, name):
    nb = json.loads((REPO_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) > 8
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = [ln for ln in "".join(cell["source"]).splitlines() if not ln.lstrip().startswith(("!", "%"))]
        compile("\n".join(lines), "<cell>", "exec")
        for ln in lines:
            if ln.startswith(("from foodmm", "import foodmm")):
                exec(ln, {})
