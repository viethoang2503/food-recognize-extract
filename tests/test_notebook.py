import json
import subprocess
import sys

from helpers import REPO_ROOT

NB = REPO_ROOT / "notebooks" / "01_milestone1.ipynb"


def _code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_generated_and_up_to_date(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "build_notebook.py"), "--out", str(out)], check=True)
    assert out.read_text(encoding="utf-8") == NB.read_text(encoding="utf-8")


def test_notebook_cells_compile_and_imports_resolve():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) > 20
    for src in _code_cells(nb):
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        compile("\n".join(lines), "<cell>", "exec")
        for ln in lines:
            if ln.startswith("from foodmm") or ln.startswith("import foodmm"):
                exec(ln, {})
