"""Shared builder and setup cells for notebooks 02-04 (notebook 01 keeps tools/build_notebook.py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


class NotebookBuilder:
    def __init__(self) -> None:
        self.cells: list[tuple[str, str]] = []

    def md(self, src: str) -> None:
        self.cells.append(("markdown", src.strip("\n")))

    def code(self, src: str) -> None:
        self.cells.append(("code", src.strip("\n")))

    def build(self) -> dict:
        cells = []
        for i, (kind, src) in enumerate(self.cells):
            cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
            if kind == "code":
                cell.update({"execution_count": None, "outputs": []})
            cells.append(cell)
        return {"cells": cells,
                "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def main(self, default_out: str | Path, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description="Generate a Colab notebook")
        parser.add_argument("--out", default=str(default_out))
        args = parser.parse_args(argv)
        out = self.write(args.out)
        print(f"Wrote {out} ({len(self.cells)} cells)")
        return 0


def add_setup(nb: NotebookBuilder) -> None:
    nb.code("!nvidia-smi")
    nb.code(r'''
import os, subprocess, sys
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

REPO_URL = "https://github.com/viethoang2503/food-recognize-extract.git"
REPO_DIR = Path("/content/food-recognize-extract")
WORK_DIR = Path("/content/drive/MyDrive/foodmm")
DATA_ROOT = Path("/content/data/upmc_food101")

if REPO_DIR.exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
BASE = [f"paths.data_root={DATA_ROOT}", f"paths.work_dir={WORK_DIR}"]
print("repo:", REPO_DIR, "| work dir:", WORK_DIR)
''')
    nb.code("!pip install -q -r requirements-colab.txt")
    nb.code(r'''
def run(script, *args):
    """Run a repo script and stream its output into the notebook."""
    cmd = [sys.executable, "-u", f"scripts/{script}", *map(str, args)]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"{script} failed with exit code {proc.returncode}")
''')


def add_data(nb: NotebookBuilder) -> None:
    nb.md(r'''
## Dữ liệu
Giải nén bản zip trên Drive về `/content` (lần đầu tải từ Kaggle, cần Colab Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`). `prepare_data.py` bỏ qua nếu manifest đã có từ Mốc 1.
''')
    nb.code(r'''
import zipfile
from google.colab import userdata

ZIP_PATH = WORK_DIR / "data" / "upmcfood101.zip"
ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
if not ZIP_PATH.exists():
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    subprocess.run(["kaggle", "datasets", "download", "-d", "gianmarco96/upmcfood101", "-p", str(ZIP_PATH.parent)], check=True)
if not DATA_ROOT.exists():
    DATA_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_ROOT)
run("prepare_data.py", "--set", *BASE)
''')
