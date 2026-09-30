#!/usr/bin/env python
"""Generate notebooks/01_milestone1.ipynb from the CELLS list below (easier to review than JSON)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "notebooks" / "01_milestone1.ipynb"
CELLS: list[tuple[str, str]] = []  # (cell_type, source); filled in the sections below


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


md(r'''
# Mốc 1: Baseline image / text / multimodal trên UPMC Food-101

Notebook điều khiển: clone repo, chuẩn bị dữ liệu, chạy 9 run và phân tích kết quả.

**Trước khi chạy**
1. Runtime → Change runtime type → chọn GPU.
2. Colab Secrets (biểu tượng chìa khóa): thêm `KAGGLE_API_TOKEN` (token mới `KGAT_...`) hoặc cặp `KAGGLE_USERNAME` + `KAGGLE_KEY` (kaggle.com → Settings → API → Create New Token), bật quyền cho notebook.
3. Chạy lần lượt từ trên xuống. Mọi lệnh train có thể chạy lại sau khi mất session: run đã xong sẽ được bỏ qua, run dở dang tự resume.
''')

code(r'''
!nvidia-smi
''')

code(r'''
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
print("repo:", REPO_DIR, "| work dir:", WORK_DIR)
''')

code(r'''
!pip install -q -r requirements-colab.txt
''')

code(r'''
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


md(r'''
## 1. Tải và giải nén dữ liệu
Lần đầu tải từ Kaggle vào Drive. Các lần sau chỉ giải nén bản trên Drive về ổ local `/content` (đọc ảnh từ Drive rất chậm).
''')

code(r'''
import zipfile
from google.colab import userdata

CACHE_DIR = WORK_DIR / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = CACHE_DIR / "upmcfood101.zip"

if not ZIP_PATH.exists():
    from google.colab.userdata import SecretNotFoundError
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)  # new token style needs a recent client
    try:  # new-style token (KGAT_...) first, then the legacy username + key pair
        os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    except SecretNotFoundError:
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    subprocess.run(["kaggle", "datasets", "download", "-d", "gianmarco96/upmcfood101", "-p", str(CACHE_DIR)], check=True)

if not DATA_ROOT.exists():
    DATA_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_ROOT)
print("extracted to", DATA_ROOT)
''')

md(r'''
Kiểm tra cấu trúc thư mục và vài dòng đầu của các file CSV. Nếu `prepare_data.py` báo lỗi định dạng, xem kết quả cell này để chỉnh `src/foodmm/data/prepare.py`.
''')

code(r'''
from foodmm.data.prepare import describe_tree

print(describe_tree(DATA_ROOT, max_depth=3, max_entries=60))
for csv_path in sorted(DATA_ROOT.rglob("*.csv"))[:5]:
    print("\n==", csv_path.relative_to(DATA_ROOT))
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        for _ in range(3):
            print(f.readline().rstrip()[:300])
''')

md(r'''
## 2. Chạy thử nhanh (5 lớp, 1 epoch)
Kiểm tra toàn bộ pipeline trên dữ liệu nhỏ, ghi vào `/content/smoke` (không đụng tới Drive). Xem cột `seconds` trong log để ước lượng thời gian một epoch thật.
''')

code(r'''
SMOKE = ["--set", f"paths.data_root={DATA_ROOT}", "paths.work_dir=/content/smoke",
         "data.subset_classes=5", "data.max_per_class=60",
         "train.image.epochs=1", "train.text.epochs=1", "train.multimodal.epochs=1"]

run("prepare_data.py", "--force", *SMOKE)
run("train_tfidf.py", "--force", *SMOKE, "data.text_mask=strict")
run("train.py", "--modality", "image", "--force", *SMOKE)
run("train.py", "--modality", "text", "--force", *SMOKE, "data.text_mask=strict")
run("late_fusion.py", "--text_run", "text_strict", "--force", *SMOKE)
run("train.py", "--modality", "multimodal", "--force", *SMOKE, "data.text_mask=strict")
run("summarize.py", *SMOKE)
''')


md(r'''
## 3. Chuẩn bị dữ liệu đầy đủ và EDA
''')

code(r'''
FULL = ["--set", f"paths.data_root={DATA_ROOT}", f"paths.work_dir={WORK_DIR}"]
run("prepare_data.py", *FULL)
''')

code(r'''
import json
import matplotlib.pyplot as plt
import pandas as pd
from foodmm.data.prepare import load_manifest

manifest = load_manifest(WORK_DIR / "data" / "manifest.csv")
classes = json.loads((WORK_DIR / "data" / "classes.json").read_text())
print(json.loads((WORK_DIR / "data" / "stats.json").read_text()))
display(manifest["split"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
manifest[manifest["split"] == "train"]["label"].value_counts().plot(ax=axes[0], title="Số ảnh train mỗi lớp", xticks=[])
manifest["text"].str.split().str.len().clip(upper=500).hist(ax=axes[1], bins=50)
axes[1].set_title("Số từ mỗi text (cắt ở 500)")
plt.show()
''')

code(r'''
from foodmm.analysis import plot_examples

sample = manifest[manifest["split"] == "train"].sample(8, random_state=0)
plot_examples(sample, DATA_ROOT, text_col="text", title_cols=["label"])
plt.show()
''')

md(r'''
**Leakage:** text còn chứa tên món đến mức nào sau mỗi chế độ che (tính trên 20.000 mẫu ngẫu nhiên cho nhanh).
`own_word_rate` > 0 ở chế độ `exact` là phần tên món mà `exact` bỏ sót (ví dụ "carbonara").
''')

code(r'''
from foodmm.analysis import leakage_table

leak = leakage_table(manifest.sample(min(20000, len(manifest)), random_state=0), classes)
display(leak.style.format({c: "{:.1%}" for c in ["own_label_rate", "own_word_rate", "other_label_rate"]}))
''')


md(r'''
## 4. Chạy 9 run
Mỗi lệnh bỏ qua run đã xong và tự resume nếu bị ngắt. Thứ tự quan trọng: late và early fusion cần run `image` và `text_<mask>` tương ứng đã xong.
''')

code(r'''
for mask in ["none", "strict"]:
    run("train_tfidf.py", *FULL, f"data.text_mask={mask}")
''')

code(r'''
run("train.py", "--modality", "image", *FULL)
''')

code(r'''
for mask in ["none", "strict"]:
    run("train.py", "--modality", "text", *FULL, f"data.text_mask={mask}")
''')

code(r'''
for mask in ["none", "strict"]:
    run("late_fusion.py", "--text_run", f"text_{mask}", *FULL)
''')

code(r'''
for mask in ["none", "strict"]:
    run("train.py", "--modality", "multimodal", *FULL, f"data.text_mask={mask}")
''')

md(r'''
## 5. Kết quả
''')

code(r'''
from foodmm.analysis import collect_results, plot_results_bar

run("summarize.py", *FULL)
results = collect_results(WORK_DIR / "runs")
display(results)
fig = plot_results_bar(results)
fig.savefig(WORK_DIR / "results" / "results_bar.png", dpi=150, bbox_inches="tight")
plt.show()
''')

code(r'''
from foodmm.analysis import plot_weight_curve

curves = {m: json.loads((WORK_DIR / "runs" / f"late_{m}" / "weight_curve.json").read_text()) for m in ["none", "strict"]}
plot_weight_curve(curves)
plt.show()
''')


md(r'''
## 6. Phân tích lỗi: image-only so với early fusion
Đổi `MASK` thành `"none"` để xem với text gốc.
''')

code(r'''
from foodmm.analysis import plot_confusion
from foodmm.late_fusion import load_preds
from foodmm.metrics import confusion, top_confused_pairs

RUNS = WORK_DIR / "runs"
MASK = "strict"
img = load_preds(RUNS / "image", "test")
early = load_preds(RUNS / f"early_{MASK}", "test")
for name, p in [("image", img), (f"early_{MASK}", early)]:
    preds = p["logits"].argmax(axis=1)
    plot_confusion(confusion(preds, p["labels"], len(classes)), title=name)
    plt.show()
    display(top_confused_pairs(preds, p["labels"], classes, k=10))
''')

code(r'''
from foodmm.analysis import per_class_gain

gain = per_class_gain(img, early, classes)
print("Lớp được lợi nhiều nhất khi thêm text:")
display(gain.head(10))
print("Lớp bị thiệt nhiều nhất:")
display(gain.tail(10))
''')

code(r'''
from foodmm.analysis import compare_predictions
from foodmm.data.text_utils import text_column

col = text_column(MASK)
cmp = compare_predictions(img, early).merge(
    manifest[["id", "image_path", col]].rename(columns={col: "shown_text"}), on="id")
cmp["true"] = cmp["label"].map(lambda i: classes[i])
cmp["image_pred"] = cmp["pred_a"].map(lambda i: classes[i])
cmp["early_pred"] = cmp["pred_b"].map(lambda i: classes[i])
fixed = cmp[~cmp["correct_a"] & cmp["correct_b"]]
broken = cmp[cmp["correct_a"] & ~cmp["correct_b"]]
print(f"early đúng / image sai: {len(fixed)} | image đúng / early sai: {len(broken)}")
for title, rows in [("Fusion sửa được lỗi của image-only", fixed), ("Fusion làm hỏng dự đoán đúng", broken)]:
    if len(rows):
        print(title)
        plot_examples(rows.sample(frac=1.0, random_state=0).head(8), DATA_ROOT, text_col="shown_text",
                      title_cols=["true", "image_pred", "early_pred"])
        plt.show()
''')


def build() -> dict:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
