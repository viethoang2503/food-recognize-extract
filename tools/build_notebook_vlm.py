#!/usr/bin/env python
"""Generate notebooks/03_vlm.ipynb (Qwen3-VL structured extraction and evaluation)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_utils import NOTEBOOK_DIR, NotebookBuilder, add_data, add_setup  # noqa: E402

nb = NotebookBuilder()
nb.md(r'''
# Sub-project 3: Trích xuất thông tin món ăn bằng Qwen3-VL

Model chạy trực tiếp trong Colab qua `transformers`. Không cần server hay API key. Mỗi ảnh cho ra một JSON gồm `dish_name`, `cuisine`, `main_ingredients`, `cooking_method`, `confidence`.

**GPU:** L4 hoặc A100 chạy bf16. T4 tự chuyển sang 4-bit (`bitsandbytes`). Bảng so sánh với classifier dùng các run `image`, `text_strict`, `xattn_strict` của Mốc 2 (`vlm.compare_runs`); run nào chưa có sẽ bị bỏ qua.
''')
add_setup(nb)
add_data(nb)

nb.md(r'''
## 1. Chạy thử 5 ảnh
Nạp model (lần đầu tải khoảng 9 GB về `/root/.cache`) và kiểm tra output. Kết quả ghi vào `foodmm/vlm/smoke`, tách khỏi lần chạy thật.
''')
nb.code(r'''
import torch

print(torch.cuda.get_device_name(0), "| compute capability", torch.cuda.get_device_capability(0))
SMOKE = ["--set", *BASE, "vlm.tag=smoke", "vlm.n_samples=5"]
run("vlm_extract.py", "--mode", "image", *SMOKE)
print((WORK_DIR / "vlm" / "smoke" / "extract_image.jsonl").read_text()[:1500])
''')

nb.md(r'''
## 2. Chạy 200 ảnh với 3 chế độ prompt
`image` (chỉ ảnh), `text` (chỉ text đã che `strict`, không gửi ảnh), `image_text` (ảnh + text đã che): bộ so sánh image-only / text-only / multimodal của đề bài. Mất session thì chạy lại cell, các ảnh đã xong được bỏ qua. Có thể thêm `vlm.model_name=Qwen/Qwen3-VL-8B-Instruct vlm.tag=qwen3vl8b` để thử bản 8B.
''')
nb.code(r'''
for mode in ("image", "text", "image_text"):
    run("vlm_extract.py", "--mode", mode, "--set", *BASE)
''')

nb.md(r'''
## 3. Đánh giá
`dish_acc_lo`/`dish_acc_hi` là khoảng tin cậy 95% (bootstrap). Bảng thứ hai là accuracy của các classifier trên đúng 200 ảnh này, để so sánh trực tiếp. `ingredient_grounding` chỉ để tham khảo: các chế độ có `uses_text=True` được đọc chính text dùng để đo nên điểm bị thổi phồng.
''')
nb.code(r'''
from IPython.display import Markdown, display

run("vlm_evaluate.py", "--set", *BASE)
VLM_DIR = WORK_DIR / "vlm" / "qwen3vl4b"
display(Markdown((VLM_DIR / "eval_summary.md").read_text()))
display(Markdown("### Classifier trên cùng mẫu\n" + (VLM_DIR / "classifier_on_sample.md").read_text()))
''')

nb.md(r'''
## 4. Xem vài ví dụ
''')
nb.code(r'''
import json

import matplotlib.pyplot as plt
from PIL import Image as PILImage

from foodmm.data.prepare import load_manifest
from foodmm.vlm.evaluate import map_dish_to_class
from foodmm.vlm.extract import read_records
from foodmm.utils import load_json

manifest = load_manifest(WORK_DIR / "data" / "manifest.csv").set_index("id")
classes = load_json(WORK_DIR / "data" / "classes.json")
recs = {m: {r["id"]: r for r in read_records(VLM_DIR / f"extract_{m}.jsonl")}
        for m in ("image", "text", "image_text")}
for sid in load_json(VLM_DIR / "sample_ids.json")[:6]:
    row = manifest.loc[sid]
    plt.figure(figsize=(3, 3))
    plt.imshow(PILImage.open(DATA_ROOT / row["image_path"]).convert("RGB"))
    plt.axis("off")
    plt.title(row["label"])
    plt.show()
    for mode, by_id in recs.items():
        r = by_id.get(sid)
        if r and r["valid"]:
            print(f"[{mode}] -> {map_dish_to_class(r['output']['dish_name'], classes)}")
            print(json.dumps(r["output"], ensure_ascii=False))
        elif r:
            print(f"[{mode}] invalid: {r['error']}")
''')

nb.md(r'''
## 5. Chấm tay
1. Mở `MyDrive/foodmm/vlm/qwen3vl4b/manual_grading.csv` (Google Sheets hoặc Excel). Các dòng đã được xáo trộn và **không ghi chế độ prompt**, để chấm mù. Đừng mở `manual_grading_key.csv` trước khi chấm xong.
2. Điền `grade_dish` (0/1), `grade_ingredients` (0: sai, 1: một phần, 2: tốt), `grade_method` (0/1). Có thể ghi chú ở `notes`.
3. Lưu lại dưới dạng CSV cùng tên, rồi chạy cell dưới để tính điểm trung bình theo chế độ.
''')
nb.code(r'''
run("vlm_evaluate.py", "--set", *BASE)
scores = VLM_DIR / "manual_scores.csv"
print(scores.read_text() if scores.exists() else "Chưa có dòng nào được chấm.")
''')

if __name__ == "__main__":
    raise SystemExit(nb.main(NOTEBOOK_DIR / "03_vlm.ipynb"))
