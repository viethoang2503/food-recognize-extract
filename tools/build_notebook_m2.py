#!/usr/bin/env python
"""Generate notebooks/02_milestone2.ipynb (CLIP features and fusion heads)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_utils import NOTEBOOK_DIR, NotebookBuilder, add_data, add_setup  # noqa: E402

nb = NotebookBuilder()
nb.md(r'''
# Mốc 2: Đặc trưng CLIP và các kiểu fusion

CLIP ViT-B/16 được đóng băng. Đặc trưng được trích xuất một lần và lưu trên Drive (`foodmm/clip/features`), sau đó các head nhẹ được huấn luyện trên đặc trưng này.

**Thứ tự chạy:** setup → dữ liệu → chạy thử → trích xuất → các run → kết quả. Mọi bước đều chạy lại được: phần đã xong sẽ được bỏ qua.
''')
add_setup(nb)
add_data(nb)

nb.md(r'''
## 1. Chạy thử nhanh (5 lớp)
Ghi vào `/content/smoke_clip`, không đụng tới Drive. Dùng để kiểm tra pipeline và ước lượng thời gian.
''')
nb.code(r'''
SMOKE = ["--set", f"paths.data_root={DATA_ROOT}", "paths.work_dir=/content/smoke_clip",
         "data.subset_classes=5", "data.max_per_class=60", "head.epochs=3", "clip_suite.masks=[strict]"]
run("prepare_data.py", "--force", *SMOKE)
run("extract_clip.py", "--parts", "image,text_strict", "--force", *SMOKE)
run("zero_shot_clip.py", "--force", *SMOKE)
run("run_clip_suite.py", *SMOKE)
run("summarize_clip.py", *SMOKE)
''')

nb.md(r'''
## 2. Trích xuất đặc trưng CLIP
Ảnh train/val/test và text theo 2 chế độ che `none`, `strict`. Nếu mất session, chạy lại cell này để tiếp tục từ shard cuối cùng.
''')
nb.code(r'''
run("extract_clip.py", "--parts", "image,text_none,text_strict", "--set", *BASE)
''')

nb.md(r'''
## 3. Zero-shot và các run
`zeroshot`, `image`, `text_{none,strict}`, `{concat,gated,xattn}_{none,strict}`, `late_{none,strict}`. Mỗi run fusion tự đánh giá thêm khi bỏ ảnh / bỏ text trên tập test (bảng thiếu modality).
''')
nb.code(r'''
run("run_clip_suite.py", "--set", *BASE)
''')

nb.md(r'''
## 4. Kết quả
''')
nb.code(r'''
from IPython.display import Markdown, display

run("summarize_clip.py", "--set", *BASE)
RESULTS = WORK_DIR / "clip" / "results"
display(Markdown("### Kết quả chính (acc_lo/acc_hi: khoảng tin cậy 95%, bootstrap)\n" + (RESULTS / "main.md").read_text()))
display(Markdown("### Kiểm định cặp (diff = acc_b − acc_a, McNemar)\n" + (RESULTS / "significance.md").read_text()))
display(Markdown("### Thiếu modality (accuracy)\n" + (RESULTS / "missing.md").read_text()))
''')

nb.md(r'''
### Gated fusion: giá trị cổng trung bình
Giá trị gần 1 nghĩa là head dựa vào ảnh nhiều hơn, gần 0 là dựa vào text nhiều hơn.
''')
nb.code(r'''
import json

from foodmm.clip.features import clip_paths
from foodmm.config import load_config

CFG = load_config(overrides=BASE)
CP = clip_paths(CFG)
for p in sorted(CP["runs"].glob("gated_*/metrics_test.json")):
    m = json.loads(p.read_text())
    print(f"{m['run']:28s} acc={m['acc'] * 100:5.2f}%  mean_gate={m['mean_gate']:.3f}")
''')

nb.md(r'''
## 5. t-SNE trên 20 lớp của tập test
So sánh đặc trưng ảnh CLIP, đặc trưng text CLIP (`strict`) và đặc trưng fusion của `xattn_strict`.
''')
nb.code(r'''
import matplotlib.pyplot as plt

from foodmm.clip.features import image_set_name, load_feature_set, text_set_name
from foodmm.clip.report import plot_tsne, sample_for_tsne, tsne_2d
from foodmm.late_fusion import load_preds
from foodmm.utils import load_json

classes = load_json(WORK_DIR / "data" / "classes.json")
xattn = load_preds(CP["runs"] / "xattn_strict", "test")
idx = sample_for_tsne(xattn["labels"], n_classes=20, per_class=50, seed=0)
sources = {
    "Ảnh (CLIP pooled)": load_feature_set(CP["features"] / image_set_name("test"), ["pooled"])["pooled"],
    "Text strict (CLIP pooled)": load_feature_set(CP["features"] / text_set_name("strict", "test"), ["pooled"])["pooled"],
    "Fusion xattn_strict": xattn["features"],
}
for title, feats in sources.items():
    plot_tsne(tsne_2d(feats[idx], seed=0), xattn["labels"][idx], classes, title)
    plt.show()
''')

nb.md(r'''
## 6. So sánh với Mốc 1
Chỉ hiện khi `results/summary.md` của Mốc 1 đã có. Lưu ý: DistilBERT ở Mốc 1 đọc 256 token, còn CLIP chỉ đọc 77 token.
''')
nb.code(r'''
m1 = WORK_DIR / "results" / "summary.md"
if m1.exists():
    display(Markdown("### Mốc 1\n" + m1.read_text()))
    display(Markdown("### Mốc 2\n" + (RESULTS / "main.md").read_text()))
else:
    print("Chưa có kết quả Mốc 1 (notebooks/01_milestone1.ipynb).")
''')

if __name__ == "__main__":
    raise SystemExit(nb.main(NOTEBOOK_DIR / "02_milestone2.ipynb"))
