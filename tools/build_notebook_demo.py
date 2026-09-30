#!/usr/bin/env python
"""Generate notebooks/04_demo.ipynb (Gradio demo on Colab)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_utils import NOTEBOOK_DIR, NotebookBuilder, add_data, add_setup  # noqa: E402

nb = NotebookBuilder()
nb.md(r'''
# Sub-project 4: Demo Gradio

Tải ảnh món ăn lên, có thể nhập thêm text, rồi xem top-5 của 3 nhánh: chỉ ảnh, chỉ text, fusion (`xattn_strict`). Có thể bật trích xuất JSON bằng Qwen3-VL.

**Cần có trước:** các run `image`, `text_strict`, `xattn_strict` của Mốc 2 trên Drive.

**Bảo mật:** `share=True` tạo link công khai. Thêm Colab Secrets `DEMO_USERNAME` và `DEMO_PASSWORD` trước khi chạy. Nếu thiếu, script sẽ từ chối tạo link.
''')
add_setup(nb)
add_data(nb)

nb.md(r'''
## 1. Tạo ví dụ có sẵn
Chạy một lần. Thêm `--with_vlm` nếu muốn lưu cả kết quả VLM (cần GPU, lâu hơn). Tab "Ví dụ có sẵn" vẫn xem được khi không có GPU.
''')
nb.code(r'''
if not (WORK_DIR / "demo" / "examples.json").exists():
    run("build_demo_examples.py", "--n", "8", "--set", *BASE)
''')

nb.md(r'''
## 2. Chạy demo
Mở link `*.gradio.live` in ra bên dưới và đăng nhập bằng tài khoản trong Colab Secrets. Dừng cell để tắt demo.
''')
nb.code(r'''
from google.colab import userdata

os.environ["DEMO_USERNAME"] = userdata.get("DEMO_USERNAME")
os.environ["DEMO_PASSWORD"] = userdata.get("DEMO_PASSWORD")
run("demo.py", "--share", "--set", *BASE)
''')

if __name__ == "__main__":
    raise SystemExit(nb.main(NOTEBOOK_DIR / "04_demo.ipynb"))
