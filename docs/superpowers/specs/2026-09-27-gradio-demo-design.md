# Sub-project 4: Demo Gradio

- Ngày: 2026-09-27
- Trạng thái: đã duyệt thiết kế, chờ duyệt spec
- Phạm vi: sub-project 4 trong 4. Dùng lại CLIP encoder và các head của Mốc 2, cùng VLM backend của sub-project 3.

## 1. Mục tiêu

Một web demo chạy trên Colab. Người dùng tải ảnh món ăn lên, có thể nhập thêm text, rồi xem top-5 dự đoán của 3 nhánh (ảnh, text, fusion) cạnh nhau. Tùy chọn trích xuất JSON bằng VLM.

### Tiêu chí hoàn thành

1. `scripts/demo.py` mở app Gradio với link `share=True` có đăng nhập. Tài khoản lấy từ Colab Secrets.
2. Hiển thị top-5 cho image, text và fusion. Thiếu ảnh hoặc thiếu text thì nhánh đó hiện "không có dữ liệu", còn fusion chạy ở chế độ thiếu modality.
3. Checkbox VLM: lần đầu bật mới nạp model, các lần sau dùng lại. Nạp lỗi thì hiện thông báo, app không bị crash.
4. Tab ví dụ có sẵn: kết quả tính trước (kể cả JSON của VLM nếu đã tạo), xem được khi không có GPU.
5. Pytest trên CPU pass.

### Ngoài phạm vi

Deploy lâu dài (HF Spaces), quản lý nhiều người dùng, lưu lịch sử.

## 2. Quyết định đã chốt

| Chủ đề | Quyết định |
|---|---|
| Mô hình phân loại | CLIP ViT-B/16 đóng băng và các head của Mốc 2: `image`, `text_<mask>`, `demo.fusion_run` (mặc định `xattn_strict`). Một encoder phục vụ cả 3 nhánh nên nhẹ và nhanh |
| Text người dùng | Làm sạch rồi che theo `demo.text_mask` (mặc định `strict`, khớp với dữ liệu huấn luyện của head). Text sau khi che được hiển thị để minh bạch |
| VLM | `HFVLMBackend` của sub-project 3, chế độ `image_hint` với top-3 của fusion. Nếu không có ảnh thì không chạy |
| Phiên bản | `gradio==6.28.0` (PyPI ngày 2026-09-27) |
| Bảo mật | `share=True` tạo URL công khai, nên bắt buộc có `auth`. Tài khoản lấy từ biến môi trường `DEMO_USERNAME`/`DEMO_PASSWORD` (notebook đọc từ Colab Secrets). Nếu thiếu thì script từ chối share, trừ khi có `--no_auth` |

## 3. Kiến trúc

```
src/foodmm/demo/
├── __init__.py
├── predictor.py   # DemoPredictor: nạp ClipEncoder + 3 head từ clip/runs, predict(image, text)
├── vlm_service.py # LazyVLM: nạp backend khi gọi lần đầu, bắt lỗi, trả dict
├── examples.py    # build_examples (tính trước), load_examples
└── app.py         # build_app(predictor, vlm, examples) -> gr.Blocks, các hàm xử lý thuần Python
scripts/
├── build_demo_examples.py   # chọn N ảnh test, tính dự đoán (+ VLM nếu có --with_vlm)
└── demo.py                  # nạp predictor, dựng app, launch
notebooks/04_demo.ipynb
```

- `DemoPredictor.predict(image: PIL | None, text: str | None) -> dict` trả về khóa `image`, `text`, `fusion` (mỗi khóa là dict `{class: prob}` top-5 hoặc `None`), `masked_text`, `top3_fusion`. Nếu cả ảnh và text đều trống thì raise `ValueError`.
- Fusion khi thiếu một modality dùng `drop_modalities` của Mốc 2. Đây là lý do nên dùng head có modality dropout.
- Các hàm xử lý (`on_predict`, `on_example`) là hàm Python thuần, nhận và trả giá trị cơ bản. Nhờ vậy test gọi trực tiếp được, không cần mở server.

### Giao diện

- **Tab "Dự đoán"**:
  - Đầu vào: `gr.Image(type="pil")`, `gr.Textbox` (text tùy chọn), `gr.Checkbox` "Trích xuất thông tin bằng VLM", nút "Dự đoán".
  - Đầu ra: 3 `gr.Label(num_top_classes=5)` (Ảnh, Text, Fusion), `gr.Textbox` text sau khi che, `gr.JSON` kết quả VLM.
- **Tab "Ví dụ có sẵn"**: `gr.Dropdown` chọn ví dụ, hiện ảnh, text, 3 label và JSON đã lưu.
- Tên lớp hiển thị dạng `apple pie`, không dùng `apple_pie`.

### Lưu trữ

```
MyDrive/foodmm/demo/
├── examples.json      # danh sách {name, id, label, image, text, pred, vlm}
└── examples/<id>.jpg
```

## 4. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Thiếu run head | `demo.py` dừng, in các run Mốc 2 cần chạy |
| Không có ảnh và text | Hiện cảnh báo `gr.Warning`, không gọi model |
| VLM nạp lỗi (không có GPU, hết VRAM) | JSON hiện `{"error": ...}`, các lần sau không nạp lại. Có gợi ý xem tab ví dụ |
| Chưa có `examples.json` | Tab ví dụ hiện hướng dẫn chạy `build_demo_examples.py` |
| Share mà không có tài khoản | Dừng, trừ khi có `--no_auth` |

## 5. Kiểm thử

- `test_demo_predictor.py` (mark `network`): tạo run head giả (khởi tạo ngẫu nhiên) với CLIP tí hon. Kiểm tra top-5 đủ ảnh và text, thiếu ảnh, thiếu text, cả hai trống thì lỗi, và text được che.
- `test_demo_app.py`: predictor và VLM giả. Kiểm tra `build_app` trả về `gr.Blocks`; `on_predict` đúng định dạng khi bật và tắt VLM; LazyVLM chỉ nạp một lần và lỗi được bắt; `on_example` khi có và khi không có `examples.json`.
- `test_demo_examples.py` (mark `network`): `build_demo_examples.py` trên dataset giả và run giả tạo ra `examples.json` cùng ảnh.
- `demo.py` từ chối share khi không có tài khoản (kiểm tra bằng `--dry_run`, thoát trước khi launch).
- Notebook 04 sinh đúng, các cell compile được.

## 6. Phụ thuộc

`gradio==6.28.0` được thêm vào `requirements-colab.txt` và `requirements-dev.txt`.

## 7. Rủi ro

- Link `share=True` công khai và hết hạn sau khoảng 1 tuần, hoặc khi session Colab tắt. Đây là demo tạm thời.
- VLM 4B và CLIP cùng nằm trên GPU: khoảng 10 GB với bf16, vừa với L4 hoặc A100. Trên T4 cần 4-bit.
- API của Gradio 6 có thể khác các ví dụ cũ trên mạng. Code chỉ dùng các component cơ bản (`Blocks`, `Tab`, `Image`, `Textbox`, `Checkbox`, `Button`, `Label`, `JSON`, `Dropdown`). Test dựng app thật để phát hiện lỗi API.
