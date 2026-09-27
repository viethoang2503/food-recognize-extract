# Sub-project 3: Trích xuất thông tin món ăn bằng VLM (Qwen3-VL)

- Ngày: 2026-09-27
- Trạng thái: đã duyệt thiết kế, chờ duyệt spec
- Phạm vi: sub-project 3 trong 4. Dùng lại manifest, config và dự đoán của Mốc 1 hoặc Mốc 2.

## 1. Mục tiêu

Chạy một VLM mã nguồn mở ngay trong Colab (qua `transformers`, không tự host server, không gọi API) để trích xuất thông tin có cấu trúc từ ảnh món ăn. Sau đó đánh giá chất lượng, và so sánh 3 cách đưa ngữ cảnh vào prompt.

### Tiêu chí hoàn thành

1. File `jsonl` kết quả cho 500 ảnh test, với 3 chế độ prompt. Chạy lại được và tự tiếp tục từ chỗ dừng.
2. Bảng so sánh 3 chế độ: tỷ lệ JSON hợp lệ, số lần retry, latency trung bình và p90, dish accuracy (sau khi ánh xạ về 101 lớp), ingredient grounding.
3. File CSV 50 mẫu để chấm tay, cùng một hàm đọc lại file đã chấm và tính điểm.
4. Pytest trên CPU: backend giả cho logic, và một test `network` với Qwen3-VL tí hon (`trl-internal-testing/tiny-Qwen3VLForConditionalGeneration`).

### Ngoài phạm vi

- Fine-tune VLM. Dùng API (Gemini, GPT). Chạy trên toàn bộ tập test (có thể làm bằng tham số `--n`, nhưng không nằm trong tiêu chí).

## 2. Quyết định đã chốt

| Chủ đề | Quyết định |
|---|---|
| Model mặc định | `Qwen/Qwen3-VL-4B-Instruct`. Bản 8B là tùy chọn qua `vlm.model_name` |
| Nạp model | `AutoModelForImageTextToText` + `AutoProcessor` (`transformers==5.17.0`) |
| Độ chính xác số | `vlm.dtype: auto`: bf16 nếu GPU hỗ trợ (L4, A100); nếu không (T4) thì 4-bit NF4 qua `bitsandbytes==0.50.2`. Có thể ép bằng `vlm.quantize: 4bit|none` |
| Giải mã | Greedy (`do_sample=False`), `max_new_tokens` 256 |
| Ảnh | Thu nhỏ để cạnh dài nhất ≤ `vlm.max_image_side` (768) trước khi đưa vào processor, giới hạn số token ảnh |
| Mẫu đánh giá | 500 ảnh test, stratified khoảng 5 ảnh mỗi lớp, seed cố định, lưu thành `sample_ids.json` |

## 3. Schema JSON

```json
{
  "dish_name": "string",
  "cuisine": "string",
  "main_ingredients": ["string", "..."],
  "cooking_method": "string",
  "confidence": 0.0
}
```

Quy tắc validate (`foodmm/vlm/parse.py`):

- Lấy object JSON đầu tiên trong output. Chấp nhận code fence ```` ```json ````, bỏ chữ thừa trước và sau.
- Đủ 5 khóa. Khóa thừa bị bỏ.
- `dish_name` là chuỗi không rỗng.
- `main_ingredients` là list chuỗi (chuỗi đơn thì tách theo dấu phẩy), tối đa 15 phần tử, chuyển chữ thường.
- `confidence` là số trong [0, 1]. Nếu là phần trăm (1 < x ≤ 100) thì chia 100.
- Hợp lệ thì trả về dict đã chuẩn hóa. Không hợp lệ thì raise `ExtractionError` kèm lý do.

## 4. Chế độ prompt (`foodmm/vlm/prompts.py`)

Cả 3 chế độ dùng chung system instruction (mô tả schema, yêu cầu chỉ trả về JSON, tiếng Anh) và một ảnh.

| Chế độ | Ngữ cảnh thêm vào |
|---|---|
| `image` | Không có |
| `image_text` | Text của mẫu ở chế độ `vlm.text_mask` (mặc định `strict`), cắt còn `vlm.max_context_chars` (1000). Ghi chú rằng text có thể nhiễu và tên món đã bị che |
| `image_hint` | Top-3 lớp cùng xác suất từ run `vlm.hint_run` (mặc định `xattn_strict` của Mốc 2 trong `clip/runs/`; đổi `vlm.hint_runs_dir: runs` để dùng `early_strict` của Mốc 1). Ghi chú rằng gợi ý có thể sai |

Retry: nếu output không hợp lệ, gửi lại một lần với prompt nhắc lại lỗi và yêu cầu chỉ trả JSON. Nếu vẫn lỗi, bản ghi có `valid: false`, `error`, và `raw` là output gốc.

## 5. Kiến trúc

```
src/foodmm/vlm/
├── __init__.py
├── parse.py      # extract_json, validate_output, ExtractionError
├── prompts.py    # SYSTEM_PROMPT, build_prompt(mode, text=None, hints=None), RETRY_PROMPT
├── backend.py    # VLMBackend (protocol), HFVLMBackend (nạp lazy, bf16 hoặc 4-bit), resize_for_vlm
├── extract.py    # select_sample, load_hints, run_extraction (resume jsonl, retry, đo latency)
└── evaluate.py   # map_dish_to_class, ingredient_grounding, summarize_mode, make_grading_sheet, score_grading_sheet
scripts/
├── vlm_extract.py    # --mode image|image_text|image_hint|all  --n 500
└── vlm_evaluate.py   # bảng tổng hợp và file chấm tay
notebooks/03_vlm.ipynb
```

`VLMBackend` chỉ có một phương thức `generate(image, prompt, system) -> str`. Nhờ vậy, logic extract và evaluate được test bằng backend giả, không cần GPU. Demo ở sub-project 4 cũng dùng lại backend này.

### Lưu trữ

```
MyDrive/foodmm/vlm/<vlm.tag>/        # mặc định qwen3vl4b
├── sample_ids.json
├── extract_{image,image_text,image_hint}.jsonl
├── eval_summary.csv, eval_summary.md
├── manual_grading.csv               # người dùng điền cột grade_* rồi chạy lại vlm_evaluate.py
└── manual_scores.csv                # sinh ra khi manual_grading.csv đã được chấm
```

Mỗi dòng jsonl gồm: `id, label, mode, valid, attempts, latency_s, output` (dict đã chuẩn hóa hoặc `null`), `raw`, `error`.

## 6. Luồng chạy

1. `select_sample(manifest, n, seed)`: lấy mẫu stratified từ split test, ghi `sample_ids.json` ở lần đầu, các lần sau đọc lại file này.
2. `image_hint` cần dự đoán của `vlm.hint_run`: đọc `preds_test.npz`, lấy top-3 theo `ids`. Thiếu file thì dừng và chỉ rõ run cần chạy.
3. Với mỗi id chưa có trong jsonl: đọc ảnh, resize, dựng prompt, gọi backend, validate, retry tối đa 1 lần, rồi ghi một dòng và flush ngay. Mất session thì chạy lại, tiếp tục từ id kế tiếp.
4. In tiến độ mỗi 20 mẫu: số đã xong, tỷ lệ hợp lệ, latency trung bình.

## 7. Đánh giá (`vlm_evaluate.py`)

- **Tỷ lệ JSON hợp lệ** và **tỷ lệ cần retry**.
- **Latency**: trung bình và p90, tính cả retry.
- **Dish accuracy**: `map_dish_to_class(dish_name, classes)`.
  - Nếu một cụm tên lớp (có biến thể số ít/số nhiều) nằm trong `dish_name` thì chọn cụm dài nhất.
  - Nếu không, lấy lớp có điểm cao nhất, với điểm là max của `difflib.SequenceMatcher` ratio và Jaccard theo từ.
  - Dưới `vlm.match_threshold` (0.6) thì là `unmapped`.
  - Báo cáo accuracy (unmapped tính là sai) và tỷ lệ unmapped.
- **Ingredient grounding**: với mỗi nguyên liệu dự đoán, kiểm tra có từ nội dung nào (≥ 3 ký tự, không phải từ nối, có biến thể số ít/số nhiều) xuất hiện trong text gốc chưa che của mẫu hay không. Điểm của một mẫu là tỷ lệ nguyên liệu khớp; báo cáo trung bình trên các mẫu hợp lệ có nguyên liệu. Text UPMC là text trang web nên nhiễu. Chỉ số này là proxy, không phải ground truth.
- **Chấm tay**: `make_grading_sheet` lấy 50 id đầu tiên của `sample_ids.json` × 3 chế độ, gồm các cột `id, label, mode, image_path, dish_name, cuisine, main_ingredients, cooking_method, confidence, grade_dish, grade_ingredients, grade_method, notes`. Thang điểm: dish 0/1, ingredients 0–2, method 0/1. File đã tồn tại thì không ghi đè. `score_grading_sheet` tính điểm trung bình theo chế độ, chỉ trên các dòng đã chấm.
- Kết quả ghi vào `eval_summary.csv` và `.md`, mỗi chế độ một hàng.

## 8. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Không có GPU | Dừng với thông báo, trừ khi có `vlm.allow_cpu=true` (chỉ dùng cho test) |
| Hết VRAM khi nạp bf16 | Thông báo gợi ý `vlm.quantize=4bit` hoặc model nhỏ hơn |
| Output không phải JSON sau retry | Ghi bản ghi `valid: false`, tiếp tục |
| Lỗi đọc ảnh | Ghi bản ghi `valid: false, error: "bad image"`, tiếp tục |
| Thiếu run cho `image_hint` | Dừng trước khi nạp model |

## 9. Kiểm thử

- `test_vlm_parse.py`: code fence, chữ thừa, JSON lồng, thiếu khóa, confidence phần trăm, nguyên liệu dạng chuỗi.
- `test_vlm_prompts.py`: đủ 3 chế độ, cắt text, định dạng hint, chế độ sai thì báo lỗi.
- `test_vlm_extract.py`: backend giả trả JSON hỏng rồi JSON đúng (retry), resume không chạy lại id đã có, lấy mẫu stratified ổn định.
- `test_vlm_evaluate.py`: ánh xạ tên món (chính xác, số nhiều, gần đúng, unmapped), grounding, bảng tổng hợp, file chấm tay và tính điểm.
- `test_vlm_backend.py` (mark `network`): Qwen3-VL tí hon trên CPU với `vlm.allow_cpu=true` trả về chuỗi (nội dung ngẫu nhiên). Mục đích là kiểm tra đường gọi API của `transformers`.
- `test_vlm_pipeline.py`: `vlm_extract.py` và `vlm_evaluate.py` chạy end-to-end với `vlm.backend=fake` trên dataset giả.
- Notebook 03 sinh đúng, các cell compile được.

## 10. Phụ thuộc

`bitsandbytes==0.50.2` (chỉ trên Colab, dùng khi 4-bit) và `accelerate` (bản có sẵn trên Colab) được thêm vào `requirements-colab.txt`. Local test không cần `bitsandbytes`.

## 11. Rủi ro

- Chưa đo latency thật. Ước tính 1–3 giây mỗi ảnh với 4B bf16 trên L4, tức khoảng 1 giờ cho 1500 lần gọi. Có thể giảm `--n` hoặc chạy từng chế độ riêng.
- Tên món VLM trả về có thể không trùng tên lớp (ví dụ "spag bol" so với `spaghetti_bolognese`). Ánh xạ mờ sẽ bỏ sót một phần. Tỷ lệ unmapped được báo cáo, và phần chấm tay dùng để đối chiếu.
- Chế độ `image_hint` có thể làm VLM chép lại gợi ý. Bảng so sánh sẽ cho thấy điều này: accuracy của `image_hint` gần với accuracy của run gợi ý.
