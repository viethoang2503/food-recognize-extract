# Sub-project 3: Trích xuất thông tin món ăn bằng VLM (Qwen3-VL)

- Ngày: 2026-09-27
- Trạng thái: đã duyệt, chưa triển khai (cập nhật 2026-09-28: thêm chế độ `text`, khoảng tin cậy, so sánh với classifier, chấm mù; thu gọn: bỏ `image_hint`, 200 mẫu, chấm tay 20 mẫu)
- Phạm vi: sub-project 3 trong 4. Dùng lại manifest, config và dự đoán của Mốc 1 hoặc Mốc 2.
- Thứ tự: chỉ bắt đầu sau khi xong Task 8b của Mốc 2 (`foodmm/stats.py`, dùng cho khoảng tin cậy). Bảng so sánh classifier cần các run Mốc 2 `image`, `text_strict`, `xattn_strict`; run nào thiếu thì bỏ qua, nên phần trích xuất chạy được mà không cần Mốc 2.

## 1. Mục tiêu

Chạy một VLM mã nguồn mở ngay trong Colab (qua `transformers`, không tự host server, không gọi API) để trích xuất thông tin có cấu trúc về món ăn từ ảnh, từ text, hoặc cả hai. Sau đó đánh giá chất lượng và trả lời phần "hiểu món ăn" của đề bài: **kết hợp ảnh và text có giúp trích xuất tốt hơn so với chỉ dùng ảnh hoặc chỉ dùng text không?**

Có 3 chế độ: `image` (chỉ ảnh), `text` (chỉ text) và `image_text` (cả hai).

### Tiêu chí hoàn thành

1. File `jsonl` kết quả cho 200 mẫu test, với 3 chế độ prompt. Chạy lại được và tự tiếp tục từ chỗ dừng.
2. Bảng so sánh 3 chế độ: tỷ lệ JSON hợp lệ, số lần retry, latency trung bình và p90, dish accuracy (sau khi ánh xạ về 101 lớp) kèm khoảng tin cậy 95%, ingredient grounding (chỉ tham khảo).
3. Bảng accuracy của các classifier (`image`, `text_strict`, `xattn_strict` của Mốc 2) trên **đúng 200 mẫu đó**, để so sánh trực tiếp với dish accuracy của VLM.
4. File CSV 20 mẫu × 3 chế độ để chấm tay **mù** (xáo trộn, ẩn chế độ), cùng một hàm đọc lại file đã chấm và tính điểm theo chế độ.
5. Pytest trên CPU: backend giả cho logic, và một test `network` với Qwen3-VL tí hon (`trl-internal-testing/tiny-Qwen3VLForConditionalGeneration`).

### Ngoài phạm vi

- Fine-tune VLM. Dùng API (Gemini, GPT). Chạy trên toàn bộ tập test (có thể tăng `vlm.n_samples`, nhưng không nằm trong tiêu chí).
- Chế độ `image_hint` (ảnh + top-3 của classifier).

## 2. Quyết định đã chốt

| Chủ đề | Quyết định |
|---|---|
| Model mặc định | `Qwen/Qwen3-VL-4B-Instruct`. Bản 8B là tùy chọn qua `vlm.model_name` |
| Nạp model | `AutoModelForImageTextToText` + `AutoProcessor` (`transformers==5.17.0`) |
| Độ chính xác số | `vlm.dtype: auto`: bf16 nếu GPU hỗ trợ (L4, A100); nếu không (T4) thì 4-bit NF4 qua `bitsandbytes==0.50.2`. Có thể ép bằng `vlm.quantize: 4bit|none` |
| Giải mã | Greedy (`do_sample=False`), `max_new_tokens` 256 |
| Ảnh | Thu nhỏ để cạnh dài nhất ≤ `vlm.max_image_side` (768) trước khi đưa vào processor, giới hạn số token ảnh |
| Mẫu đánh giá | 200 mẫu test, stratified khoảng 2 mẫu mỗi lớp, seed cố định, lưu thành `sample_ids.json`. Cả 3 chế độ dùng chung mẫu này |
| Chế độ chỉ có text | Cùng model Qwen3-VL, không gửi ảnh. Không cần model text riêng, nên so sánh giữa các chế độ chỉ khác nhau ở input |
| Chỉ số chính khi so sánh chế độ | Dish accuracy (tự động, có khoảng tin cậy) và điểm chấm tay. Ingredient grounding chỉ để tham khảo (xem mục 7) |

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

Cả 3 chế độ dùng chung system instruction (mô tả schema, yêu cầu chỉ trả về JSON, tiếng Anh). Mọi chế độ trừ `text` đều gửi kèm một ảnh.

| Chế độ | Ngữ cảnh thêm vào |
|---|---|
| `image` | Không có |
| `text` | Không có ảnh. Chỉ có text của mẫu ở chế độ `vlm.text_mask` (mặc định `strict`), cắt còn `vlm.max_context_chars`. Ghi chú rằng text có thể nhiễu và tên món đã bị che |
| `image_text` | Text của mẫu ở chế độ `vlm.text_mask` (mặc định `strict`), cắt còn `vlm.max_context_chars` (1000). Ghi chú rằng text có thể nhiễu và tên món đã bị che |

Retry: nếu output không hợp lệ, gửi lại một lần với prompt nhắc lại lỗi và yêu cầu chỉ trả JSON. Nếu vẫn lỗi, bản ghi có `valid: false`, `error`, và `raw` là output gốc.

## 5. Kiến trúc

```
src/foodmm/vlm/
├── __init__.py
├── parse.py      # extract_json, validate_output, ExtractionError
├── prompts.py    # MODES, TEXT_MODES, SYSTEM_PROMPT, build_prompt(mode, text=None), RETRY_PROMPT
├── backend.py    # VLMBackend (protocol), HFVLMBackend (nạp lazy, bf16 hoặc 4-bit), resize_for_vlm
├── extract.py    # select_sample, run_extraction (resume jsonl, retry, đo latency)
└── evaluate.py   # map_dish_to_class, ingredient_grounding, summarize_mode, classifier_on_sample,
                  # make_grading_sheet, score_grading_sheet
scripts/
├── vlm_extract.py    # --mode image|text|image_text|all  [--n N]
└── vlm_evaluate.py   # bảng tổng hợp và file chấm tay
notebooks/03_vlm.ipynb
```

`VLMBackend` chỉ có một phương thức `generate(image | None, prompt, system) -> str`; `image=None` dùng cho chế độ `text`. Nhờ vậy, logic extract và evaluate được test bằng backend giả, không cần GPU. Demo ở sub-project 4 cũng dùng lại backend này. Khoảng tin cậy dùng `bootstrap_ci` trong `foodmm/stats.py` (Mốc 2, Task 8b).

### Lưu trữ

```
MyDrive/foodmm/vlm/<vlm.tag>/        # mặc định qwen3vl4b
├── sample_ids.json
├── extract_{image,text,image_text}.jsonl
├── eval_summary.csv, eval_summary.md
├── classifier_on_sample.csv, .md    # accuracy của classifier trên cùng 200 mẫu
├── manual_grading.csv               # xáo trộn, không có cột mode; người dùng điền cột grade_* rồi chạy lại vlm_evaluate.py
├── manual_grading_key.csv           # row -> mode, chỉ dùng khi tính điểm
└── manual_scores.csv                # sinh ra khi manual_grading.csv đã được chấm
```

Mỗi dòng jsonl gồm: `id, label, mode, valid, attempts, latency_s, output` (dict đã chuẩn hóa hoặc `null`), `raw`, `error`.

## 6. Luồng chạy

1. `select_sample(manifest, n, seed)`: lấy mẫu stratified từ split test, ghi `sample_ids.json` ở lần đầu, các lần sau đọc lại file này.
2. Với mỗi id chưa có trong jsonl: dựng prompt, đọc và resize ảnh (bỏ qua ở chế độ `text`), gọi backend, validate, retry tối đa 1 lần, rồi ghi một dòng và flush ngay. Mất session thì chạy lại, tiếp tục từ id kế tiếp.
3. In tiến độ mỗi 20 mẫu: số đã xong, tỷ lệ hợp lệ, latency trung bình.

## 7. Đánh giá (`vlm_evaluate.py`)

- **Tỷ lệ JSON hợp lệ** và **tỷ lệ cần retry**.
- **Latency**: trung bình và p90, tính cả retry.
- **Dish accuracy**: `map_dish_to_class(dish_name, classes)`.
  - Nếu một cụm tên lớp (có biến thể số ít/số nhiều) nằm trong `dish_name` thì chọn cụm dài nhất.
  - Nếu không, lấy lớp có điểm cao nhất, với điểm là max của `difflib.SequenceMatcher` ratio và hệ số overlap theo từ (|A∩B| / min(|A|, |B|), để "fries" khớp `french_fries`).
  - Dưới `vlm.match_threshold` (0.6) thì là `unmapped`.
  - Báo cáo accuracy (output không hợp lệ và unmapped tính là sai), khoảng tin cậy 95% bằng bootstrap (`vlm.n_boot` = 1000 lần), và tỷ lệ unmapped.
- **So sánh với classifier trên cùng mẫu** (`classifier_on_sample`): đọc `preds_test.npz` của các run trong `vlm.compare_runs` (mặc định `image`, `text_strict`, `xattn_strict` của Mốc 2), chỉ lấy 200 id của mẫu, tính accuracy và khoảng tin cậy. Run nào chưa có thì bỏ qua và in thông báo. Nhờ vậy có thể đặt cạnh nhau: VLM `image` với classifier `image`, VLM `text` với `text_strict`, VLM `image_text` với `xattn_strict`.
- **Ingredient grounding**: với mỗi nguyên liệu dự đoán, kiểm tra có từ nội dung nào (≥ 3 ký tự, không phải từ nối, có biến thể số ít/số nhiều) xuất hiện trong text gốc chưa che của mẫu hay không. Điểm của một mẫu là tỷ lệ nguyên liệu khớp; báo cáo trung bình trên các mẫu hợp lệ có nguyên liệu. Text UPMC là text trang web nên nhiễu. Chỉ số này là proxy, không phải ground truth. **Không dùng chỉ số này để so sánh các chế độ:** chế độ `text` và `image_text` được đọc chính text dùng để đo, nên điểm bị thổi phồng. Bảng tổng hợp có cột `uses_text` để đánh dấu, và file `.md` có ghi chú.
- **Chấm tay (mù)**: đây là chỉ số chính cho nguyên liệu và cách nấu. `make_grading_sheet` lấy `vlm.n_grading` (20) id đầu tiên của `sample_ids.json` × 3 chế độ, xáo trộn thứ tự (seed cố định), đánh số `row`. File chấm gồm các cột `row, id, label, image_path, dish_name, cuisine, main_ingredients, cooking_method, confidence, grade_dish, grade_ingredients, grade_method, notes` và **không có cột mode**. Cột mode được ghi riêng vào `manual_grading_key.csv` (`row, mode`). Thang điểm: dish 0/1, ingredients 0–2, method 0/1. File đã tồn tại thì không ghi đè. `score_grading_sheet` ghép lại với file key, rồi tính điểm trung bình theo chế độ, chỉ trên các dòng đã chấm. Thiếu file key thì dừng với thông báo.
- Kết quả ghi vào `eval_summary.csv` và `.md` (mỗi chế độ một hàng, các cột `mode, uses_text, n, valid_rate, retry_rate, latency_mean, latency_p90, dish_acc, dish_acc_lo, dish_acc_hi, unmapped_rate, ingredient_grounding`), và `classifier_on_sample.csv` / `.md`.

## 8. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Không có GPU | Dừng với thông báo, trừ khi có `vlm.allow_cpu=true` (chỉ dùng cho test) |
| Hết VRAM khi nạp bf16 | Thông báo gợi ý `vlm.quantize=4bit` hoặc model nhỏ hơn |
| Output không phải JSON sau retry | Ghi bản ghi `valid: false`, tiếp tục |
| Lỗi đọc ảnh | Ghi bản ghi `valid: false, error: "bad image"`, tiếp tục |

## 9. Kiểm thử

- `test_vlm_parse.py`: code fence, chữ thừa, JSON lồng, thiếu khóa, confidence phần trăm, nguyên liệu dạng chuỗi.
- `test_vlm_prompts.py`: đủ 3 chế độ, chế độ `text` không nhắc tới ảnh, cắt text, chế độ sai (kể cả `image_hint`) thì báo lỗi.
- `test_vlm_extract.py`: backend giả trả JSON hỏng rồi JSON đúng (retry), resume không chạy lại id đã có, lấy mẫu stratified ổn định, chế độ `text` không gửi ảnh.
- `test_vlm_evaluate.py`: ánh xạ tên món (chính xác, số nhiều, gần đúng, unmapped), grounding, bảng tổng hợp có khoảng tin cậy và `uses_text`, classifier trên cùng mẫu (bỏ qua run thiếu), file chấm mù (không có cột mode, ghép lại qua key) và tính điểm.
- `test_vlm_backend.py` (mark `network`): Qwen3-VL tí hon trên CPU với `vlm.allow_cpu=true` trả về chuỗi (nội dung ngẫu nhiên), cả khi có ảnh và khi `image=None`. Mục đích là kiểm tra đường gọi API của `transformers`.
- `test_vlm_pipeline.py`: `vlm_extract.py` và `vlm_evaluate.py` chạy end-to-end với `vlm.backend=fake` trên dataset giả.
- Notebook 03 sinh đúng, các cell compile được.

## 10. Phụ thuộc

`bitsandbytes==0.50.2` (chỉ trên Colab, dùng khi 4-bit) và `accelerate` (bản có sẵn trên Colab) được thêm vào `requirements-colab.txt`. Local test không cần `bitsandbytes`.

## 11. Rủi ro

- Chưa đo latency thật. Ước tính 1–3 giây mỗi mẫu với 4B bf16 trên L4, tức khoảng 10–30 phút cho 600 lần gọi (chế độ `text` nhanh hơn vì không có token ảnh).
- Chế độ `text` dùng text đã che `strict`, nên dish accuracy của nó sẽ thấp. Đây là chủ đích, giống Mốc 1 và Mốc 2: đo xem text giúp được bao nhiêu khi không còn tên món. Có thể chạy thêm với `vlm.text_mask=none vlm.tag=qwen3vl4b_none` để thấy mức trần khi có leakage.
- Với 200 mẫu, khoảng tin cậy của dish accuracy rộng khoảng ±6–7 điểm phần trăm. Chênh lệch nhỏ hơn mức đó giữa các chế độ cần được trình bày là chưa kết luận được.
- Tên món VLM trả về có thể không trùng tên lớp (ví dụ "spag bol" so với `spaghetti_bolognese`). Ánh xạ mờ sẽ bỏ sót một phần. Tỷ lệ unmapped được báo cáo, và phần chấm tay dùng để đối chiếu.
