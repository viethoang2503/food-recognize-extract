# Sub-project 5: Tổng hợp kết quả và slide thuyết trình

- Ngày: 2026-09-28
- Trạng thái: bản nháp, chờ duyệt
- Phạm vi: sub-project 5, thêm sau 4 sub-project code. Không viết model mới, chỉ dùng kết quả của Mốc 1, Mốc 2, sub-project 3 và 4. Môn học không yêu cầu nộp báo cáo, nên sản phẩm cuối là slide, demo và các notebook kết quả.

## 1. Mục tiêu

Làm slide thuyết trình trả lời trực tiếp đề bài số 14:

> - Recognize food categories and extract useful information from both food images and related text.
> - Compare image-only, text-only, and multimodal approaches to determine whether combining both modalities improves food recognition and understanding.

Mọi kết luận trên slide phải chỉ ra được bảng hoặc hình làm bằng chứng (mục 3).

### Tiêu chí hoàn thành

1. Slide theo outline ở mục 5. Mỗi câu hỏi nghiên cứu có một câu kết luận, trích số liệu kèm khoảng tin cậy.
2. Có một slide demo (ảnh chụp màn hình hoặc video ngắn, phòng khi link Gradio hết hạn).
3. Bảng và hình trên slide được chép từ `MyDrive/foodmm/**/results/`, không vẽ tay lại số liệu.
4. Mọi con số trên slide khớp với file kết quả (kiểm tra lại lần cuối trước khi thuyết trình).
5. Một slide phụ lục ghi các hạn chế chính: chỉ một seed huấn luyện (CI chỉ phản ánh độ bất định do tập test); CLIP đọc 77 token text còn DistilBERT đọc 256; `strict` có thể che quá tay; text UPMC nhiễu; VLM chỉ đánh giá trên 200 mẫu và chấm tay 20 mẫu.

### Ngoài phạm vi

- Báo cáo viết (môn không yêu cầu).
- Thí nghiệm mới. Nếu thấy thiếu bằng chứng, ghi vào slide hạn chế thay vì mở thêm thí nghiệm, trừ khi còn thời gian (xem mục 6).

## 2. Câu hỏi nghiên cứu

| # | Câu hỏi | Ý của đề |
|---|---|---|
| RQ1 | Nhận diện 101 loại món từ ảnh, từ text, và từ cả hai đạt độ chính xác bao nhiêu? | Recognize food categories |
| RQ2 | Kết hợp ảnh và text có tốt hơn đơn modality một cách có ý nghĩa thống kê không, khi text không còn chứa tên món? | Compare … whether combining improves recognition |
| RQ3 | Kiểu fusion nào tốt nhất khi dùng cùng encoder (late, concat, gated, cross-attention)? Model học sâu so với TF-IDF và CLIP zero-shot thì sao? | Multimodal approaches |
| RQ4 | Fusion còn hoạt động thế nào khi thiếu một modality lúc test? | Understanding (mở rộng) |
| RQ5 | VLM trích xuất thông tin (tên món, nguyên liệu, cách nấu) từ ảnh, từ text và từ cả hai thì chế độ nào tốt nhất? | Extract useful information, understanding |
| RQ6 | Leakage tên món trong text ảnh hưởng tới kết luận như thế nào? | Điều kiện để so sánh công bằng |

## 3. Bằng chứng cho từng câu hỏi

| RQ | Bảng / hình | Nguồn |
|---|---|---|
| RQ1 | Bảng 9 run của Mốc 1 (Acc, Top-5, Macro-F1); bảng `main` của Mốc 2 | `results/summary.md`, `clip/results/main.md` |
| RQ2 | Bảng `significance`: image vs xattn, text vs xattn, image vs early, text vs early | `clip/results/significance.md` |
| RQ2 | Lưới ví dụ: image sai và fusion đúng, và ngược lại; accuracy theo lớp (lớp nào được lợi khi thêm text) | notebook 01 |
| RQ3 | Bảng `main` theo head; các cặp late/concat/gated vs xattn, early (Mốc 1) vs xattn (Mốc 2), zero-shot vs image | `clip/results/main.md`, `significance.md` |
| RQ4 | `missing.md`; t-SNE | `clip/results/`, notebook 02 |
| RQ5 | `eval_summary.md` (bỏ qua cột ingredient grounding khi so sánh chế độ), `classifier_on_sample.md`, `manual_scores.csv`, 2–3 ví dụ JSON | `vlm/qwen3vl4b/` |
| RQ6 | Bảng leakage của EDA; chênh lệch giữa run `none` và `strict` | notebook 01, `summary.md`, `main.md` |

## 4. Quy tắc viết kết luận

- **Chế độ che chính là `strict`.** Kết luận "multimodal có giúp không" dựa trên các run `strict`. Run `none` được trình bày như mức trần khi text chứa sẵn tên món, không dùng làm kết luận chính.
- **Chỉ nói "tốt hơn"** khi khoảng tin cậy 95% của chênh lệch không chứa 0 và McNemar p < 0.05. Nếu không đạt thì viết "chưa đủ bằng chứng", không viết "tương đương".
- Mọi accuracy trong phần kết luận đều kèm khoảng tin cậy.
- Với VLM, so sánh các chế độ bằng dish accuracy (kèm CI) và điểm chấm tay. Ingredient grounding chỉ để tham khảo, lý do ghi ở spec sub-project 3, mục 7.
- Khi so sánh VLM với classifier, nêu rõ VLM trả lời tự do rồi mới được ánh xạ về 101 lớp, còn classifier chọn trong tập đóng.

## 5. Outline slide (khoảng 14 slide)

1. Tiêu đề và thành viên
2. Đề bài và câu hỏi nghiên cứu
3. Dữ liệu, ví dụ ảnh kèm text, vấn đề leakage
4. Tổng quan pipeline (sơ đồ 4 sub-project)
5. Mốc 1: các baseline
6. Mốc 2: CLIP và các head fusion
7. Kết quả chính: image vs text vs fusion (`strict`), kèm CI
8. Kiểm định: bảng significance rút gọn
9. `none` vs `strict`: ảnh hưởng của leakage
10. Thiếu modality: bảng `missing`
11. VLM: ví dụ JSON và bảng 3 chế độ
12. VLM so với classifier trên cùng 200 mẫu
13. Demo
14. Kết luận và hạn chế

## 6. Thứ tự ưu tiên khi thiếu thời gian

Nếu không kịp làm hết các sub-project, bảo đảm có đủ bằng chứng theo thứ tự sau:

1. Mốc 1 và bảng `main` + `significance` của Mốc 2 (RQ1, RQ2, RQ3, RQ6). Đây là phần cốt lõi của đề.
2. VLM với 3 chế độ `image`, `text`, `image_text` (RQ5).
3. Bảng thiếu modality (RQ4). Bảng này tự có khi chạy xong các run chính của Mốc 2.
4. Demo. Nếu không kịp, dùng tab ví dụ có sẵn hoặc ảnh chụp màn hình.

Nếu còn thời gian: chạy 3 seed cho các head chính của Mốc 2 (mỗi head chỉ mất vài phút) để trình bày thêm mean ± std; nhờ người thứ hai chấm lại một phần file chấm tay để tính Cohen's kappa.
