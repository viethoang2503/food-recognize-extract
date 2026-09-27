# Mốc 1: Pipeline dữ liệu và các baseline image / text / multimodal trên UPMC Food-101

- Ngày: 2026-09-27
- Trạng thái: chờ duyệt
- Phạm vi: sub-project 1 trong 4 (1. baseline, 2. fusion nâng cao và ablation, 3. trích xuất thông tin bằng VLM, 4. demo Gradio)

## 1. Mục tiêu

Xây dựng pipeline có thể tái lập để trả lời câu hỏi: kết hợp ảnh và text có giúp nhận diện món ăn tốt hơn so với chỉ dùng một modality không, và kết quả thay đổi thế nào khi text không còn chứa tên món.

### Tiêu chí hoàn thành

1. Notebook EDA gồm: phân bố lớp, độ dài text, lưới ảnh mẫu kèm text, bảng leakage (tỷ lệ text chứa tên lớp đúng, tỷ lệ chứa tên lớp khác, số token còn lại) theo 3 chế độ che `none` / `exact` / `strict`.
2. Bảng kết quả trên tập test (Accuracy, Top-5, Macro-F1) cho 9 run:
   `tfidf_none`, `tfidf_strict`, `text_none`, `text_strict`, `image`, `late_none`, `late_strict`, `early_none`, `early_strict`.
3. Confusion matrix của image-only và early fusion, top 10 cặp lớp hay nhầm, accuracy theo lớp.
4. Lưới ví dụ định tính: mẫu mà image-only sai nhưng early fusion đúng, và ngược lại.
5. Bộ test pytest chạy trên CPU và pass.

### Ngoài phạm vi

- Embedding CLIP/ViT, các kiểu fusion gated và cross-attention, ablation thiếu modality, thêm nhiễu, giảm dữ liệu (Mốc 2).
- Trích xuất thông tin bằng VLM (sub-project 3) và demo (sub-project 4).
- Món ăn Việt Nam và text tiếng Việt: đã bỏ khỏi toàn bộ dự án.

## 2. Quyết định đã chốt

| Chủ đề | Quyết định |
|---|---|
| Dataset | UPMC Food-101 (Kaggle `gianmarco96/upmcfood101`), 101 lớp, ảnh và text ghép cặp |
| Tổ chức code | Module `.py` trong `src/`, notebook Colab chỉ điều khiển và phân tích |
| Đồng bộ | Repo GitHub **public**, notebook `git clone` / `git pull` mỗi session |
| Lưu trữ | Dữ liệu, checkpoint, kết quả trên Google Drive `MyDrive/foodmm/`, không commit vào repo |
| Backbone ảnh | `resnet50` (timm, pretrained ImageNet), tên model đặt trong config |
| Backbone text | `distilbert/distilbert-base-uncased` (HuggingFace), tên model đặt trong config |
| Early fusion | Fine-tune end-to-end cả 2 encoder, khởi tạo từ checkpoint đơn modality cùng chế độ text, modality dropout 10% |
| Late fusion | `w·p_img + (1−w)·p_txt`, dò w trên tập val |
| Chế độ text cho train | `none` và `strict`. `exact` chỉ dùng trong EDA (chạy thêm được bằng một tham số) |

## 3. Kiến trúc và luồng làm việc

### Cấu trúc repo

```
foodmm/
├── configs/default.yaml
├── src/foodmm/
│   ├── config.py            # đọc YAML, ghi đè bằng --set a.b=c
│   ├── utils.py             # seed, thiết bị, json, thư mục
│   ├── data/
│   │   ├── prepare.py       # quét ảnh + text, xây manifest, tách val
│   │   ├── text_utils.py    # làm sạch text, che tên món, đo leakage
│   │   └── datasets.py      # FoodDataset, transform, Collator
│   ├── models/
│   │   ├── image_model.py   # ImageClassifier (timm)
│   │   ├── text_model.py    # TextClassifier (HF, mean pooling)
│   │   ├── fusion.py        # MultimodalClassifier, late fusion
│   │   └── __init__.py      # build_model(modality, cfg, num_classes)
│   ├── engine.py            # train/eval, AMP, checkpoint, resume, predict
│   ├── metrics.py           # acc, top-5, macro-F1, per-class, confusion
│   ├── late_fusion.py       # đọc file dự đoán, kết hợp xác suất, dò w
│   └── analysis.py          # bảng kết quả, bảng leakage, so sánh run, hình (dùng trong notebook)
├── scripts/
│   ├── prepare_data.py
│   ├── train.py             # --modality image|text|multimodal
│   ├── train_tfidf.py
│   ├── late_fusion.py
│   └── summarize.py
├── tools/build_notebook.py  # sinh notebook từ danh sách cell (dễ review hơn JSON)
├── notebooks/01_milestone1.ipynb
├── tests/
├── requirements-colab.txt
└── .gitignore               # data/, runs/, *.pt, *.npz
```

### Luồng trên Colab

1. Mount Drive, clone hoặc pull repo, `pip install -r requirements-colab.txt`.
2. Lần đầu: tải dataset bằng Kaggle API (key lấy từ Colab Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`), lưu file zip vào `MyDrive/foodmm/data/`.
3. Mỗi session: giải nén file zip từ Drive ra `/content/data` (ổ local). Không đọc ảnh trực tiếp từ Drive.
4. Chạy script bằng `!python scripts/... --config configs/default.yaml --set ...`.

### Lưu trữ trên Drive

```
MyDrive/foodmm/
├── data/        # upmcfood101.zip, manifest.csv, classes.json, stats.json
├── runs/<run_name>/
│   ├── config.yaml, history.json
│   ├── last.pt, best.pt
│   ├── preds_val.npz, preds_test.npz   # logits, features, labels, ids
│   └── metrics_test.json
└── results/     # summary.csv, summary.md, hình
```

### Tái lập và chống mất việc

- Seed cố định. Config đầy đủ được ghi vào mỗi thư mục run.
- Tập val là 10% tách stratified từ train, lưu cố định trong manifest. Mọi run dùng chung manifest.
- Lưu `last.pt` sau mỗi epoch. Chạy lại cùng lệnh thì tự resume. `best.pt` được cập nhật khi val accuracy tăng.

## 4. Pipeline dữ liệu

### 4.1 Xây manifest (`prepare_data.py`)

Định dạng text trong bản Kaggle chưa được xác minh, nên loader tự nhận diện:

- **Ảnh:** quét đệ quy các đuôi `.jpg/.jpeg/.png/.webp`. Nhãn là tên thư mục cha. Split là `train` / `test` theo thư mục trên đường dẫn.
- **Text dạng CSV:** đọc không header, bỏ dòng đầu nếu đó là header. Cột ảnh là cột có nhiều giá trị mang đuôi ảnh nhất (phải đạt ít nhất 90%). Cột text là cột còn lại có độ dài trung bình lớn nhất. Nhãn luôn lấy từ thư mục ảnh, không dùng cột nhãn trong CSV. Split lấy từ tên file CSV. CSV không nhận diện được sẽ bị bỏ qua kèm cảnh báo.
- **Text dạng `.txt`:** nếu không có CSV, ghép mỗi ảnh với file `.txt` cùng tên gốc.
- Ghép ảnh với text theo `(filename, split)`, nếu không có split thì ghép theo `filename`.
- Nếu không nhận diện được định dạng, script dừng và in cấu trúc thư mục cùng vài dòng mẫu.
- Bỏ mẫu thiếu ảnh hoặc có text rỗng. Báo số lượng bị bỏ. Dừng nếu tỷ lệ bỏ vượt `data.max_drop_ratio` (mặc định 0.05).
- Tùy chọn `data.subset_classes` (lấy N lớp đầu theo thứ tự tên) và `data.max_per_class` để chạy thử.

**Đầu ra:** `manifest.csv` với các cột `id, image_path, label, label_idx, split, text, text_exact, text_strict`, cùng `classes.json` và `stats.json`.

### 4.2 Làm sạch text

Bỏ thẻ HTML và URL, gộp khoảng trắng, chuyển chữ thường, cắt ở `data.max_chars` (mặc định 5000).

### 4.3 Che tên món

Thứ tự xử lý: làm sạch (4.2) trước, rồi mới che. Nhờ vậy token `[MASK]` giữ nguyên chữ hoa, và tokenizer của DistilBERT nhận nó là special token.

- **Biến thể của một cụm tên:** dạng gốc (`_` đổi thành khoảng trắng), bỏ `s` cuối, thêm `s` cuối.
- **`exact`:** thay mọi biến thể tên của **tất cả** 101 lớp bằng `[MASK]`. Regex dùng word boundary, sắp theo độ dài giảm dần để cụm dài được che trước.
- **`strict`:** giống `exact`, và che thêm từng từ lẻ (có biến thể số ít/số nhiều) xuất hiện trong bất kỳ tên lớp nào. Bỏ qua từ nối (`and, with, de, the, a, of, in, la, le, au, en`) và từ ngắn hơn 3 ký tự.
- **Đo leakage:** với mỗi chế độ, sau khi che, tính 4 chỉ số:
  - tỷ lệ text còn chứa cụm tên lớp đúng (`own_label_rate`)
  - tỷ lệ text còn chứa ít nhất một từ lẻ trong tên lớp đúng (`own_word_rate`, ví dụ "carbonara" của `spaghetti_carbonara`)
  - tỷ lệ text chứa cụm tên của lớp khác (`other_label_rate`)
  - số token trung bình còn lại (`mean_tokens`)

  `own_word_rate` cho thấy phần leakage mà chế độ `exact` còn bỏ sót.

### 4.4 Dataset và DataLoader

- `FoodDataset(df, data_root, text_col, use_image, use_text, transform)` trả về dict gồm `idx, label, image?, text?`.
- **Transform khi train:** RandomResizedCrop(224, scale 0.6–1), lật ngang, ColorJitter(0.2). **Khi eval:** Resize(256), CenterCrop(224). Mean/std lấy từ pretrained config của timm.
- `Collator` tokenize theo batch (dynamic padding, truncation `text.max_len` = 256).
- Ảnh lỗi được thay bằng tensor 0 và đếm lại. Dừng nếu số ảnh lỗi vượt `data.max_bad_images` (mặc định 50).

## 5. Mô hình và huấn luyện

### 5.1 Mô hình

- **ImageClassifier:** `timm.create_model(backbone, pretrained, num_classes=0)`, qua dropout 0.2, rồi Linear. Có `forward_features`.
- **TextClassifier:** `AutoModel`, mean pooling có mask, qua dropout 0.1, rồi Linear. Có `forward_features`.
- **MultimodalClassifier:**
  - Mỗi nhánh: LayerNorm, Linear về 512, GELU.
  - Nối 2 nhánh (1024 chiều), qua Dropout 0.3, Linear 512, GELU, Dropout 0.3, Linear 101.
  - Modality dropout khi train: mỗi mẫu có xác suất `fusion.modality_dropout` (mặc định 0.1) bị zero một nhánh, nhánh được chọn ngẫu nhiên đều giữa ảnh và text. Không bao giờ zero cả hai nhánh. Tắt khi eval.
  - Encoder được nạp từ `best.pt` của run đơn modality. Head đơn modality bị bỏ, không đưa vào optimizer.
- **Late fusion:** softmax logits của 2 run. Dò `w ∈ {0, 0.05, …, 1}` để tối đa accuracy trên val, rồi áp dụng lên test. Kiểm tra 2 file có cùng thứ tự `ids` trước khi kết hợp.
- **TF-IDF + LR:** `TfidfVectorizer(ngram_range=(1,2), max_features=200000, sublinear_tf=True, min_df=2)` với `LogisticRegression(max_iter=1000)`. Chọn C từ {1, 5, 20} trên val.

### 5.2 Huấn luyện chung (`engine.py`)

- AdamW với 2 nhóm tham số (encoder và head). Cosine schedule, warmup 5%.
- AMP fp16 khi có CUDA. Grad clip 1.0. Label smoothing 0.1.
- Early stopping theo val accuracy, patience 2.
- Sau khi train: nạp `best.pt`, dự đoán trên val và test, lưu `preds_*.npz` và `metrics_test.json`.
- Run đã hoàn tất (có `metrics_test.json`) sẽ bị bỏ qua, trừ khi có `--force`.

### 5.3 Siêu tham số mặc định

| | Epoch | Batch | LR encoder | LR head | Weight decay |
|---|---|---|---|---|---|
| Image | 10 | 64 | 1e-4 | 1e-3 | 0.05 |
| Text | 4 | 32 | 3e-5 | 1e-3 | 0.01 |
| Early fusion | 4 | 32 | 3e-5 (ảnh) / 2e-5 (text) | 1e-3 | 0.01 |

### 5.4 Danh sách run

| Run | Lệnh (rút gọn) |
|---|---|
| `tfidf_none`, `tfidf_strict` | `train_tfidf.py --set data.text_mask=...` |
| `image` | `train.py --modality image` |
| `text_none`, `text_strict` | `train.py --modality text --set data.text_mask=...` |
| `late_none`, `late_strict` | `late_fusion.py --image_run image --text_run text_...` |
| `early_none`, `early_strict` | `train.py --modality multimodal --set data.text_mask=... fusion.image_init=image fusion.text_init=text_...` |

Thời gian chạy chưa đo thực tế. Sẽ đo ở lần chạy thử với `subset_classes=5`.

## 6. Đánh giá và phân tích

- `metrics.py`: accuracy, top-5, macro-F1, báo cáo theo lớp, confusion matrix.
- `summarize.py`: gom `runs/*/metrics_test.json` thành `results/summary.csv` và `summary.md`. Các cột: run, modality, text_mask, acc, top5, macro_f1.
- Notebook phân tích gồm:
  - Biểu đồ cột 9 run.
  - Đường cong val accuracy theo w của late fusion.
  - Confusion matrix và top 10 cặp nhầm của image và early fusion.
  - Accuracy theo lớp: lớp nào được lợi nhất khi thêm text.
  - Lưới ví dụ image-only sai / fusion đúng và ngược lại, kèm text và dự đoán.

## 7. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Không nhận diện được định dạng text | Dừng, in cấu trúc thư mục và mẫu |
| Bỏ quá `max_drop_ratio` mẫu | Dừng, in thống kê |
| Ảnh lỗi | Thay bằng tensor 0, đếm lại, dừng nếu vượt `max_bad_images` |
| Thiếu checkpoint đơn modality hoặc sai chế độ text khi train fusion | Dừng với thông báo chỉ rõ run cần chạy trước |
| Late fusion: `ids` không khớp giữa 2 run | Dừng |
| Session Colab bị ngắt | Chạy lại cùng lệnh, resume từ `last.pt` |

## 8. Kiểm thử (pytest, CPU)

- `test_prepare.py`: dataset giả 3 lớp, cả 2 định dạng (CSV và `.txt`). Kiểm tra nhận diện đúng, tách split, tách val stratified, đủ các cột.
- `test_text_utils.py`:
  - Che số ít/số nhiều, cụm nhiều từ, không che từ con (ví dụ "pie" trong "piece").
  - `strict` che từ lẻ nhưng giữ từ nối.
  - Chỉ số leakage đúng trên ví dụ nhỏ.
- `test_pipeline_smoke.py`:
  - 1 epoch với `resnet18` (không pretrained) và một DistilBERT tí hon (`hf-internal-testing/tiny-random-DistilBertModel`) cho image, text, multimodal.
  - Kiểm tra các file đầu ra, rồi chạy `late_fusion` và `summarize`.
- Notebook có cell chạy thử với `subset_classes=5` trước khi chạy thật.

## 9. Phụ thuộc

`requirements-colab.txt` chỉ pin các gói có thể cần cài thêm. Phiên bản được kiểm tra trên PyPI ngày 2026-09-27:

```
timm==1.0.30
transformers==5.17.0
```

`torch`, `torchvision`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`, `kaggle` dùng bản có sẵn trên Colab.

## 10. Rủi ro và điểm chưa xác minh

- **Định dạng text của bản Kaggle chưa xác minh.** Loader tự nhận diện và có bước in cấu trúc thư mục. Nếu định dạng khác dự kiến, chỉ cần sửa `prepare.py`.
- **Tương thích `transformers==5.17.0` với bản torch có sẵn trên Colab chưa kiểm tra.** Nếu xung đột, bỏ pin và dùng bản có sẵn. Code chỉ dùng API ổn định (`AutoModel`, `AutoTokenizer`).
- **Thời gian train chỉ là ước lượng.** Đọc ảnh có thể là nút thắt, xử lý bằng cách tăng `num_workers`. Nếu cần, thêm bước resize ảnh trước khi train (ngoài phạm vi hiện tại).
- **`strict` có thể che quá tay** (ví dụ "chicken" xuất hiện trong mô tả của nhiều món). Đây là chủ đích để tạo cận dưới của leakage, và sẽ được nêu rõ khi trình bày.
