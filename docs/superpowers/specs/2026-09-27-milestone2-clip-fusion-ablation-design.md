# Mốc 2: Embedding CLIP, các kiểu fusion nâng cao và ablation

- Ngày: 2026-09-27
- Trạng thái: đã duyệt thiết kế, chờ duyệt spec
- Phạm vi: sub-project 2 trong 4. Dùng lại manifest, config, metrics, late fusion và analysis của Mốc 1.

## 1. Mục tiêu

Dùng CLIP ViT-B/16 đóng băng làm encoder chung cho ảnh và text, trích xuất đặc trưng một lần, rồi huấn luyện nhiều head fusion nhẹ trên đặc trưng đã lưu. Nhờ vậy có thể chạy nhiều ablation với chi phí thấp, và trả lời:

1. Kiểu fusion nào tốt nhất khi encoder giống nhau (late, concat, gated, cross-attention)?
2. Fusion chịu được thiếu modality và nhiễu tới mức nào? Modality dropout có giúp không?
3. Khi ít dữ liệu train, multimodal có lợi hơn đơn modality nhiều hơn không?
4. Leakage tên món trong text ảnh hưởng thế nào (3 chế độ che)?

### Tiêu chí hoàn thành

1. Baseline zero-shot CLIP trên test (prompt `a photo of {món}, a type of food`).
2. Bảng test (Acc, Top-5, Macro-F1) cho 6 head (`image`, `text`, `late`, `concat`, `gated`, `xattn`) × 3 chế độ che (`image` chỉ có 1 run), với modality dropout mặc định.
3. Bảng thiếu modality: mỗi head fusion, chế độ che chính, huấn luyện với modality dropout 0 và 0.3, đánh giá khi đủ / thiếu ảnh / thiếu text.
4. Bảng và biểu đồ nhiễu: blur, gaussian noise (ảnh) và word drop (text), mỗi loại 3 mức, trên test.
5. Đường cong theo tỷ lệ dữ liệu train 10/25/50/100% cho `image`, `text`, `concat`, `xattn`.
6. t-SNE đặc trưng ảnh, text và đặc trưng fusion trên một mẫu test.
7. Pytest trên CPU với CLIP tí hon (`hf-internal-testing/tiny-random-CLIPModel`) và pass.

### Ngoài phạm vi

- Fine-tune CLIP. Fusion bilinear. VLM (sub-project 3), demo (sub-project 4).

## 2. Quyết định đã chốt

| Chủ đề | Quyết định |
|---|---|
| Encoder | `openai/clip-vit-base-patch16` qua `transformers.CLIPModel`, đóng băng, fp16 trên GPU |
| Không dùng `open_clip` | Tránh thêm phụ thuộc; `transformers` đã được pin ở Mốc 1 |
| Đặc trưng lưu | Vector pooled đã chiếu (512, chuẩn hóa L2) và 16 token cho mỗi modality |
| Token ảnh | Lưới patch 14×14 (bỏ CLS) → adaptive average pool 4×4 = 16 token, 768 chiều |
| Token text | Token hợp lệ (tối đa 77) chia đều thành 16 đoạn liên tiếp, mean-pool mỗi đoạn, 512 chiều, kèm mask 16 phần tử (đoạn rỗng = 0) |
| Lưu trữ | Thư mục `.npy` fp16 trên Drive, đọc bằng mmap. Ước tính khoảng 4 GB cho mỗi chế độ che, dư chỗ với 5 TB |
| Chế độ che chính cho ablation | `strict` (đặt trong config `clip.main_mask`) |
| Head | Huấn luyện trên đặc trưng đã lưu, vài phút mỗi run |

## 3. Kiến trúc

### Module mới

```
src/foodmm/clip/
├── __init__.py
├── encoder.py      # ClipEncoder: nạp CLIPModel + processor, encode_images, encode_texts, encode_prompts
├── features.py     # lưu/đọc feature store (shard .npy), căn theo ids của manifest
├── corrupt.py      # blur, gaussian noise cho ảnh; word drop cho text (có seed)
├── heads.py        # ImageHead, TextHead, ConcatHead, GatedHead, CrossAttnHead, build_head
├── train_heads.py  # FeatureDataset, fit_head, predict_head, đánh giá thiếu modality và nhiễu
└── zero_shot.py    # logits zero-shot từ embedding ảnh và prompt
scripts/
├── extract_clip.py     # trích xuất đặc trưng (resume theo shard)
├── zero_shot_clip.py
├── train_head.py       # một run head
├── run_clip_suite.py   # chạy toàn bộ run chính và ablation, bỏ qua run đã xong
└── summarize_clip.py   # gom bảng, vẽ hình
notebooks/02_milestone2.ipynb   # sinh bởi tools/build_notebook.py
```

`src/foodmm/late_fusion.py` và `scripts/late_fusion.py` của Mốc 1 được dùng lại. Script có thêm tham số `--name` để đặt tên run đầu ra.

Notebook của Mốc 2, 3, 4 dùng chung `tools/nb_utils.py` (lớp `NotebookBuilder` và các cell setup chung: mount Drive, clone repo, cài gói, hàm `run`, giải nén dữ liệu). `tools/build_notebook.py` của Mốc 1 giữ nguyên.

### Lưu trữ trên Drive

```
MyDrive/foodmm/clip/
├── features/vitb16/<tên>/       # ids.npy, pooled.npy, tokens.npy, token_mask.npy (text), done.json
│   ├── image_{train,val,test}
│   ├── text_{none,exact,strict}_{train,val,test}
│   ├── image_test_{blur1,blur2,blur4,noise0.05,noise0.1,noise0.2}
│   └── text_strict_test_{drop0.25,drop0.5,drop0.75}
├── runs/<run>/                  # config.yaml, best.pt, history.json, preds_{val,test}.npz,
│                                # preds_robust.npz, metrics_test.json, metrics_robust.json
└── results/                     # bảng .csv/.md và hình .png
```

Run của Mốc 2 nằm trong `clip/runs/`, tách khỏi `runs/` của Mốc 1, nên `summarize.py` của Mốc 1 không bị lẫn.

## 4. Trích xuất đặc trưng (`extract_clip.py`)

- Đọc manifest của Mốc 1 (chạy `prepare_data.py` trước). Ảnh đọc từ `paths.data_root`, text lấy theo cột của chế độ che.
- `ClipEncoder.encode_images(list[PIL]) -> (pooled[N,512], tokens[N,16,768])`: `image_embeds` đã chuẩn hóa L2; token là `last_hidden_state[:, 1:]` reshape về lưới rồi adaptive average pool 4×4.
- `ClipEncoder.encode_texts(list[str]) -> (pooled[N,512], tokens[N,16,512], mask[N,16])`: truncation 77; `text_embeds` chuẩn hóa L2; token từ `last_hidden_state` theo cách chia đoạn ở mục 2.
- `[MASK]` không phải token đặc biệt của CLIP, tokenizer tách thành vài mảnh. Điều này chấp nhận được vì mọi chế độ che đều được xử lý như nhau.
- Ghi theo shard (`clip.shard_size`, mặc định 2048 mẫu). Chạy lại thì bỏ qua shard đã có. Khi đủ shard thì ghép thành `.npy` và ghi `done.json`. Feature set đã có `done.json` thì bỏ qua, trừ khi có `--force`.
- Tham số: `--parts image,text_none,text_exact,text_strict,corrupt` và `--splits train,val,test`. `corrupt` chỉ chạy trên test, với các mức trong `clip.corruptions`.
- **Nhiễu ảnh** áp lên ảnh PIL gốc trước processor: Gaussian blur bán kính {1, 2, 4}; gaussian noise độ lệch chuẩn {0.05, 0.1, 0.2} trên thang [0, 1].
- **Nhiễu text:** bỏ mỗi từ với xác suất {0.25, 0.5, 0.75}, áp lên cột text của `clip.main_mask`.
- Seed của mỗi mẫu là `seed + crc32(id)`, nên kết quả tái lập được.

## 5. Zero-shot (`zero_shot_clip.py`)

Prompt `clip.prompt` (mặc định `a photo of {}, a type of food`), `{}` là tên lớp với `_` đổi thành khoảng trắng. Logits = 100 · cos(image_embeds, prompt_embeds). Kết quả lưu vào run `zeroshot` gồm `preds_test.npz` và `metrics_test.json`, cùng định dạng với các run khác.

## 6. Head fusion (`foodmm/clip/heads.py`)

Mọi head nhận batch dict `img` [B,512], `txt` [B,512], và nếu cần thì `img_tok` [B,16,768], `txt_tok` [B,16,512], `txt_mask` [B,16]. Head trả về `(logits, features)`, trong đó `features` là vector ngay trước lớp phân loại cuối.

| Head | Cấu trúc |
|---|---|
| `image`, `text` | LayerNorm, Dropout, Linear→hidden, GELU, Dropout, Linear→C |
| `concat` | Mỗi nhánh LayerNorm, Linear→hidden, GELU; nối lại; Dropout, Linear→hidden, GELU, Dropout, Linear→C |
| `gated` | Chiếu 2 nhánh như `concat`; g = sigmoid(Linear([h_i; h_t])); h = g·h_i + (1−g)·h_t; Dropout, Linear→C. Lưu g trung bình để phân tích |
| `xattn` | Chiếu token ảnh và text về `head.xattn_dim` (256), cộng embedding vị trí. `head.xattn_layers` (1) khối: text token là query, image token là key/value, residual, LayerNorm, FFN. Mean-pool có mask trên text token → z. Nối [z, h_i, h_t], rồi MLP→C |
| `late` | Không huấn luyện. Dùng `search_weight` / `combine` của Mốc 1 trên xác suất của run `image` và `text_<mask>` |

**Thiếu modality:** một hàm dùng chung `drop_modalities(batch, drop_img, drop_txt)` đặt về 0 vector pooled và token của modality bị bỏ, và đặt `txt_mask` = 0 nếu bỏ text. Khi train, mỗi mẫu có xác suất `head.modality_dropout` bị bỏ một modality, chọn ngẫu nhiên đều, không bao giờ bỏ cả hai. Khi eval, cùng hàm đó được dùng để tạo điều kiện thiếu ảnh / thiếu text. Nếu mask text toàn 0 thì mean-pool trả về 0.

## 7. Huấn luyện head (`train_head.py`, `foodmm/clip/train_heads.py`)

- Đặc trưng được nạp hết vào RAM ở dạng fp16 (khoảng 4 GB với token của train), mỗi batch mới chuyển sang fp32. Token chỉ được nạp khi head cần (`xattn`).
- AdamW (lr 1e-3, weight decay 0.01), cosine schedule, warmup 5%, label smoothing 0.1, batch 256, tối đa 30 epoch, early stopping theo val accuracy với patience 3.
- `head.train_frac < 1`: lấy mẫu stratified theo lớp từ tập train, seed cố định, mỗi lớp giữ ít nhất 1 mẫu.
- Tên run: `image`, `text_<mask>`, `<head>_<mask>`, thêm hậu tố `_md<p>` nếu modality dropout khác mặc định, và `_frac<f>` nếu `train_frac < 1`. Ví dụ: `xattn_strict_md0.3`, `concat_strict_frac0.1`.
- Sau khi train: nạp `best.pt`, lưu `preds_val.npz` và `preds_test.npz` (cùng khóa với Mốc 1: `logits, labels, ids, features`), `metrics_test.json` (khóa của Mốc 1, thêm `head, modality_dropout, train_frac, best_val_acc`, và `mean_gate` với `gated`).
- **Đánh giá độ bền** trên test: điều kiện `full`, `no_image`, `no_text` (bỏ điều kiện không áp dụng cho head đơn modality), và mọi feature set nhiễu đã có. Nhiễu ảnh đi với text sạch, nhiễu text đi với ảnh sạch. Kết quả ghi vào `metrics_robust.json` (danh sách `{condition, acc, top5, macro_f1}`) và `preds_robust.npz` (logits theo từng điều kiện).
- `late_<mask>`: w dò trên val sạch. Với từng điều kiện, kết hợp logits bền của `image` và `text_<mask>`. Khi thiếu ảnh chỉ dùng xác suất text, và ngược lại.
- Run đã có `metrics_test.json` sẽ bị bỏ qua, trừ khi có `--force`. Head train nhanh nên không cần resume giữa chừng.

## 8. Danh sách run (`run_clip_suite.py`)

`run_clip_suite.py --stage main|missing|frac|all` gọi các hàm trong process, bỏ qua run đã xong, và in bảng tiến độ.

| Nhóm | Run |
|---|---|
| Zero-shot | `zeroshot` |
| Chính (md 0.1) | `image`; `text_{none,exact,strict}`; `{concat,gated,xattn}_{none,exact,strict}`; `late_{none,exact,strict}` |
| Thiếu modality | `{concat,gated,xattn}_strict_md0` và `_md0.3`. Run md 0.1 lấy từ nhóm chính |
| Tỷ lệ dữ liệu | `{image,text_strict,concat_strict,xattn_strict}_frac{0.1,0.25,0.5}`. Tỷ lệ 1.0 lấy từ nhóm chính |

Tổng cộng khoảng 35 lần train head. Ước tính mỗi run mất vài phút trên GPU; con số thật sẽ đo trên Colab.

## 9. Tổng hợp và phân tích (`summarize_clip.py`, `foodmm/clip/report.py`)

- `main.csv/.md`: Acc, Top-5, Macro-F1 cho zero-shot và các run chính.
- `missing.csv/.md`: hàng là head × md, cột là `full / no_image / no_text`.
- `robust.csv` và `robust.png`: accuracy theo mức nhiễu, mỗi loại nhiễu một subplot, mỗi head một đường.
- `frac.csv` và `frac.png`: accuracy theo tỷ lệ dữ liệu train.
- Trong notebook: t-SNE (sklearn, perplexity 30) trên 20 lớp × tối đa 50 mẫu test, cho 3 loại đặc trưng: pooled ảnh, pooled text (`strict`) và `features` của `xattn_strict`. Có thêm so sánh Mốc 1 với Mốc 2 nếu `runs/` của Mốc 1 đã có kết quả.

## 10. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Chưa có manifest | Dừng, nhắc chạy `prepare_data.py` |
| Thiếu feature set mà head cần | Dừng, in lệnh `extract_clip.py` cần chạy |
| Ảnh lỗi khi trích xuất | Dùng ảnh đen, đếm lại, dừng nếu vượt `data.max_bad_images` |
| `ids` giữa các feature set không khớp manifest | Dừng |
| Late fusion thiếu run `image` hoặc `text_<mask>` | Dừng, chỉ rõ run cần chạy |
| Mất session khi trích xuất | Chạy lại, tiếp tục từ shard cuối |

## 11. Kiểm thử (pytest, CPU)

- `test_clip_heads.py`: shape đầu ra của mọi head; `drop_modalities` đúng; mean-pool khi mask toàn 0 trả về 0; modality dropout không bao giờ bỏ cả 2 nhánh.
- `test_clip_corrupt.py`: nhiễu tái lập theo seed, đúng mức, word drop giữ ít nhất 1 từ.
- `test_clip_features.py`: ghi shard, resume, ghép, đọc lại, và kiểm tra ids.
- `test_clip_encoder.py` (mark `network`): CLIP tí hon cho ra shape đúng, token text có mask.
- `test_clip_pipeline.py` (mark `network`): dataset giả, rồi `prepare_data` → `extract_clip` (tất cả các phần) → `zero_shot_clip` → `run_clip_suite --stage all` với cấu hình tí hon → `summarize_clip`, kiểm tra file và khóa đầu ra.
- `test_notebooks.py`: notebook 02 sinh ra đúng, các cell compile, các import `foodmm` chạy được.

## 12. Phụ thuộc

Không thêm gói mới. CLIP dùng `transformers==5.17.0`, t-SNE dùng `scikit-learn`.

## 13. Rủi ro

- CLIP chỉ thấy tối đa 77 token text, trong khi Mốc 1 cho DistilBERT thấy 256 token. So sánh text giữa 2 mốc cần nêu rõ điểm này.
- Head `xattn` với chỉ 16 token mỗi bên là phiên bản gọn. Nếu cần, có thể tăng số token bằng cách trích xuất lại (tham số `clip.n_tokens`).
- Tốc độ đọc ảnh có thể là nút thắt khi trích xuất. Dùng DataLoader với `data.num_workers` để giảm.
