# Full Pipeline — AIC-HCMC 2026 Retrieval System

Tài liệu này mô tả toàn diện kiến trúc hệ thống tìm kiếm truy vấn video phục vụ cuộc thi **AI Challenge TP.HCM 2026 (AIC-HCMC-26)**, kèm theo **đầy đủ đường dẫn file mã nguồn (code files)** tương ứng với từng giai đoạn để tiện theo dõi, đọc hiểu và review mã nguồn.

---

## 1. Tổng quan kiến trúc & Cấu hình tập trung

- **File cấu hình trung tâm:** [`src/config.py`]
  - Quản lý toàn bộ đường dẫn (`DATA_DIR`, `INDEX_DIR`, `KEYFRAMES_DIR`, `METADATA_PATH`).
  - Thiết lập model (`ViT-SO400M-14-SigLIP`, `dim=1152`).
  - Bật/tắt LLM Query Expansion (`ENABLE_LLM_EXPANSION = True`, `LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"`).
  - Cấu hình server chấm DRES (`DRES_BASE_URL`, `DRES_SESSION_ID`, `DRES_EVALUATION_ID`).
- **Backend API & WebSockets:** [`src/server.py`]
  - Lớp `SearchService` nạp model SigLIP, chỉ mục FAISS, metadata Polars và LLM Query Expander.
  - Cung cấp các endpoint REST và WebSocket theo thời gian thực.
- **Frontend giao diện Web:** [`baseline/retrievalBaseline/frontend/`]
  - Giao diện trực quan cho thí sinh: tìm kiếm Text, tìm kiếm Ảnh, phân trang 100 frame/trang, ghim kết quả, xem video MP4 gốc và nộp bài DRES 1-click.

---

## 2. Phase 1: Offline Ingestion (Xử lý tiền kỳ một lần)

Giai đoạn này xử lý toàn bộ dữ liệu thô (video MP4, keyframe, file CSV) để tạo ra các bảng chỉ mục tốc độ cao.

```
[Raw Keyframes L01–L24] ──► [extract_siglip_features_gpu.py] ──► [data/siglip-features-so400m/*.npy]
                                                                                │
                                                                                ▼
[CSV files: Map-Keyframes] ──► [metadata_loader.py] ──► [indexes/metadata.parquet]
                                                                │
[Feature .npy files]        ──► [build_index.py]     ──► [indexes/faiss_siglip.index]
```

### 2.1 — Master Metadata Table
- **File thực thi:** [`src/offline_pipeline/metadata_loader.py`]
- **Output:** [`indexes/metadata.parquet`] (734.544 dòng, 726 videos `L01`–`L24`)
- **Vai trò:** Đọc toàn bộ các file CSV mapping của ban tổ chức, nối thành 1 bảng Parquet nén Snappy. Cung cấp ánh xạ chuẩn giữa `keyframe_name` (file ảnh), `pts_time` (mốc giây), `faiss_idx` (vị trí vector) và `frame_id` (**số thứ tự frame gốc MP4 cần nộp thi**).

### 2.2 — Trích xuất đặc trưng SigLIP-SO400M
- **File thực thi:** [`scripts/extract_siglip_features_gpu.py`]
- **Output:** Thư mục vector `data/siglip-features-so400m/` (mỗi video 1 file `.npy`, float32 shape `[N_frames, 1152]`).
- **Mô hình:** `ViT-SO400M-14-SigLIP` (pretrained `webli` qua thư viện `open_clip`).
- **Server trích xuất keyframe gốc:** [`scripts/extract_keyframes_server.py`](chạy trên server Ubuntu giải mã ffmpeg).

### 2.3 — Xây dựng chỉ mục FAISS
- **File thực thi:** [`src/offline_pipeline/build_index.py`]
- **Output:** [`indexes/faiss_siglip.index`] (Dung lượng 3.38 GB, 734.544 vectors 1152 chiều).
- **Cơ chế:** Nạp tất cả file `.npy`, chuẩn hóa $L_2$ (`normalize=True`), đưa vào `faiss.IndexFlatIP` cho phép tìm kiếm inner product (tương đương Cosine Similarity) với thời gian < 50ms trên CPU RAM.

---

## 3. Phase 2: Online Retrieval Pipeline (Quy trình tìm kiếm trực tuyến)

```
[Truy vấn Tiếng Việt]
         │
         ▼
[src/online_pipeline/query_expander.py] ──► Sửa lỗi đánh máy & sinh 3 prompt English
         │
         ▼
[src/server.py: encode_and_expand_query] ──► Mã hóa SigLIP & Weighted Blending (0.5*v1 + 0.25*v2 + 0.25*v3)
         │
         ├──[Có từ khóa thời gian?] ──► [src/online_pipeline/temporal_search.py]
         │
         ▼
[src/online_pipeline/retrieval.py] ──────► FAISS IndexFlatIP Search (Top-K)
         │
         ▼
[Join Polars Metadata] ──────────────────► Trả về URL ảnh, video_id, frame_id, score
         │
         ▼
[baseline/retrievalBaseline/frontend/] ──► Hiển thị lưới ảnh + Nút nộp DRES 1-click
```

### 3.1 — Multi-Query Expansion & Typo Correction
- **File code:** [`src/online_pipeline/query_expander.py`] (Class `QueryExpander`)
- **Model:** `Qwen/Qwen2.5-3B-Instruct` chạy ở chế độ **4-bit quantization** qua `bitsandbytes` (~ 2 GB VRAM).
- **Chức năng:**
  1. Nhận diện và sửa triệt để các lỗi gõ Telex, lỗi dấu thanh tiếng Việt (ví dụ: `tay đưa xe đạp` $\rightarrow$ `tay đua xe đạp`).
  2. Phân rã ngữ nghĩa câu hỏi thành 3 góc nhìn trực quan chuyên biệt cho SigLIP:
     - Dòng 1 (*Scene*): Mô tả tổng thể bối cảnh, ánh sáng, trang phục, góc nhìn.
     - Dòng 2 (*Action*): Tập trung vào chủ thể chính và hành động đang diễn ra.
     - Dòng 3 (*Tags*): Danh sách từ khóa các vật thể và thuộc tính quan trọng.
- **Fallback nhanh:** [`src/utils/translation.py`] (Google GTX endpoint có in-memory LRU cache để dự phòng khi không dùng LLM).

### 3.2 — Weighted Embedding Blending
- **File code:** [`src/server.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/src/server.py) (Hàm `encode_and_expand_query`)
- **Cấu hình trọng số:** Được khai báo tập trung trong [`src/config.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/src/config.py) (`EXPANSION_WEIGHTS = [0.5, 0.25, 0.25]`, có thể override qua biến môi trường `AIC_EXPANSION_WEIGHTS`).
- **Công thức:**
  $$\vec{v}_{\text{query}} = \text{Normalize}\Big(w_1 \cdot \vec{v}_{\text{scene}} + w_2 \cdot \vec{v}_{\text{action}} + w_3 \cdot \vec{v}_{\text{tags}}\Big)$$
- Giúp vector truy vấn có đầy đủ cả thông tin bối cảnh lớn lẫn chi tiết hành động nhỏ của vật thể.

### 3.3 — Tìm kiếm thời gian (Temporal Chronological Search)
- **File code:** [`src/online_pipeline/temporal_search.py`]
- **Chức năng:** Phát hiện các liên từ thời gian (`"sau đó"`, `"tiếp theo"`, `"then"`), tự động phân tách thành chuỗi sự kiện con, truy vấn độc lập và chấm điểm ưu tiên cho các video có thứ tự khung hình tăng dần theo thời gian: $\text{frame\_id}(E_1) < \text{frame\_id}(E_2)$.

### 3.4 — Truy xuất FAISS & Metadata Join
- **File code:** [`src/online_pipeline/retrieval.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/src/online_pipeline/retrieval.py) (Class `RetrievalEngine`)
- **Chức năng:** Nhận vector truy vấn đã chuẩn hóa, gọi `faiss_index.search(q_vec, top_k)` và ghép dữ liệu trực tiếp với bảng `metadata.parquet` để trả về danh sách đầy đủ thông tin khung hình.

### 3.5 — Tra cứu OCR & Cộng điểm thưởng (Soft Boost)
- **File code:** [`src/online_pipeline/text_search.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/src/online_pipeline/text_search.py) (Class `TextSearchEngine`)
- **Nguyên lý Chống ảo giác (Anti-Hallucination):** Dùng luật Deterministic Regex trích xuất con số (`115`), biển số xe (`51F`), từ trong ngoặc kép (`"..."`) hoặc từ khóa có tiền tố (`biển số`, `số áo`). Nếu query chỉ là miêu tả hình ảnh bình thường, OCR tự động tắt 100%.
- **Cơ chế Soft Boost:** Tra cứu trong [`indexes/ocr_text.parquet`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/indexes/) và cộng `+0.35` điểm thưởng cho các khung hình chứa đúng chữ/số, đẩy chúng lên Top 1 mà không gạt bỏ các ứng viên thị giác khác của FAISS.

### 3.6 — VLM Reranking Top-20 (Qwen2-VL-7B 4-bit)
- **File code:** [`src/online_pipeline/vlm_reranker.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/src/online_pipeline/vlm_reranker.py) (Class `VLMReranker`)
- **Chức năng:** So khớp thị giác đa phương thức sâu sắc giữa câu query gốc và 20 khung hình tiềm năng nhất.
- **Công thức:**
  $$\text{Final\_Score} = 0.65 \cdot \text{VLM\_Score} + 0.35 \cdot \text{Initial\_Score}$$
- **Kích hoạt linh hoạt:** Thí sinh có thể bấm nút **`⚡ Rerank AI (Top 20)`** trên Web UI để AI chấm điểm và đảo ngược thứ tự đẩy frame đúng lên Top 1–3 trong ~1.5 giây.

---

## 4. Chi tiết 4 đầu Task thi đấu & File mã nguồn tương ứng

### 4.1 — KIS-T (Textual Known-Item Search)
- **File Backend:** [`src/tasks/kis_t.py`] và [`src/server.py`] (Endpoint `/TextQuery` và WebSocket `/ws/group_search`)
- **File Frontend:** [`baseline/retrievalBaseline/frontend/src/scripts/script.js`]
- **Quy trình:** Nhập mô tả tiếng Việt $\rightarrow$ `QueryExpander` $\rightarrow$ SigLIP $\rightarrow$ FAISS $\rightarrow$ Hiển thị kết quả $\rightarrow$ Nộp bài theo cú pháp `<video_id>, <frame_id>`.

---

### 4.2 — KIS-V (Visual Known-Item Search)
- **File Backend:** [`src/server.py`] (Endpoint `/similarity_search` và `/mixed_query`)
- **File Frontend:** [`baseline/retrievalBaseline/frontend/src/scripts/script.js`](Hàm `searchSimilarImage`)
- **Quy trình:**
  1. Thí sinh chụp màn hình (`Win + Shift + S`) 1 frame rõ nhất từ video 5s của đề thi.
  2. Kéo thả file ảnh vào ô tìm kiếm ảnh hoặc bấm nút **"⚡ Tương đồng"** trên bất kỳ khung hình nào.
  3. Backend mã hóa ảnh qua SigLIP Image Encoder $\rightarrow$ tìm trong FAISS $\rightarrow$ Trả về khung hình gốc trùng khớp ở Top 1 với độ tương đồng $> 0.85$.
- *Định hướng nâng cấp (Hybrid Multi-Modal KIS-V):* Kết hợp thêm 1 ô text mô tả hành động phụ trợ để giải quyết các cảnh đường sá hoặc đám đông bị trùng lặp.

---

### 4.3 — KIS-QA (Visual Question Answering)
- **File Backend:** [`src/tasks/vqa.py`] (Tích hợp sinh câu trả lời tự động bằng `Qwen2-VL-7B-Instruct`)
- **File Frontend:** [`baseline/retrievalBaseline/frontend/src/scripts/export.js`]
- **Quy trình:**
  1. Thí sinh dùng thanh tìm kiếm để tìm phân cảnh có chứa đáp án câu hỏi.
  2. Bấm nút **Ghim (Pin)** khung hình vào khu vực làm bài (Export Area).
  3. Nhập trực tiếp text câu trả lời vào ô `Đáp án QA` trên card của khung hình đó.
  4. Hệ thống tự động tính `pts_time * 1000` thành miligiây và tạo chuỗi nộp chuẩn BTC:
     ```
     QA-<ANSWER>-<VIDEO_ID>-<TIME_MS>
     ```
  5. Bấm nộp trực tiếp sang DRES bằng 1 click.

---

### 4.4 — TRAKE (Temporal Retrieval of Anchoring Keyframes)
- **File Backend:** [`src/tasks/trake.py`]
- **File Frontend:** [`baseline/retrievalBaseline/frontend/src/scripts/export.js`]
- **Quy trình:**
  1. Ghim các khung hình tương ứng với các sự kiện liên tiếp của đề bài vào khu vực Export.
  2. Hệ thống kiểm tra thứ tự thời gian tăng dần (`frame_id_1 < frame_id_2 < ...`).
  3. Tự động sinh chuỗi nộp chuẩn TRAKE:
     ```
     TR-<VIDEO_ID>-<FRAME_1>-<FRAME_2>-...
     ```
  4. Bấm nộp trực tiếp sang DRES.

---

## 5. DRES Submission Pipeline

- **File quản lý kết nối DRES trên Frontend:** [`baseline/retrievalBaseline/frontend/src/scripts/submit_dres.js`]
  - Quản lý cấu hình Session ID, Server URL (`https://aiclub.uit.edu.vn/dres` hoặc local `http://192.168.20.150:8080`).
  - Gửi request nộp bài qua DRES API v2, bắt mã lỗi HTTP và hiển thị thông báo alert trạng thái nộp bài trực quan.
- **Script kiểm tra & nộp thử CLI:** [`baseline/retrievalBaseline/dres_submit_demo.py`]
- **Script kiểm tra định dạng file nộp tổng hợp:** [`scripts/validate_submission.py`]

---

## 6. Bảng tra cứu toàn bộ file mã nguồn (Source Code Map)

Dưới đây là bảng tổng hợp đầy đủ các file code trong dự án để bạn tiện tra cứu và review:

| Thành phần | Đường dẫn file | Vai trò chính |
|---|---|---|
| **Cấu hình** | [`src/config.py`] | Định nghĩa toàn bộ đường dẫn index, data, model ID, session DRES |
| **Backend Server** | [`src/server.py`] | FastAPI & WebSocket server, nạp `SearchService`, điều phối toàn bộ luồng tìm kiếm |
| **Giao diện Web** | [`baseline/retrievalBaseline/frontend/index.html`] | Trang web chính phục vụ thao tác thi đấu thời gian thực |
| **Logic Frontend** | [`baseline/retrievalBaseline/frontend/src/scripts/script.js`] | Điều khiển giao diện, gửi query Text/Ảnh qua WebSocket & REST |
| **Xử lý Ghim & Nộp** | [`baseline/retrievalBaseline/frontend/src/scripts/export.js`] | Quản lý khung hình ghim, định dạng chuỗi nộp KIS-QA và TRAKE |
| **Client DRES** | [`baseline/retrievalBaseline/frontend/src/scripts/submit_dres.js`] | Quản lý phiên DRES, gửi lệnh nộp bài trực tiếp đến server chấm thi |
| **Mở rộng truy vấn** | [`src/online_pipeline/query_expander.py`] | Sửa lỗi chính tả tiếng Việt & sinh 3 prompt trực quan bằng Qwen2.5-3B 4-bit |
| **Dịch thuật fallback** | [`src/utils/translation.py`] | Dịch nhanh qua Google GTX có cache, dùng khi không bật LLM |
| **Mã hóa truy vấn** | [`src/online_pipeline/query_encoder.py`] | Chuyển đổi text/image thành vector đặc trưng SigLIP-SO400M (1152 chiều) |
| **Tìm kiếm FAISS** | [`src/online_pipeline/retrieval.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/src/online_pipeline/retrieval.py) | Quét vector trên chỉ mục FAISS và ghép metadata tương ứng |
| **Tra cứu OCR** | [`src/online_pipeline/text_search.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/src/online_pipeline/text_search.py) | Trích xuất số/mã hiệu bằng luật cứng & cộng Soft Boost (+0.35) |
| **VLM Reranker** | [`src/online_pipeline/vlm_reranker.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/src/online_pipeline/vlm_reranker.py) | Chấm điểm xác thực thị giác Top 20 bằng Qwen2-VL-7B 4-bit |
| **Trích xuất OCR** | [`scripts/extract_ocr_easyocr.py`](file:///d:/STUDY/BIN/CS/PROJECT/AIC-HCMC-26/scripts/extract_ocr_easyocr.py) | Trích xuất text subsampled (stride 4s) từ keyframe GPU |
| **Xử lý chuỗi thời gian** | [`src/online_pipeline/temporal_search.py`] | Phân tách truy vấn đa sự kiện và lọc kết quả theo thứ tự xuất hiện |
| **Task KIS-T** | [`src/tasks/kis_t.py`] | Logic điều phối độc lập cho bài thi Textual KIS |
| **Task KIS-QA** | [`src/tasks/vqa.py`] | Logic điều phối cho bài thi hỏi đáp KIS-QA với VLM Qwen2-VL-7B |
| **Task TRAKE** | [`src/tasks/trake.py`] | Logic điều phối cho bài thi chuỗi sự kiện tuần tự TRAKE |
| **Tạo Metadata** | [`src/offline_pipeline/metadata_loader.py`] | Chuyển đổi 726 file CSV mapping thành file `metadata.parquet` |
| **Tạo FAISS Index** | [`src/offline_pipeline/build_index.py`] | Ghép các file vector `.npy` thành file `faiss_siglip.index` |
| **Trích xuất SigLIP GPU** | [`scripts/extract_siglip_features_gpu.py`] | Script trích xuất vector đặc trưng SigLIP từ keyframes trên GPU |
| **Trích xuất Keyframe** | [`scripts/extract_keyframes_server.py`] | Script giải mã video MP4 thành các file ảnh keyframe bằng ffmpeg |
| **Giao diện Streamlit** | [`app.py`] | Giao diện phụ trợ chạy bằng Streamlit để test nhanh |
