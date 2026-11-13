# Full Pipeline — AIC-HCMC 2026 Retrieval System

Tài liệu này mô tả toàn diện kiến trúc hệ thống tìm kiếm truy vấn video phục vụ cuộc thi **AI Challenge TP.HCM 2026 (AIC-HCMC-26)**, được cập nhật theo phiên bản mới nhất:
- **Trích xuất đặc trưng**: Google SigLIP-SO400M (`ViT-SO400M-14-SigLIP`, 1152 chiều).
- **Mở rộng truy vấn (Multi-Query Expansion & Typo Correction)**: LLM `Qwen2.5-3B-Instruct` 4-bit chạy local.
- **Quy mô dữ liệu**: Toàn bộ Batch L (`L01`–`L24`), gồm 726 video với 734.544 keyframe.
- **Hỗ trợ đầy đủ 4 đầu task thi đấu**: KIS-T, KIS-V, KIS-QA, TRAKE.
- **Tích hợp nộp bài trực tiếp vào hệ thống DRES của BTC**.

---

## 1. Tổng quan kiến trúc (The Big Picture)

Hệ thống hoạt động dựa trên nguyên lý **không gian biểu diễn đa phương thức đồng nhất (Unified Multi-modal Embedding Space)** của mô hình Google SigLIP:
1. **Semantic Similarity (Độ tương đồng ngữ nghĩa hình ảnh - văn bản)**: Cả câu mô tả (text) và khung hình (image) được ánh xạ về cùng không gian vector 1152 chiều.
2. **Typo Correction & Query Expansion**: Nhận diện lỗi gõ phím tiếng Việt (Telex/VNI) và phân rã truy vấn thành 3 góc nhìn trực quan (Scene, Action, Tags) bằng LLM local trước khi tìm kiếm.
3. **Temporal Search**: Nhận diện các sự kiện diễn ra tuần tự theo thời gian để lọc và xếp hạng chuỗi khung hình.
4. **Fast Vector Search (FAISS IndexFlatIP)**: So khớp độ tương đồng Cosine trên 734.544 vectors chỉ trong < 50ms.

---

## 2. Phase 1: Offline Ingestion (Xử lý tiền kỳ một lần)

### 2.1 — Master Metadata Table (`indexes/metadata.parquet`)
- **Nguồn dữ liệu**: Các file CSV ánh xạ do BTC cung cấp cho 726 video (`L01` đến `L24`).
- **Nội dung lưu trữ**:
  - `video_id`: Mã video (ví dụ `L20_V015`).
  - `keyframe_name`: Tên file ảnh (ví dụ `299`).
  - `frame_id`: Chỉ số frame gốc trong file video MP4 — **đây là giá trị chính xác để nộp bài**.
  - `pts_time`: Mốc thời gian thực tế trong video tính bằng giây.
  - `faiss_idx`: Chỉ số nguyên liên tục $(0 \dots 734.543)$ khớp 1-1 với dòng trong FAISS index.
  - `keyframe_path`: Đường dẫn file ảnh keyframe.
- **Định dạng**: Apache Parquet nén Snappy, đọc siêu tốc bằng Polars DataFrame.

### 2.2 — SigLIP-SO400M Feature Extraction & FAISS Index
- **Model**: `ViT-SO400M-14-SigLIP` (`timm/ViT-SO400M-14-SigLIP`, pretrained `webli`).
- **Đặc trưng**: Vector 1152 chiều float32 được chuẩn hóa $L_2$ (`normalize=True`).
- **Index**: `indexes/faiss_siglip.index` (Dung lượng 3.38 GB, chứa 734.544 vector, dùng `IndexFlatIP`).
- **Ưu điểm so với CLIP ViT-B-32 cũ**:
  - Phân biệt màu sắc, chi tiết trang phục, biển hiệu, góc máy vượt trội.
  - Loại bỏ hoàn toàn hiện tượng nhầm lẫn giữa các vật thể có màu sắc tương tự.

---

## 3. Phase 2: Online Retrieval Pipeline (Khi người dùng tìm kiếm)

```
[Truy vấn Tiếng Việt của người dùng]
                 │
                 ▼
   [QueryExpander: Qwen2.5-3B-Instruct (4-bit)]
     ├── 1. Tự động sửa lỗi chính tả / Telex (VD: "tay đưa" -> "tay đua")
     └── 2. Phân rã thành 3 prompt Tiếng Anh chuyên biệt:
            • Prompt 1: Chi tiết bối cảnh trực quan (Scene description)
            • Prompt 2: Trọng tâm chủ thể & hành động (Subject & Action)
            • Prompt 3: Từ khóa vật thể & thuộc tính (Tags & Attributes)
                 │
                 ▼
     [SigLIP Text Encoder (ViT-SO400M)]
     ├── Encode 3 prompts -> 3 vector 1152-dim
     └── Weighted Blending: v_query = 0.5*v1 + 0.25*v2 + 0.25*v3 (L2 normalized)
                 │
                 ├── [Có từ khóa thời gian? (sau đó, rồi, tiếp theo)] ──► Temporal Multi-Event Search
                 │
                 ▼
     [FAISS IndexFlatIP Search (734.544 vectors)]
                 │
                 ▼ (Thời gian < 50ms)
     [Join Metadata (video_id, frame_id, pts_time, image_path)]
                 │
                 ▼
     [Hiển thị kết quả trên Web UI & Nộp bài DRES 1-Click]
```

### 3.1 — Multi-Query Expansion & Typo Correction
- **File**: `src/online_pipeline/query_expander.py`
- **Bộ nhớ**: Chạy lượng tử hóa 4-bit (`BitsAndBytesConfig`, `nf4`), chỉ chiếm **2.05 GB VRAM**.
- **Cơ chế**:
  - Nếu gặp từ gõ sai ngữ nghĩa (ví dụ `tay đưa xe đạp` thay vì `tay đua xe đạp`), mô hình nhận diện ngữ cảnh và tự sửa về `cyclist / bicycle racer`.
  - Sinh 3 mô tả trực quan bổ trợ cho nhau nhằm bao phủ nhiều khía cạnh của video.
  - Có bộ nhớ đệm (LRU Cache) để các truy vấn lặp lại trả về ngay lập tức (0.001s).
  - Có cơ chế Fallback tự động sang Google GTX translation (`translate.googleapis.com`) nếu GPU bận.

### 3.2 — Weighted Embedding Blending
- Vector truy vấn tổng hợp:
  $$\vec{v}_{\text{blend}} = \text{Normalize}\Big(0.5 \cdot \vec{v}_{\text{scene}} + 0.25 \cdot \vec{v}_{\text{action}} + 0.25 \cdot \vec{v}_{\text{tags}}\Big)$$
- Giúp kết quả tìm kiếm vừa đúng toàn cảnh bối cảnh, vừa bắt đúng hành động và vật thể chi tiết.

### 3.3 — Temporal Detection & Chronological Search
- **File**: `src/online_pipeline/temporal_search.py`
- Khi truy vấn chứa các từ nối thời gian (`"sau đó"`, `"tiếp theo"`, `"then"`, `"after"`):
  - Phân tách thành chuỗi các sự kiện con: $E_1 \rightarrow E_2 \rightarrow \dots \rightarrow E_n$.
  - Tìm kiếm ứng viên cho từng sự kiện.
  - Lọc và chấm điểm cao cho các video thỏa mãn điều kiện thời gian: $\text{frame\_id}(E_1) < \text{frame\_id}(E_2)$.

---

## 4. Chi tiết 4 đầu Task thi đấu

### 4.1 — KIS-T (Textual Known-Item Search)
- **Mục tiêu**: Cho một đoạn mô tả văn bản tiếng Việt/Anh, tìm chính xác video và frame tương ứng.
- **Quy trình**:
  1. Người dùng nhập câu mô tả vào thanh tìm kiếm.
  2. `QueryExpander` sửa lỗi và sinh 3 prompt tiếng Anh.
  3. SigLIP mã hóa và tìm Top-100/200 ứng viên từ FAISS.
  4. Người dùng quan sát lưới kết quả, click chọn khung hình đúng nhất.
  5. Bấm nút **"Nộp DRES"** trực tiếp trên thẻ ảnh hoặc xuất mã nộp: `<video_id>, <frame_id>`.

---

### 4.2 — KIS-V (Visual Known-Item Search)
- **Mục tiêu**: Đề thi là 1 đoạn video clip 5 giây (hoặc một bức ảnh mẫu). Thí sinh phải xác định video nào và ở giây thứ mấy chứa phân cảnh đó.
- **Quy trình hiện tại**:
  1. Khi video 5s chiếu lặp trên màn hình thi, người dùng nhấn `Win + Shift + S` chụp 1 khung hình rõ nét nhất.
  2. Kéo thả file ảnh chụp vào ô **Tìm kiếm ảnh (Visual KIS)** hoặc dán từ clipboard.
  3. Backend truyền ảnh qua `SigLIP Image Encoder` $\rightarrow$ sinh vector 1152 chiều $\rightarrow$ truy vấn FAISS.
  4. Vì ảnh mẫu cắt từ cùng video gốc trong dataset, độ tương đồng Cosine đạt cực cao ($> 0.85 - 0.95$), frame đáp án xuất hiện ngay tại **Top 1**.
  5. Người dùng click nộp bài trực tiếp vào DRES.
- **Định hướng mở rộng (Hybrid Multi-Modal KIS-V)**:
  - Cho phép kết hợp: `[Ảnh chụp từ clip 5s]` + `[Mô tả bổ trợ ngắn (Text)]`.
  - Phù hợp cho các cảnh khó (ví dụ: cảnh đường phố chung chung nhiều xe giống nhau, nhưng có thêm mô tả *"xe tải màu đỏ vượt đèn"* để định hướng trọng tâm).

---

### 4.3 — KIS-QA (Question & Answering)
- **Mục tiêu**: Đề thi đặt một câu hỏi về một chi tiết trong video (Ví dụ: *"Người đàn ông áo xanh cầm vật gì trên tay?"*).
- **Quy trình**:
  1. Dùng thanh tìm kiếm để xác định phân cảnh xảy ra sự việc.
  2. Bấm nút **Ghim (Pin)** khung hình nghi vấn vào khay làm bài (Export Area).
  3. Nhập trực tiếp câu trả lời dạng chữ vào ô `Đáp án QA` trên khung hình đã ghim.
  4. Hệ thống tự động tạo chuỗi nộp đúng chuẩn BTC:
     ```
     QA-<ANSWER>-<VIDEO_ID>-<TIME_MS>
     ```
     *(Trong đó `TIME_MS` được tính tự động từ `pts_time * 1000`)*.
  5. Bấm nộp trực tiếp sang DRES bằng 1 click.

---

### 4.4 — TRAKE (Temporal Retrieval of Anchoring Keyframes)
- **Mục tiêu**: Tìm một chuỗi gồm nhiều mốc hành động liên tiếp trong cùng một video theo đúng thứ tự thời gian.
- **Quy trình**:
  1. Tìm kiếm và bấm Ghim (Pin) các frame thể hiện từng sự kiện theo thứ tự xuất hiện.
  2. Hệ thống kiểm tra điều kiện thời gian:
     $$\text{frame\_id}_1 < \text{frame\_id}_2 < \dots < \text{frame\_id}_n$$
  3. Tự động sinh chuỗi nộp chuẩn TRAKE:
     ```
     TR-<VIDEO_ID>-<FRAME_1>-<FRAME_2>-...
     ```
  4. Bấm nộp trực tiếp vào DRES.

---

## 5. DRES Submission Pipeline

Hệ thống kết nối trực tiếp với server chấm thi DRES qua module `src/dres_client.py`:
- **Endpoint**: `https://aiclub.uit.edu.vn/dres` (hoặc mạng nội bộ thi đấu `http://192.168.20.150:8080`).
- **Session ID**: Lưu trong `src/config.py` và có thể chỉnh sửa động từ Web UI.
- **Xử lý Batch L vs Batch K**:
  - Đề thi thử hiện tại sử dụng evaluation ID `bcb482c6-2b9e-4908-877a-e5ccd7c04640` (Chỉ chấp nhận các video `Lxx_Vxxx`).
  - Hệ thống tự động kiểm tra task target để đảm bảo không nộp nhầm tập dữ liệu gây lỗi 500.

---

## 6. Kế hoạch phát triển tiếp theo (Roadmap)

1. **VLM Reranking Top-30/50 (Qwen2-VL-7B-Instruct)**:
   - Dùng SigLIP FAISS lấy nhanh Top-30 khung hình tiềm năng (~50ms).
   - Dùng Qwen2-VL-7B (4-bit, ~4.5 GB VRAM) chấm điểm tương quan chi tiết giữa câu hỏi và từng khung hình để đẩy khung hình chuẩn xác nhất lên Top 1–3.
2. **Hybrid KIS-V (Image + Text Composed Query)**:
   - Hỗ trợ nhập kèm text gợi ý khi tìm kiếm bằng ảnh trong KIS-V để xử lý các phân cảnh gây nhiễu.
3. **OCR & Whisper ASR**:
   - Trích xuất chữ biển hiệu, text màn hình (PaddleOCR).
   - Nhận dạng giọng nói/lời thoại tiếng Việt (vinai/PhoWhisper-small có sẵn trong cache).
