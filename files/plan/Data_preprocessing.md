# Data preprocessing

**Triết lý thiết kế:** Kết hợp chiến lược trích xuất keyframe thông minh của SOICT, embedding hình ảnh mạnh hơn MERVIN, và hệ thống caption + temporal memory của U-CESE.
Mục tiêu phục vụ 3 task: **KIS-T** (visual embedding mạnh), **TRAKE** (caption + temporal memory), **VQA** (caption chi tiết thuộc tính).

---

### Stage 1 — Shot Detection: TransNetV2

TransNetV2 là model phát hiện shot boundary state-of-the-art, phát hiện cả hard cut lẫn gradual transition (dissolve, fade) mà AutoShot và DAKE bỏ sót.
Chạy local, không cần GPU lớn.

**Output:** Danh sách shot boundary timestamps cho mỗi video.

---

### Stage 2 — Keyframe Selection: Semantic-Diff within Shots (SOICT principle)

Trong mỗi shot đã phát hiện:
1. Extract frame tại mỗi vị trí thứ 8 trong shot (sampling thô).
2. Tính SigLIP embedding cho từng frame đó.
3. Giữ lại frame nếu cosine similarity với frame được giữ trước đó < ngưỡng (ví dụ 0.90).
4. Kết quả: 1–5 keyframe/shot tùy mức độ biến đổi hình ảnh.

**Lý do tốt hơn MERVIN (3 fixed positions):** MERVIN's 0.15/0.50/0.85 có thể chọn ra 3 frame gần như giống nhau trong shot tĩnh (talking-head). Cách này đảm bảo mọi keyframe trong database đều có nội dung khác biệt về mặt ngữ nghĩa, không lưu frame dư thừa.

**Output:** Tập keyframe JPEG đại diện, đã khử trùng lặp về mặt ngữ nghĩa.

---

### Stage 3A — Visual Embedding: SigLIP-SO400M

Thay thế ViT-B/32 (đã bị giới hạn bởi BTC) bằng **SigLIP-SO400M** của Google.
- Attribute binding tốt hơn đáng kể: phân biệt được "áo đỏ" vs "áo trắng" — thứ ViT-B/32 hoàn toàn không làm được.
- Zero-shot performance mạnh hơn ViT-B/32 và ViT-H/14.
- Chạy được trên RTX 4060 (8GB VRAM).

Encode tất cả keyframe → vector embedding → FAISS IndexFlatIP.

**Output:** `faiss_siglip.index` (phục vụ KIS-T)

---

### Stage 3B — Object Detection: YOLOv11m

Chạy YOLO inference trên toàn bộ keyframe.
- Lưu toàn bộ detection (không lọc theo confidence ngưỡng cố định — giữ confidence thô).
- Lưu label, confidence, bbox (normalized ymin/xmin/ymax/xmax) vào Parquet.
- Thêm cột `source = "yolo"` để phân biệt với object detection của BTC (nếu cần merge sau).

**Output:** `objects.parquet` (phục vụ object boost filter)

---

### Stage 3C — Transcript Extraction: Whisper

Chạy OpenAI Whisper locally trên toàn bộ file MP4 có sẵn.
- Segment transcript theo timestamp, aligned với shot boundary từ Stage 1.
- Lưu: `video_id`, `shot_id`, `start_time`, `end_time`, `text`.

**Output:** `transcripts.parquet` (phục vụ caption context và VQA)

---

### Stage 4 — Keyframe Captioning: Gemini + Context + Whisper (U-CESE)

Với mỗi keyframe, gửi Gemini:
- **Input 1:** Keyframe hiện tại (ảnh)
- **Input 2:** k=2 keyframe xung quanh (temporal context)
- **Input 3:** Đoạn Whisper transcript aligned với shot

**Instruction (3 bước, theo U-CESE):**
1. Suy luận ngữ cảnh nền từ surrounding frames và phụ đề.
2. Phân tích nội dung trực quan: đối tượng, thuộc tính (màu sắc, trang phục), tương tác, setting.
3. Sinh caption đầy đủ ngữ cảnh — bao gồm cả thông tin nghe được (nếu có), không chỉ quan sát trực quan.

**Ví dụ output caption tốt:**
> *"A man in a red polo shirt stands at an outdoor podium with dense green trees behind him. He gestures with his right hand toward the crowd. The subtitle reads 'we must protect...' suggesting a policy speech at an open-air press conference."*

**Lý do cần captioning:** CLIP image embedding không thể encode được "áo đỏ", "đang chỉ tay", hay "cuộc họp báo ngoài trời" vào 1 vector. Caption text được encode bằng sentence-embedding mạnh hơn rất nhiều cho các truy vấn có thuộc tính chi tiết.

**Output:** `captions.parquet` (`video_id`, `keyframe_name`, `caption`)

---

### Stage 5 — ReCap: Shot-Level Temporal Memory (U-CESE)

Xử lý tuần tự theo thứ tự shot trong mỗi video, duy trì bộ nhớ tích lũy `M_t`:

$$(C_t, M_t) = f_{\text{Gemini}}(S_t,\; M_{t-1})$$

- `S_t`: tập caption của các keyframe trong shot t
- `M_{t-1}`: shot summary tích lũy từ các shot trước
- Output `C_t`: shot-level narrative summary có tính liên kết thời gian

**Tại sao quan trọng cho TRAKE:** Shot summary biết "trước đó người đàn ông này vừa mở cửa bước vào" — thông tin mà caption đơn lẻ của keyframe hiện tại không thể có.

**Output:** `shot_summaries.parquet` (`video_id`, `shot_id`, `summary`, `memory_context`)

---

### Stage 6 — Video-Level Summary

Aggregate tất cả shot summary của 1 video → 1 Gemini call → global video description.

**Output:** `video_summaries.parquet` (`video_id`, `summary`)
Phục vụ **TRAKE Phase A** (xác định video nào khớp nhất với toàn bộ mô tả chuỗi sự kiện).

---

### Stage 7 — Caption Embedding Index

Encode tất cả keyframe caption (Stage 4) bằng strong multilingual sentence-embedding model:
- Đề xuất: **bge-m3** (BGE multilingual, mạnh trên tiếng Việt và tiếng Anh) hoặc **paraphrase-multilingual-mpnet-base-v2**
- Lý do không dùng all-MiniLM-L6-v2 ở đây: caption là tiếng Anh nhưng query gốc là tiếng Việt — cần model multilingual để khoảng cách embedding không bị lệch sau dịch.

Tất cả caption vector → FAISS IndexFlatIP thứ 2.

**Output:** `faiss_caption.index` (phục vụ TRAKE Phase B và VQA)

---

### Tổng kết Output và Task mapping

| Output | Phục vụ Task |
|---|---|
| `faiss_siglip.index` — SigLIP image embeddings | **KIS-T** |
| `faiss_caption.index` — Caption text embeddings | **TRAKE Phase B**, **VQA** |
| `video_summaries.parquet` — Global video descriptions | **TRAKE Phase A** |
| `shot_summaries.parquet` — Temporal shot memory | **TRAKE** temporal context |
| `objects.parquet` — YOLO detections + confidence | Object boost filter (tất cả task) |
| `transcripts.parquet` — Whisper segments | Caption context, **VQA** |
| `captions.parquet` — Per-keyframe captions | Lookup và reranking |

---

### Tại sao tốt hơn các pipeline tham khảo

| | U-CESE | MERVIN | SOICT | **Our Pipeline** |
|---|---|---|---|---|
| Shot detection | DAKE (proxy) | TransNetV2 | AutoShot | **TransNetV2** |
| Keyframe selection | File-size dynamic | 3 fixed positions | BEiT-3 semantic diff | **SigLIP semantic diff** |
| Visual embedding | CLIP (weak) | PE-Core-bigG | BEiT-3 | **SigLIP-SO400M** |
| Caption quality | Gemini + ReCap | Gemini Flash only | None | **Gemini + Whisper + ReCap** |
| Caption embedding | (not specified) | Vietnamese-embedding3 | None | **bge-m3 (multilingual)** |
| TRAKE capability | Outstanding | Weak | Weak | **Outstanding** |
| KIS-T capability | Weak | Good | Good | **Strong** |
| Object detection | (not specified) | (not specified) | (not specified) | **YOLOv11m** |

