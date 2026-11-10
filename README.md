# AIC-HCMC 2026 — Hệ thống Truy vấn Video & Nộp bài

Hệ thống tìm kiếm keyframe video cho nội dung KIS-T, TRAKE và VQA.

---

## 1. Pipeline hiện tại (Kiến trúc hệ thống)

Hệ thống hoạt động theo mô hình truy vấn đa phương thức (Multimodal Video Retrieval) cục bộ:

```
[User Query (Text / Ảnh)] 
       │
       ▼
[Tự động dịch Tiếng Việt → Tiếng Anh (mBART / Google Fallback)]
       │
       ▼
[OpenCLIP Encoder (ViT-B-32-quickgelu, 512-dim)]
       │
       ▼
[FAISS FlatIP Index (Tìm kiếm vector ~20-30ms)]
       │
       ▼
[Temporal Re-ranking / Multi-event Boost (TRAKE & KIS)]
       │
       ▼
[Web UI hiển thị danh sách Keyframe & Hỗ trợ nộp bài DRES]
```

### Các tính năng cốt lõi:
- **GPU RTX 4060 + OpenCLIP**: Sử dụng model `ViT-B-32-quickgelu` (checkpoint `openai`, 512 dimensions), tương thích 100% với đặc trưng vector của BTC.
- **FAISS Vector Index**: Index nạp trực tiếp vào RAM, tìm kiếm siêu tốc 1.000 frame tương đồng nhất trong **< 30ms**.
- **Hỗ trợ truy vấn Tiếng Việt tự động**: Nhập trực tiếp tiếng Việt hoặc tiếng Anh, backend tự chuẩn hóa và chuyển ngữ trong suốt mà không cần chỉnh công tắc thủ công.
- **Xử lý chuỗi sự kiện TRAKE**: Hỗ trợ thêm không giới hạn các sự kiện (`Event 1 -> Event 2 -> Event N`), tự động boost điểm cho các video có chuỗi sự kiện diễn ra đúng thứ tự thời gian.
- **Tìm kiếm tương đồng (Similarity Search)**: Bấm nút `⚡ Tương đồng` trên bất kỳ keyframe nào để tìm lại các frame tương tự trong toàn bộ dữ liệu.

---

## 2. Cách cài đặt môi trường & Khởi động Web UI

### Bước 2.1: Tạo môi trường

Mở PowerShell / Terminal tại thư mục gốc của project:

- **Cách 1: Sử dụng Conda**:
  ```powershell
  # 1. Tạo môi trường mới với Python 3.10
  conda create -n <env_name> python=3.10 -y

  # 2. Kích hoạt môi trường
  conda activate <env_name>
  ```

- **Cách 2: Sử dụng venv chuẩn của Python**:
  ```powershell
  # 1. Tạo môi trường ảo venv
  python -m venv venv

  # 2. Kích hoạt môi trường trên Windows PowerShell
  .\venv\Scripts\activate
  ```

---

### Bước 2.2: Cài đặt PyTorch hỗ trợ GPU CUDA & Các thư viện

1. **Cài đặt PyTorch với CUDA 12.1** (để chạy OpenCLIP trên GPU):
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Cài đặt toàn bộ thư viện từ `requirements.txt`**:
   ```powershell
   pip install -r requirements.txt
   ```

---

### Bước 2.3: Chuẩn bị file dữ liệu Index (Bắt buộc)

Trước khi khởi động, hãy đảm bảo trong thư mục `indexes/` đã có sẵn 2 file cốt lõi:
1. `indexes/faiss_clip.index`: File index chứa vector đặc trưng của các keyframe.
2. `indexes/metadata.parquet`: Bảng thông tin ánh xạ (`video_id`, `frame_id`, `pts_time`).

*(Nếu chưa có, bạn có thể chạy script build offline hoặc contact me).*

---

### Bước 2.4: Khởi động Server & Mở Web UI

Chạy lệnh sau tại thư mục gốc:

```powershell
# Chạy server FastAPI (tự động nạp FAISS, nạp CLIP model lên GPU và host Web UI)
python src/server.py --port 8006
```

Khi terminal xuất hiện dòng thông báo:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8006
```

👉 Mở trình duyệt web và truy cập địa chỉ: **`http://localhost:8006`**

---

### Bước 2.5 (Tùy chọn): Kết nối Server Remote chứa ảnh Keyframe

Sử dụng server trường cấp (IP `192.168.28.112`):
1. Kết nối OpenVPN vào mạng trường bằng file `.ovpn`.
2. Trên máy chủ remote, bật HTTP server để chia sẻ ảnh:
   ```bash
   python3 -m http.server 8007 --directory /path/to/data
   ```
Web UI đã được cấu hình sẵn để tự động tải ảnh từ `http://192.168.28.112:8007`.

---

## 3. Hướng dẫn sử dụng Giao diện & Cách Nộp bài

### Bước 1: Tìm kiếm Keyframe
1. Chọn nội dung tại menu trên cùng: **`KIS-T`** hoăc **`VQA`** hoặc **`TRAKE`**.
2. Nhập câu mô tả vào ô text.
3. Nhấn **`Enter`** / nút **`Tìm kiếm`**.
4. *(Đối với TRAKE)*: Click nút **`+ Thêm sự kiện`** để thêm sự kiện.

### Bước 2: Ghim kết quả vào ô nộp bài (Export Area)
1. Ở bảng kết quả, rê chuột vào các frame đúng với yêu cầu đề thi.
2. Click vào nhãn **`📌 Ghim`** (hoặc kéo sang khung **Export Area** ở bên phải).
3. Frame nằm ở vị trí đầu tiên trong Export Area sẽ được chọn làm kết quả chính để nộp.

### Bước 3: Nộp bài lên hệ thống chấm thi
- **Cách 1: Nộp trực tiếp lên máy chấm DRES (Khuyên dùng)**:
  - Bấm nút **`🚀 Submit DRES`** ở góc trên bên phải.
  - Hệ thống sẽ tự lấy thông tin (Video ID, Frame ID) của frame và gửi đến server DRES (`https://eventretrieval.one`).
- **Cách 2: Xuất file kết quả thủ công**:
  - Bấm nút **`📁 Export Area`** để xem danh sách hoặc tải file CSV kết quả về máy để kiểm tra.

---

## 4. Cấu trúc thư mục chính

```
AIC-HCMC-26/
├── src/
│   ├── server.py              # Server FastAPI trung tâm (Backend API + Web static)
│   ├── config.py              # Cấu hình đường dẫn index, data, model
│   ├── online_pipeline/       # Bộ mã hóa OpenCLIP & công cụ tìm kiếm FAISS
│   ├── tasks/                 # Logic thuật toán xử lý KIS-T và TRAKE
│   └── utils/translation.py   # Module chuyển ngữ tự động Việt → Anh
├── baseline/retrievalBaseline/frontend/
│   ├── index.html             # Giao diện chính (Dark Mode)
│   ├── src/scripts/           # Xử lý sự kiện, gọi API backend, nộp DRES
│   └── src/styles/            # Tùy biến giao diện (dark_theme.css)
├── indexes/                   # Chứa faiss_clip.index và metadata.parquet
└── scripts/                   # Script đồng bộ dữ liệu server remote & setup
```

