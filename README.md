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

## 2. Quy trình khởi động hệ thống hàng ngày (Workflow chuẩn)

Mỗi khi mở máy tính và đã **kết nối thành công OpenVPN vào mạng trường**, bạn thực hiện các bước sau theo thứ tự:

---

### Bước 2.1: Đảm bảo Media Server cổng 8007 đang chạy trên Ubuntu Server
Mở terminal SSH vào server (`team12@192.168.28.112`) và kiểm tra:
```bash
# Kiểm tra xem media server cổng 8007 có đang chạy không:
ps aux | grep 8007 | grep -v grep

# Nếu chưa chạy, bật chạy ngầm bằng lệnh:
nohup python3 -m http.server 8007 --bind 0.0.0.0 --directory /home/team12/aic-data > /home/team12/aic-data/media.log 2>&1 &
```

---

### Bước 2.2: Mở đường ống SSH Tunnel trên máy Local (Bắt buộc)
Vì firewall mạng trường chặn cổng lạ giữa 2 dải IP, ta dùng SSH Tunnel để dẫn cổng 8007 về máy:

Mở một tab **PowerShell (Terminal 1)** trên máy bạn và chạy:
```powershell
ssh -N -L 8007:localhost:8007 team12@192.168.28.112
```
*(Nhập password: `123456`. Giữ nguyên cửa sổ terminal này chạy ngầm trong suốt quá trình làm việc).*

---

### Bước 2.3: Khởi động Backend Web Server (Port 8006)
Mở một tab **PowerShell (Terminal 2)** tại thư mục dự án, kích hoạt conda `aic26`:

```powershell
# 1. Kích hoạt môi trường conda có GPU RTX 4060
conda activate aic26

# 2. Chạy server FastAPI
python src/server.py --port 8006
```

👉 Mở trình duyệt web truy cập: **`http://localhost:8006`** để bắt đầu tìm kiếm!

---

## 3. Cách chạy tiếp trích xuất CLIP Features (Resume từ điểm đã dừng)

Script trích xuất được tích hợp cơ chế **Resume thông minh**: Tự động quét và bỏ qua trong 1 giây các video đã hoàn thành (như L01–L05), tiếp tục xử lý các video còn lại mà không bao giờ tính toán lại hay làm tràn ổ cứng.

### Lệnh chạy tiếp trích xuất:
Mở PowerShell, kích hoạt `conda activate aic26` và chạy:
```powershell
python scripts/extract_clip_features_gpu.py --batch-size 128 --workers 32
```
*Bạn có thể bấm `Ctrl + C` để dừng bất kỳ lúc nào, lần sau gõ lại lệnh trên nó sẽ tự động chạy tiếp từ video kế tiếp.*

### Cập nhật Database sau khi chạy thêm dữ liệu:
Bất cứ khi nào bạn muốn cập nhật các video mới vừa sinh vào thanh tìm kiếm:
```powershell
# 1. Gom các file CSV thành metadata.parquet
python -m src.offline_pipeline.metadata_loader

# 2. Xây dựng lại FAISS Index
python -m src.offline_pipeline.build_index

# 3. Khởi động lại server backend:
python src/server.py --port 8006
```

---

## 4. Hướng dẫn sử dụng Giao diện & Cách Nộp bài

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

