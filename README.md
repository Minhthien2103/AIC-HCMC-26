# Hướng dẫn chạy hệ thống

Mở 3 cửa sổ và giữ chúng chạy trong lúc sử dụng.

## 1. Anaconda Prompt — chạy web và backend

```bat
cd /d "C:\My Work\UIT\HK5\System\AIC-HCMC-26"
conda activate system
python src/server.py --port 8006
```

Thay đường dẫn dự án nếu bạn lưu ở thư mục khác.

## 2. PowerShell mới — chạy media server hỗ trợ tua trên Ubuntu

```powershell
ssh team12@192.168.28.112
```

Nhập mật khẩu. Sau khi vào Ubuntu, chạy:

```bash
python3 ~/media_server_test.py --directory /home/team12/aic-data --port 8008
```

## 3. PowerShell mới nữa — nối máy bạn tới media server

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 8007:localhost:8008 team12@192.168.28.112
```

Nhập mật khẩu nếu được yêu cầu và giữ cửa sổ này mở.

## Mở hệ thống

- Web: [http://localhost:8006](http://localhost:8006)
- Trang đánh giá DRES: [https://aiclub.uit.edu.vn/dres/evaluation/list](https://aiclub.uit.edu.vn/dres/evaluation/list)
