# Media server hỗ trợ HTTP Range

Python 3.10+, không cần cài thêm thư viện. Giữ nguyên thư mục media và URL
`/videos/{video_id}.mp4`, `/keyframes/...`, `/map-keyframes/...`.

## Thử trên Ubuntu mà chưa dừng server cũ

Chạy từ thư mục repo chứa code nhánh `test` trên Ubuntu:

```bash
python3 scripts/media_server.py --directory /home/team12/aic-data --port 8008
```

Trong terminal khác trên Ubuntu:

```bash
curl -r 0-1023 -D - -o /dev/null http://127.0.0.1:8008/videos/L06_V029.mp4
```

Kỳ vọng `206 Partial Content`, `Content-Range: bytes 0-1023/<file-size>` và
`Content-Length: 1024`. Dùng đường dẫn video thực tế nếu file mẫu không tồn tại.

Muốn thử bằng trình duyệt Windows, mở tunnel riêng (giữ tunnel 8007 hiện tại):

```powershell
ssh -N -L 8008:localhost:8008 team12@192.168.28.112
```

Mở `http://localhost:8008/videos/L06_V029.mp4`, thử phát rồi tua tới giữa video.

### Dùng server thử trong web retrieval

Giữ cấu hình frontend và backend ở **localhost:8007**. Nếu server Range vẫn
chạy trên Ubuntu ở cổng 8008, đóng tunnel đang giữ cổng local 8007 rồi mở:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 8007:localhost:8008 team12@192.168.28.112
```

Lúc này local 8007 trỏ tới remote 8008, cả ảnh, video và VLM dùng cùng nguồn.
Giữ terminal server thử và tunnel mở. Khi tắt mọi terminal, server chạy foreground
có thể dừng; cần chạy lại trước khi mở tunnel. Để trở lại server cũ, đóng tunnel
này rồi dùng lệnh 8007:localhost:8007 trong README. Không mở hai tunnel cùng
chiếm cổng local 8007.

## Chuyển sang cổng 8007

Sau khi thử thành công, dừng server thử bằng Ctrl+C. Kiểm tra tiến trình đang
giữ cổng 8007 trên Ubuntu:

```bash
ss -ltnp 'sport = :8007'
```

Dừng đúng tiến trình `python -m http.server` cũ (Ctrl+C ở terminal đang chạy,
hoặc `kill <PID>` sau khi xác nhận PID), rồi chạy từ thư mục repo:

```bash
nohup python3 scripts/media_server.py --directory /home/team12/aic-data --port 8007 > media_server.log 2>&1 &
```

Mặc định chỉ bind loopback, dùng được qua SSH tunnel. Nếu cần phục vụ trực tiếp
qua LAN, thêm `--bind 0.0.0.0`.

Không chạy lại toàn bộ `setup_remote_server.sh` chỉ để đổi media server:
script đó còn tải dữ liệu, giải nén và xây index.

## Kiểm tra trên Windows

```powershell
curl.exe --max-time 5 -r 0-1023 -D - -o NUL http://localhost:8007/videos/L06_V029.mp4
```

Server hỗ trợ range đóng, mở và suffix; range ngoài file trả 416. Multipart
range hoặc If-Range được xử lý bằng phản hồi đầy đủ 200. HEAD trả thông tin
toàn file. Server truyền theo khối, không nạp cả video vào RAM.

HTTP Range cho phép lấy đúng đoạn để tua; không đảm bảo thời gian phát 3–5 giây.
Thời gian thực tế còn phụ thuộc mạng, bitrate và vị trí metadata trong MP4.
Player web dùng HLS.js cho MP4 là vấn đề riêng, chưa được sửa trong thay đổi này.

## Test local không cần pytest

```powershell
python -m unittest discover -s tests -p test_media_server.py -v
```
