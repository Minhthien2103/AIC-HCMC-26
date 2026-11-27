"""Static media server with single-byte-range support (Python 3.10+).

Run: python3 scripts/media_server.py --directory /home/team12/aic-data
"""
import argparse
import os
import re
import io
from functools import lru_cache
from threading import BoundedSemaphore
from urllib.parse import parse_qs, urlsplit
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

thumbnail_slots = BoundedSemaphore(2)


@lru_cache(maxsize=256)
def thumbnail_bytes(path, mtime_ns, size):
    from PIL import Image, ImageOps
    with thumbnail_slots, Image.open(path) as source:
        source.draft('RGB', (320, 180))
        preview = ImageOps.exif_transpose(source)
        preview.thumbnail((320, 180))
        output = io.BytesIO()
        preview.convert('RGB').save(output, format='JPEG', quality=65)
        return output.getvalue()


class MediaHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self.remaining = None
        path = self.translate_path(self.path)
        if parse_qs(urlsplit(self.path).query).get('thumb') == ['1']:
            resolved = Path(path).resolve()
            root = Path(self.directory).resolve()
            if not resolved.is_relative_to(root) or resolved.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                self.send_error(400, 'Invalid thumbnail path')
                return None
            try:
                stat = resolved.stat()
                data = thumbnail_bytes(str(resolved), stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                self.send_error(404, 'Image not found')
                return None
            except Exception:
                self.send_error(503, 'Thumbnail unavailable; install Pillow on media server')
                return None
            etag = f'"thumb-v1-{stat.st_mtime_ns}-{stat.st_size}"'
            if self.headers.get('If-None-Match') == etag:
                self.send_response(304)
                self.send_header('ETag', etag)
                self.end_headers()
                return None
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'public, max-age=3600')
            self.send_header('ETag', etag)
            self.end_headers()
            return io.BytesIO(data)
        value = self.headers.get("Range", "")
        # HEAD ignores Range. Unsupported/multipart ranges and If-Range
        # conservatively fall back to a full response, never a stale fragment.
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
        if (self.command != "GET" or not match or not any(match.groups())
                or self.headers.get("If-Range") or not os.path.isfile(path)):
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        try:
            stat = os.fstat(source.fileno())
            size = stat.st_size
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(0, size - int(last))
                end = size - 1
            if size == 0 or start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                source.close()
                return None
            self.remaining = end - start + 1
            source.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(self.remaining))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return source
        except Exception:
            source.close()
            raise

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            remaining = self.remaining
            while remaining > 0:
                chunk = source.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers cancel the current transfer when the user seeks.
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8007)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"Media directory does not exist: {args.directory}")
    handler = partial(MediaHandler, directory=str(args.directory.resolve()))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f"Serving {args.directory} on {args.bind}:{args.port} with HTTP Range", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
