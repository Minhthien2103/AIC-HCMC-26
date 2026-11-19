"""Real HTTP requests; run without pytest: python -m unittest discover -s tests -p test_media_server.py"""
import http.client
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts.media_server import MediaHandler


class MediaServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.payload = bytes(range(256)) * 32
        Path(cls.directory.name, "video.mp4").write_bytes(cls.payload)
        Path(cls.directory.name, "empty.mp4").touch()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(
            MediaHandler, directory=cls.directory.name))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.directory.cleanup()

    def request(self, headers=None, method="GET", path="/video.mp4"):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            conn.request(method, path, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_ranges_return_exact_bytes(self):
        for value, start, end in [("bytes=0-1023", 0, 1023),
                                  ("bytes=4096-", 4096, 8191),
                                  ("bytes=-128", 8064, 8191),
                                  ("bytes=8000-9999", 8000, 8191)]:
            with self.subTest(value=value):
                status, headers, body = self.request({"Range": value})
                self.assertEqual(status, 206)
                self.assertEqual(headers["Content-Range"], f"bytes {start}-{end}/8192")
                self.assertEqual(int(headers["Content-Length"]), len(body))
                self.assertEqual(body, self.payload[start:end + 1])

    def test_unsatisfiable_and_empty(self):
        for path, value, size in [("/video.mp4", "bytes=8192-", 8192),
                                   ("/video.mp4", "bytes=-0", 8192),
                                   ("/empty.mp4", "bytes=0-10", 0)]:
            status, headers, body = self.request({"Range": value}, path=path)
            self.assertEqual(status, 416)
            self.assertEqual(headers["Content-Range"], f"bytes */{size}")
            self.assertEqual(body, b"")

    def test_full_and_fallback(self):
        for headers in [{}, {"Range": "garbage"}, {"Range": "bytes=0-1,4-5"},
                        {"Range": "bytes=0-1", "If-Range": '"old-version"'}]:
            status, result_headers, body = self.request(headers)
            self.assertEqual(status, 200)
            self.assertEqual(result_headers["Accept-Ranges"], "bytes")
            self.assertEqual(body, self.payload)

    def test_head_and_missing(self):
        status, headers, body = self.request({"Range": "bytes=0-1"}, method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(int(headers["Content-Length"]), len(self.payload))
        self.assertEqual(body, b"")
        self.assertEqual(self.request(path="/missing.mp4")[0], 404)


if __name__ == "__main__":
    unittest.main()
