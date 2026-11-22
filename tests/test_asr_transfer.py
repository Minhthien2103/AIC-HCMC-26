import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.asr_local_gpu import download


class Channel:
    def __init__(self, code=0):
        self.payload = b'RIFF' + b'0' * 100
        self.code = code
        self.closed = False
    def exec_command(self, command):
        self.command = command
    def recv_ready(self):
        return bool(self.payload)
    def recv(self, size):
        payload, self.payload = self.payload, b''
        return payload
    def recv_stderr_ready(self):
        return False
    def exit_status_ready(self):
        return not self.payload
    def recv_exit_status(self):
        return self.code
    def close(self):
        self.closed = True


class Client:
    def __init__(self, channel, size=100):
        self.channel, self.size = channel, size
    def get_transport(self):
        return SimpleNamespace(open_session=lambda **kw: self.channel)
    def open_sftp(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def stat(self, path):
        return SimpleNamespace(st_size=self.size, st_mtime=10)
    def close(self):
        pass


class TransferTests(unittest.TestCase):
    def test_success_failure_cancel_and_source_change_cleanup(self):
        for case in ['ok', 'failed', 'cancelled', 'changed']:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                channel = Channel(1 if case == 'failed' else 0)
                stop = threading.Event()
                if case == 'cancelled':
                    stop.set()
                destination = Path(directory) / 'audio.wav'
                client = Client(channel, 200 if case == 'changed' else 100)
                args = SimpleNamespace(max_audio_seconds=60, max_audio_mb=1)
                source = dict(path='/videos/a video.mp4', size=100, mtime=10)
                with patch('scripts.asr_local_gpu.connect', return_value=client):
                    if case == 'ok':
                        self.assertEqual(download(args, 'unused', source, destination, stop), destination)
                        self.assertTrue(destination.exists())
                        self.assertIn("'/videos/a video.mp4'", channel.command)
                    else:
                        with self.assertRaises((RuntimeError, InterruptedError)):
                            download(args, 'unused', source, destination, stop)
                        self.assertFalse(destination.exists())
                self.assertTrue(channel.closed)


if __name__ == '__main__':
    unittest.main()
