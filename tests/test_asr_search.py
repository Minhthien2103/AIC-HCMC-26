import tempfile
import unittest
import shutil
import wave
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pathlib import Path

import polars as pl

from src.online_pipeline.asr_search import ASRSearchEngine
from scripts.extract_asr import SCHEMA, audio_windows, transcribe_audio


class ASRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'asr.parquet'
        self.engine = ASRSearchEngine(self.path)
        self.rows = [dict(id=i, score=0.5, distance=0.5, score_space='unit',
                         entity=dict(video=v, time=t)) for i, (v, t) in enumerate([('V1', 12), ('V1', 20), ('V2', 12)])]

    def write(self, text='Lễ hội múa lân'):
        pl.DataFrame([dict(video_id='V1', start=10.0, end=20.0, text=text)], schema=SCHEMA).write_parquet(self.path)

    def test_empty_input_no_io_or_score_change(self):
        result, warning = self.engine.boost(self.rows, '')
        self.assertIs(result, self.rows)
        self.assertFalse(warning)

    def test_missing_file_warns_then_new_file_is_loaded(self):
        result, warning = self.engine.boost(self.rows, 'múa lân')
        self.assertIs(result, self.rows)
        self.assertTrue(warning)
        self.write()
        result, warning = self.engine.boost(self.rows, 'MÚA LÂN, lễ hội')
        self.assertFalse(warning)
        self.assertAlmostEqual(result[0]['score'], 0.7/1.2)
        self.assertEqual(len(result), 3)
        self.assertNotIn('asr_match', result[1]['entity'])  # End boundary excluded.
        self.assertNotIn('asr_match', result[2]['entity'])  # Different video.
        self.assertEqual(self.rows[0]['score'], 0.5)

    def test_whole_phrases_and_no_hit_keep_scores(self):
        self.write('số 115, đang đi')
        result, warning = self.engine.boost(self.rows, '15')
        self.assertIs(result, self.rows)
        result, warning = self.engine.boost(self.rows, '115')
        self.assertIn('asr_match', result[0]['entity'])

    def test_corrupt_file_does_not_reuse_old_matches(self):
        self.write()
        self.engine.boost(self.rows, 'múa lân')
        self.path.write_bytes(b'broken')
        result, warning = self.engine.boost(self.rows, 'múa lân')
        self.assertIs(result, self.rows)
        self.assertTrue(warning)

    def test_batching_preserves_timestamps_and_flushes_tail_after_silence(self):
        batches = []
        class Recognizer:
            def create_stream(self):
                return SimpleNamespace(accept_waveform=lambda rate, samples: None,
                                       result=SimpleNamespace(text='hello'))
            def decode_streams(self, streams):
                batches.append(len(streams))
        def windows(*args):
            for i in range(19):
                yield i * 13, np.zeros(16000) if i == 3 else np.ones(16000)
        with patch('scripts.extract_asr.audio_windows', windows):
            table = transcribe_audio(Recognizer(), 'unused', 'V1', batch_size=16)
        self.assertEqual(batches, [16, 2])
        self.assertEqual(table.height, 18)
        self.assertNotIn(39, table['start'].to_list())
        self.assertEqual(table['end'][-1], 235)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_audio_windows_keep_absolute_time_and_tail(self):
        path = Path(self.temp.name) / 'audio.wav'
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\0\0' * 16000 * 32)
        windows = list(audio_windows(path, 15, 2))
        self.assertEqual([start for start, _ in windows], [0, 13, 26])
        self.assertEqual([len(samples)/16000 for _, samples in windows], [15, 15, 6])
        with self.assertRaises(RuntimeError):
            list(audio_windows(path.with_name('missing.wav'), 15, 2))


if __name__ == '__main__':
    unittest.main()
