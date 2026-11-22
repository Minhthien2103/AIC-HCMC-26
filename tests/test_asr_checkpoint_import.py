import unittest
from scripts.asr_checkpoint_import import reusable_cpu, validate_cpu_source


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(video_id='V1', path='/videos/V1.mp4', size=100, mtime=123)
        self.cpu = dict(source='/dataset/V1.mp4', size=100, mtime=123000000456,
                        model='cpu-hash', version=1, window=15, overlap=2, silence_rms=0.001)
        self.signature = dict(source=self.source, host='server', model='gpu-hash',
                              provider='cuda', batch=16, window=15, overlap=2,
                              silence_rms=0.001, max_audio_seconds=0)

    def test_source_resolution_model_and_audio_settings_must_match(self):
        self.assertTrue(validate_cpu_source(self.cpu, self.source, '/dataset/V1.mp4', 'cpu-hash'))
        for key, value in [('size', 101), ('mtime', 124000000000), ('model', 'other'),
                           ('source', '/other/V1.mp4'), ('window', 30), ('overlap', 0),
                           ('silence_rms', 0.01), ('version', 2)]:
            with self.subTest(key=key):
                self.assertFalse(validate_cpu_source(dict(self.cpu, **{key:value}), self.source,
                                                     '/dataset/V1.mp4', 'cpu-hash'))

    def test_import_provenance_reused_independent_of_gpu_batch_but_not_source(self):
        imported = dict(self.signature, kind='imported_cpu', original_cpu_manifest=self.cpu)
        self.assertTrue(reusable_cpu(imported, self.signature))
        self.assertTrue(reusable_cpu(imported, dict(self.signature, batch=8)))
        for signature in [dict(self.signature, model='changed'), dict(self.signature, host='other'),
                          dict(self.signature, max_audio_seconds=60),
                          dict(self.signature, source=dict(self.source, mtime=124))]:
            self.assertFalse(reusable_cpu(imported, signature))
        self.assertFalse(reusable_cpu(dict(imported, kind='cuda'), self.signature))


if __name__ == '__main__':
    unittest.main()
