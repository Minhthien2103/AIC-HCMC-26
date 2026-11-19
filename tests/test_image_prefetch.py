from contextlib import closing
import threading
import unittest

from src.online_pipeline.image_prefetch import prefetch_images


class Image:
    def __init__(self, value):
        self.value = value
        self.closed = False

    def close(self):
        self.closed = True


class PrefetchTests(unittest.TestCase):
    def test_downloads_overlap_but_results_keep_input_order(self):
        barrier = threading.Barrier(2)
        images = []
        def fetch(value):
            if value < 2:
                barrier.wait(timeout=3)  # Would fail for sequential downloads.
            return Image(value)
        with closing(prefetch_images(range(6), fetch, workers=2)) as stream:
            for item, image in stream:
                self.assertIsNotNone(image)
                self.assertFalse(image.closed)
                self.assertEqual(item, image.value)
                images.append(image)
        self.assertEqual([image.value for image in images], list(range(6)))
        self.assertTrue(all(image.closed for image in images))

    def test_prefetch_window_is_bounded_and_early_exit_closes_images(self):
        fetched = []
        third_ready = threading.Event()
        def fetch(value):
            image = Image(value)
            fetched.append(image)
            if value == 2:
                third_ready.set()
            return image
        with closing(prefetch_images(range(100), fetch, workers=2)) as stream:
            item, image = next(stream)
            self.assertEqual(item, 0)
            self.assertTrue(third_ready.wait(3))
            self.assertEqual(len(fetched), 3)  # Two ahead plus current image.
            self.assertFalse(image.closed)
        self.assertTrue(all(image.closed for image in fetched))

    def test_one_download_failure_does_not_stop_later_images(self):
        def fetch(value):
            if value == 1:
                raise OSError('test timeout')
            return Image(value)
        with self.assertLogs('src.online_pipeline.image_prefetch', level='WARNING'):
            with closing(prefetch_images(range(3), fetch, workers=2)) as stream:
                result = [(item, image.value if image is not None else None) for item, image in stream]
        self.assertEqual(result, [(0, 0), (1, None), (2, 2)])

    def test_empty_input(self):
        self.assertEqual(list(prefetch_images([], lambda item: self.fail('unexpected fetch'))), [])


if __name__ == '__main__':
    unittest.main()
