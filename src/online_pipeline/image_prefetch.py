"""Bounded, ordered image downloads overlapping sequential GPU inference."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import time

logger = logging.getLogger(__name__)


def _close_image(image):
    if image is not None and hasattr(image, 'close'):
        image.close()


def prefetch_images(items, fetch, workers=4):
    """Yield (item, image) in input order; own and close all downloaded images.

    Keep at most `workers` pending images plus the one being consumed. Callers
    must close the generator on early exit (e.g. contextlib.closing).
    """
    workers = max(1, min(8, int(workers)))
    iterator = iter(items)
    pending = deque()
    wait_seconds = 0.0
    completed = 0
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='vlm-image')

    def submit_next():
        try:
            item = next(iterator)
        except StopIteration:
            return
        pending.append((item, executor.submit(fetch, item)))

    try:
        for _ in range(workers):
            submit_next()
        while pending:
            item, future = pending.popleft()
            wait_started = time.perf_counter()
            try:
                image = future.result()
            except Exception as exc:
                logger.warning('VLM image download failed: %s', exc)
                image = None
            wait_seconds += time.perf_counter() - wait_started
            completed += 1
            try:
                # Start the next download before GPU scoring of this image.
                submit_next()
                yield item, image
            finally:
                _close_image(image)
    finally:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        for _, future in pending:
            if not future.cancelled() and future.exception() is None:
                _close_image(future.result())
        logger.info('[VLM Prefetch] workers=%d, images=%d, foreground wait=%.3fs',
                    workers, completed, wait_seconds)
