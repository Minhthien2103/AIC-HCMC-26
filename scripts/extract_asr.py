"""CPU Zipformer ONNX transcription; bounded audio memory and per-video resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import polars as pl

SCHEMA = {'video_id': pl.String, 'start': pl.Float64, 'end': pl.Float64, 'text': pl.String}
RATE = 16000


def atomic_parquet(table, path):
    temporary = path.with_suffix('.tmp.parquet')
    table.write_parquet(temporary)
    os.replace(temporary, path)


def audio_windows(path, seconds, overlap, ffmpeg='ffmpeg'):
    size, keep = int(seconds * RATE) * 2, int(overlap * RATE) * 2
    # Stderr goes to a temporary file so a full stderr pipe cannot deadlock decoding.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([ffmpeg, '-nostdin', '-v', 'error', '-i', str(path),
                                    '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(RATE),
                                    '-f', 's16le', 'pipe:1'], stdout=subprocess.PIPE, stderr=errors)
        pending, offset = b'', 0
        try:
            while True:
                fresh = process.stdout.read(size - len(pending))
                if not fresh:
                    break
                pending += fresh
                samples = np.frombuffer(pending, dtype='<i2').astype(np.float32) / 32768.0
                yield offset / RATE, samples
                if len(pending) < size:
                    break
                offset += (size - keep) // 2
                pending = pending[-keep:] if keep else b''
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read(4000).decode('utf-8', errors='replace'))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


def transcribe_audio(recognizer, path, video_id, window=15, overlap=2,
                     silence_rms=0.001, batch_size=1, ffmpeg='ffmpeg'):
    """Bounded batches retain absolute timestamps, including the final partial batch."""
    rows, batch = [], []

    def flush():
        streams = [recognizer.create_stream() for _ in batch]
        for stream, (_, samples) in zip(streams, batch):
            stream.accept_waveform(RATE, samples)
        recognizer.decode_streams(streams)
        for stream, (start, samples) in zip(streams, batch):
            text = stream.result.text.strip()
            if text:
                rows.append(dict(video_id=video_id, start=start,
                                 end=start + len(samples) / RATE, text=text))
        batch.clear()

    windows = audio_windows(path, window, overlap, ffmpeg)
    try:
        for start, samples in windows:
            if not len(samples) or float(np.sqrt(np.mean(samples ** 2))) < silence_rms:
                continue
            batch.append((start, samples))
            if len(batch) == batch_size:
                flush()
        if batch:
            flush()
    finally:
        windows.close()
    return pl.DataFrame(rows, schema=SCHEMA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True, help='ONNX directory, not the .pt checkpoint')
    parser.add_argument('--output', type=Path, default=Path('indexes/asr_segments.parquet'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--window', type=float, default=15)
    parser.add_argument('--overlap', type=float, default=2)
    parser.add_argument('--silence-rms', type=float, default=0.001)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; use a separate --output')
    args = parser.parse_args()
    if not (1 <= args.window <= 60 and 0 <= args.overlap < args.window and args.threads > 0 and args.silence_rms >= 0 and args.limit >= 0):
        parser.error('Invalid window/overlap/threads/silence-rms')
    videos = sorted(p for p in args.videos_dir.rglob('*') if p.suffix.lower() in {'.mp4', '.mkv', '.avi', '.webm', '.wav', '.mp3'} and p.is_file())
    if not videos:
        parser.error('No video/audio files found')
    if len({p.stem for p in videos}) != len(videos):
        parser.error('Duplicate video stems: video_id must be unique')
    if args.limit:
        videos = videos[:args.limit]
    files = {}
    for name, pattern in [('encoder', 'encoder*.onnx'), ('decoder', 'decoder*.onnx'), ('joiner', 'joiner*.onnx'), ('tokens', 'tokens.txt')]:
        found = sorted(args.model_dir.glob(pattern))
        if len(found) != 1:
            parser.error(f'Expected exactly one {pattern} in model-dir')
        files[name] = found[0]
    # Hash model contents once; old transcripts are not reused after model changes.
    model_hash = hashlib.sha256()
    for name, path in sorted(files.items()):
        model_hash.update(name.encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                model_hash.update(block)
    import sherpa_onnx
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        **{key: str(path) for key, path in files.items()}, num_threads=args.threads,
        sample_rate=RATE, feature_dim=80, decoding_method='modified_beam_search', max_active_paths=4,
        provider='cpu')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / (args.output.stem + '_parts')
    cache.mkdir(exist_ok=True)
    completed, failures = [], []
    for video in videos:
        key = hashlib.sha256(video.stem.encode()).hexdigest()
        shard, manifest = cache / f'{key}.parquet', cache / f'{key}.json'
        stat = video.stat()
        signature = dict(source=str(video.resolve()), size=stat.st_size, mtime=stat.st_mtime_ns,
                         model=model_hash.hexdigest(), window=args.window, overlap=args.overlap,
                         silence_rms=args.silence_rms, version=1)
        try:
            if shard.exists() and manifest.exists() and json.loads(manifest.read_text()) == signature:
                completed.append(pl.read_parquet(shard))
                print(f'RESUME {video.stem}', flush=True)
                continue
            table = transcribe_audio(recognizer, video, video.stem, args.window,
                                     args.overlap, args.silence_rms, ffmpeg=args.ffmpeg)
            atomic_parquet(table, shard)
            temp = manifest.with_suffix('.tmp.json')
            temp.write_text(json.dumps(signature), encoding='utf-8')
            os.replace(temp, manifest)
            completed.append(table)
            print(f'OK {video.stem}: {len(table)} segments', flush=True)
        except Exception as exc:
            failures.append({'video': str(video), 'error': str(exc)})
            print(f'FAILED {video.stem}: {exc}', flush=True)
    if completed:
        atomic_parquet(pl.concat(completed), args.output)
    args.output.with_suffix('.errors.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed {len(completed)}/{len(videos)}; output={args.output}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
