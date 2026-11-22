"""Remote audio prefetch + local CUDA batches. Run manually; resumes per video."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import threading
import time


def atomic_json(value, path):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def connect(args, password):
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / '.ssh/known_hosts'))
    client.connect(args.host, username=args.user, password=password, timeout=10,
                   auth_timeout=15, banner_timeout=15, look_for_keys=not password,
                   allow_agent=not password)
    return client


def download(args, password, source, destination, stop):
    """Stream only audio to disk, drain stderr, and stop bounded prefetch on Ctrl+C."""
    client = connect(args, password)
    channel = None
    success = False
    try:
        channel = client.get_transport().open_session(timeout=15)
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', source['path']]
        if args.max_audio_seconds:
            command += ['-t', str(args.max_audio_seconds)]
        command += ['-vn', '-map', '0:a:0', '-ac', '1', '-ar', '16000', '-f', 'wav', 'pipe:1']
        channel.exec_command(shlex.join(command))
        last_data, size, errors = time.monotonic(), 0, b''
        with destination.open('wb') as handle:
            while True:
                if stop.is_set():
                    raise InterruptedError('Download cancelled')
                active = False
                if channel.recv_ready():
                    chunk = channel.recv(1024 * 1024)
                    handle.write(chunk)
                    size += len(chunk)
                    if size > args.max_audio_mb * 1024 * 1024:
                        raise RuntimeError('Audio exceeds --max-audio-mb')
                    active = True
                if channel.recv_stderr_ready():
                    errors = (errors + channel.recv_stderr(65536))[-8000:]
                    active = True
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                    break
                if active:
                    last_data = time.monotonic()
                elif time.monotonic() - last_data > 120:
                    raise TimeoutError('No SSH audio data for 120 seconds')
                else:
                    time.sleep(0.01)
        if channel.recv_exit_status() or size < 44:
            raise RuntimeError(errors.decode('utf-8', errors='replace') or 'Empty audio')
        # Do not cache a transcript if the source changed while it was being read.
        with client.open_sftp() as ftp:
            current = ftp.stat(source['path'])
            if (current.st_size, current.st_mtime) != (source['size'], source['mtime']):
                raise RuntimeError('Source video changed during download')
        success = True
        return destination
    finally:
        if channel is not None:
            channel.close()
        client.close()
        if not success:
            destination.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='192.168.28.112')
    p.add_argument('--user', default='team12')
    p.add_argument('--remote-dir', default='/home/team12/aic-data/videos')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path, required=True)
    p.add_argument('--ssh-runtime-dir', type=Path)
    p.add_argument('--dll-dir', type=Path)
    p.add_argument('--output', type=Path, default=Path('indexes/asr_segments.parquet'))
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--video-id', action='append', help='Only selected IDs; repeat for several videos')
    p.add_argument('--max-audio-seconds', type=int, default=0, help='Trial only; requires separate output')
    p.add_argument('--max-audio-mb', type=int, default=512)
    p.add_argument('--key-auth', action='store_true', help='Use SSH agent/keys instead of password prompt')
    p.add_argument('--import-cpu-parts', help='Import completed CPU checkpoints from this remote directory')
    p.add_argument('--prepare-only', action='store_true', help='Validate/import checkpoints without downloading audio or decoding')
    args = p.parse_args()
    if not 1 <= args.batch_size <= 32 or args.threads < 1 or args.max_audio_seconds < 0 or args.max_audio_mb < 1:
        p.error('Invalid batch size / threads / audio limit')
    if (args.max_audio_seconds or args.video_id) and args.output.resolve() == Path('indexes/asr_segments.parquet').resolve():
        p.error('Trial/subset audio must use a separate --output')
    if args.ssh_runtime_dir:
        sys.path.insert(0, str(args.ssh_runtime_dir.resolve()))
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    dll_handle = None
    if args.dll_dir and os.name == 'nt':
        dll_handle = os.add_dll_directory(str(args.dll_dir.resolve()))
        os.environ['PATH'] = str(args.dll_dir.resolve()) + os.pathsep + os.environ['PATH']
    from filelock import FileLock
    import polars as pl
    import sherpa_onnx
    from extract_asr import atomic_parquet, transcribe_audio
    from asr_checkpoint_import import fetch_cpu_checkpoint, reusable_cpu
    if 'cuda' not in sherpa_onnx.__version__:
        raise RuntimeError('Requires the tested CUDA Sherpa wheel; CPU-only wheel refused')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output) + '.lock', timeout=0):
        password = None if args.key_auth else getpass.getpass('SSH password (not saved): ')
        cache = args.output.parent / (args.output.stem + '_gpu_parts')
        cache.mkdir(exist_ok=True)
        audio_dir = cache / 'audio'
        audio_dir.mkdir(exist_ok=True)
        files, model_hash = {}, hashlib.sha256()
        for name, pattern in [('encoder', 'encoder*.onnx'), ('decoder', 'decoder*.onnx'), ('joiner', 'joiner*.onnx'), ('tokens', 'tokens.txt')]:
            found = list(args.model_dir.glob(pattern))
            if len(found) != 1:
                raise ValueError(f'Expected exactly one {pattern}')
            files[name] = str(found[0])
            model_hash.update(name.encode())
            with found[0].open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    model_hash.update(block)
        cpu_model_hash = hashlib.sha256()
        for name, filename in sorted(files.items()):
            cpu_model_hash.update(name.encode())
            with open(filename, 'rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    cpu_model_hash.update(block)
        with connect(args, password) as client, client.open_sftp() as ftp:
            sources = []
            for name in sorted(ftp.listdir(args.remote_dir)):
                path = PurePosixPath(args.remote_dir) / name
                if path.suffix.lower() not in {'.mp4', '.mkv', '.avi', '.webm'}:
                    continue
                if args.video_id and path.stem not in args.video_id:
                    continue
                stat = ftp.stat(str(path))  # Follow video symlinks on school server.
                sources.append(dict(video_id=path.stem, path=str(path), size=stat.st_size, mtime=stat.st_mtime))
        ids = [s['video_id'] for s in sources]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('No videos or duplicate video IDs')
        if args.video_id and set(args.video_id) - set(ids):
            raise ValueError('Requested video IDs not found')
        completed, pending, failures = [], [], []
        imported_count = 0
        import_client = connect(args, password) if args.import_cpu_parts else None
        import_ftp = import_client.open_sftp() if import_client else None
        remote_manifests = set(import_ftp.listdir(args.import_cpu_parts)) if import_ftp else set()
        for source in sources:
            key = hashlib.sha256(source['video_id'].encode()).hexdigest()
            shard, manifest = cache / f'{key}.parquet', cache / f'{key}.json'
            signature = dict(source=source, host=args.host, model=model_hash.hexdigest(),
                             provider='cuda', runtime=sherpa_onnx.__version__, batch=args.batch_size,
                             window=15, overlap=2, silence_rms=0.001,
                             max_audio_seconds=args.max_audio_seconds, version=1)
            try:
                saved = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
                if shard.exists() and (saved == signature or reusable_cpu(saved, signature)):
                    completed.append(pl.read_parquet(shard))
                    print(f'RESUME {source["video_id"]} ({saved.get("kind", "cuda")})', flush=True)
                    continue
            except (OSError, ValueError, pl.exceptions.PolarsError):
                pass
            if import_ftp and key + '.json' in remote_manifests and not args.max_audio_seconds:
                try:
                    table, imported = fetch_cpu_checkpoint(import_ftp, args.import_cpu_parts, key,
                                                           source, signature, cpu_model_hash.hexdigest())
                    atomic_parquet(table, shard)
                    atomic_json(imported, manifest)
                    completed.append(table)
                    imported_count += 1
                    print(f'IMPORT CPU {source["video_id"]}: {len(table)} segments', flush=True)
                    continue
                except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
                    print(f'SKIP CPU {source["video_id"]}: {exc}', flush=True)
            pending.append((source, shard, manifest, signature, audio_dir / f'{key}.wav'))
        if import_ftp:
            import_ftp.close()
            import_client.close()
        def publish(status):
            if completed:
                atomic_parquet(pl.concat(completed), args.output)
            atomic_json(dict(status=status, total=len(sources), completed=len(completed),
                             failures=failures), args.output.with_suffix('.progress.json'))
        if args.prepare_only:
            publish('ready' if pending else 'complete')
            print(f'PREPARED total={len(sources)} reusable={len(completed)} imported={imported_count} pending={len(pending)}', flush=True)
            return 0
        publish('running')
        stop, queue = threading.Event(), deque()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            if pending:
                recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    **files, num_threads=args.threads, sample_rate=16000, feature_dim=80,
                    decoding_method='modified_beam_search', max_active_paths=4, provider='cuda')
            iterator = iter(pending)
            def schedule():
                item = next(iterator, None)
                if item is not None:
                    queue.append((item, pool.submit(download, args, password, item[0], item[4], stop)))
            schedule()
            schedule()  # At most current audio plus one prefetched audio.
            while queue:
                item, future = queue.popleft()
                source, shard, manifest, signature, audio = item
                started = time.monotonic()
                try:
                    future.result()
                    table = transcribe_audio(recognizer, audio, source['video_id'], batch_size=args.batch_size)
                    atomic_parquet(table, shard)
                    atomic_json(signature, manifest)
                    completed.append(table)
                    print(f'OK {source["video_id"]}: {len(table)} segments; wait+decode {time.monotonic()-started:.1f}s', flush=True)
                except Exception as exc:
                    failures.append(dict(video_id=source['video_id'], error=str(exc)))
                    print(f'FAILED {source["video_id"]}: {exc}', flush=True)
                    if any(marker in str(exc).lower() for marker in ('cuda', 'out of memory', 'cublas', 'cudnn')):
                        raise RuntimeError('GPU failure; stop and resume with a smaller batch if needed') from exc
                finally:
                    if not future.done():
                        stop.set()
                        try:
                            future.result()
                        except Exception:
                            pass
                    audio.unlink(missing_ok=True)
                publish('running')
                schedule()
            publish('complete' if not failures else 'partial')
        except BaseException:
            stop.set()
            publish('interrupted')
            raise
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            for item in pending:
                item[4].unlink(missing_ok=True)
            if dll_handle:
                dll_handle.close()
        print(f'Completed {len(completed)}/{len(sources)}; output={args.output}', flush=True)
        return int(bool(failures))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Stopped. Finished video checkpoints preserved.', flush=True)
        raise SystemExit(130)
