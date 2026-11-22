"""Explicitly reuse completed CPU transcripts, preserving their provenance."""
import hashlib
import io
import json
from pathlib import PurePosixPath

import polars as pl


def reusable_cpu(manifest, signature):
    return (manifest.get('kind') == 'imported_cpu'
            and signature['max_audio_seconds'] == 0
            and all(manifest.get(key) == signature[key] for key in
                    ('source', 'host', 'model', 'window', 'overlap', 'silence_rms')))


def validate_cpu_source(cpu, source, canonical_path, cpu_model_hash):
    return (cpu.get('version') == 1
            and cpu.get('source') == canonical_path
            and cpu.get('size') == source['size']
            and int(cpu.get('mtime', -1)) // 1000000000 == source['mtime']
            and cpu.get('model') == cpu_model_hash
            and cpu.get('window') == 15 and cpu.get('overlap') == 2
            and cpu.get('silence_rms') == 0.001)


def fetch_cpu_checkpoint(ftp, remote_parts, key, source, signature, cpu_model_hash):
    """Manifest is the commit marker; ignore absent, changing or incompatible data."""
    prefix = str(PurePosixPath(remote_parts) / key)
    with ftp.open(prefix + '.json', 'rb') as handle:
        before = handle.read()
    cpu = json.loads(before)
    if not validate_cpu_source(cpu, source, ftp.normalize(source['path']), cpu_model_hash):
        raise ValueError('CPU checkpoint source/model/settings do not match')
    stat_before = ftp.stat(prefix + '.parquet')
    with ftp.open(prefix + '.parquet', 'rb') as handle:
        content = handle.read()
    stat_after = ftp.stat(prefix + '.parquet')
    with ftp.open(prefix + '.json', 'rb') as handle:
        after = handle.read()
    if before != after or (stat_before.st_size, stat_before.st_mtime) != (stat_after.st_size, stat_after.st_mtime):
        raise ValueError('CPU checkpoint changed while downloading')
    table = pl.read_parquet(io.BytesIO(content))
    required = {'video_id', 'start', 'end', 'text'}
    if not required.issubset(table.columns):
        raise ValueError('Invalid transcript schema')
    for row in table.iter_rows(named=True):
        import math
        if (row['video_id'] != source['video_id'] or not isinstance(row['text'], str)
                or not math.isfinite(row['start']) or not math.isfinite(row['end'])
                or not 0 <= row['start'] < row['end']):
            raise ValueError('Invalid transcript row')
    manifest = {key: signature[key] for key in ('source', 'host', 'model', 'window', 'overlap', 'silence_rms')}
    manifest.update(kind='imported_cpu', original_cpu_manifest=cpu,
                    imported_from=prefix, source_parquet_sha256=hashlib.sha256(content).hexdigest())
    return table, manifest
