"""Manual transcript matching, restricted to existing visual candidates."""
import math
import re
import unicodedata
from pathlib import Path

import polars as pl

from src.online_pipeline.score_fusion import normalized_score


def normalize_text(text):
    text = unicodedata.normalize('NFC', str(text)).casefold()
    return ' '.join(re.findall(r'\w+', text, flags=re.UNICODE))


class ASRSearchEngine:
    def __init__(self, path):
        self.path = Path(path)
        self.signature = None
        self.rows = []
        self.warning = ''

    def reload_if_changed(self):
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self.signature:
                return
            table = pl.read_parquet(self.path)
            required = {'video_id', 'start', 'end', 'text'}
            if not required.issubset(table.columns):
                raise ValueError('Thiếu cột video_id/start/end/text')
            rows = []
            for row in table.select(sorted(required)).iter_rows(named=True):
                start, end = float(row['start']), float(row['end'])
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    raise ValueError('Timestamp ASR không hợp lệ')
                rows.append((str(row['video_id']), start, end, normalize_text(row['text']), str(row['text'])))
            self.rows, self.signature, self.warning = rows, signature, ''
        except FileNotFoundError:
            self.rows, self.signature = [], None
            self.warning = 'Chưa có transcript ASR. Chạy scripts/extract_asr.py trước; kết quả hiện tại chưa dùng ASR.'
        except Exception as exc:
            self.rows, self.signature = [], None
            self.warning = f'Không đọc được transcript ASR ({type(exc).__name__}); kết quả hiện tại chưa dùng ASR.'

    def boost(self, results, manual_text, weight=0.2):
        if not manual_text.strip():
            return results, ''
        self.reload_if_changed()
        if self.warning:
            return results, self.warning
        phrases = list(dict.fromkeys(filter(None, (normalize_text(p) for p in manual_text.split(',')))))
        matches = {}
        videos = {str(row.get('entity', {}).get('video', '')) for row in results}
        for video, start, end, normalized, text in self.rows:
            if video not in videos:
                continue
            for phrase in phrases:
                if f' {phrase} ' in f' {normalized} ':
                    matches.setdefault(video, []).append((start, end, phrase, text))
                    break
        hits = {}
        for i, row in enumerate(results):
            entity = row.get('entity', {})
            timestamp = entity.get('time')
            if timestamp is None:
                continue
            for start, end, phrase, text in matches.get(str(entity.get('video', '')), []):
                if start <= float(timestamp) < end:
                    hits[i] = dict(keyword=phrase, text=text, start=start, end=end)
                    break
        if not hits:
            return results, 'Không có frame ứng viên khớp ASR; giữ nguyên thứ hạng.'
        weight = max(0.0, float(weight))
        boosted = []
        for i, row in enumerate(results):
            copy = dict(row, entity=dict(row['entity']))
            score = (normalized_score(row) + weight * (i in hits)) / (1 + weight)
            copy.update(score=score, distance=score, score_space='unit')
            if i in hits:
                copy['entity']['asr_match'] = hits[i]
            boosted.append(copy)
        return sorted(boosted, key=lambda row: -row['score']), ''
