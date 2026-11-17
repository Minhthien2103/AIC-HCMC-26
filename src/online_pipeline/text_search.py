"""OCR Text Search & Entity Extraction Engine for AIC-HCMC 2026.

Scans indexes/ocr_text.parquet using deterministic rules to prevent LLM hallucinations:
- Extracts numbers (115, 60, 08...), license plates (51F, 59A...), and quoted strings ("...").
- Matches against pre-extracted OCR text table using Polars string expressions.
- Returns boost bonuses and matched text for relevant keyframes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OCR_PATH = REPO_ROOT / "indexes" / "ocr_text.parquet"


class TextSearchEngine:
    """Fast in-memory OCR text searcher using Polars."""

    def __init__(self, ocr_parquet_path: Path = DEFAULT_OCR_PATH):
        self.ocr_path = Path(ocr_parquet_path)
        self.df: Optional[pl.DataFrame] = None
        self._load_table()

    def _load_table(self):
        """Load or reload the OCR table if it exists."""
        if self.ocr_path.is_file():
            try:
                self.df = pl.read_parquet(self.ocr_path)
                logger.info(f"Loaded OCR database: {len(self.df):,} records from {self.ocr_path}")
            except Exception as e:
                logger.warning(f"Failed to load OCR parquet {self.ocr_path}: {e}")
                self.df = None
        else:
            logger.info(f"OCR parquet not found at {self.ocr_path} (extraction may be in progress).")
            self.df = None

    def reload(self):
        """Force reload from disk (e.g., after extraction completes)."""
        self._load_table()

    @staticmethod
    def extract_deterministic_keywords(query: str) -> List[str]:
        """Extract hard keywords/entities that require OCR verification without hallucination.

        Rules:
        1. Quoted terms: "cơm tấm", "chợ rẫy"
        2. Numbers: 115, 60, 08, 19, 2024, etc.
        3. License plates / alphanumeric codes: 51F, 59A-123.45, VTV9, HTV7
        4. Explicit OCR terms: words directly following 'biển số', 'số', 'in chữ', 'biển hiệu'
        """
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        keywords: Set[str] = set()

        # 1. Quoted terms: "..."
        quoted = re.findall(r'["\']([^"\']{2,})["\']', clean_q)
        for term in quoted:
            keywords.add(term.strip())

        # 2. License plates / alphanumeric codes (e.g., 51F, 51F-1234, 59A-892.45, VTV9, HTV7)
        plates = re.findall(r'\b\d{2}[A-Za-z][\s\-_]?\d{3,5}(?:\.\d{2})?\b', clean_q)
        for p in plates:
            keywords.add(p.strip())

        # 3. Standalone significant numbers (2+ digits, e.g. 115, 114, 113, 60, 08, 19, 24)
        numbers = re.findall(r'\b\d{2,}\b', clean_q)
        for n in numbers:
            keywords.add(n.strip())

        # 4. Explicit prefix patterns (e.g., "biển số 51F", "số áo 10", "chữ SÀI GÒN")
        explicit_matches = re.findall(
            r'(?:biển\s*số|số\s*hiệu|số\s*áo|in\s*chữ|biển\s*hiệu|tên\s*là)\s+([A-Za-z0-9\s\-]+?)(?=[,;\.]|$)',
            clean_q,
            flags=re.IGNORECASE
        )
        for em in explicit_matches:
            cleaned = em.strip()
            if len(cleaned) >= 2 and len(cleaned.split()) <= 4:
                keywords.add(cleaned)

        return sorted(list(keywords), key=lambda x: len(x), reverse=True)

    def search_candidates(
        self,
        keywords: List[str],
        boost_score: float = 0.35
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Search OCR database for keyframes containing any of the keywords.

        Returns a dictionary mapping (video_id, keyframe_name) -> {
            'boost': boost_score,
            'matched_text': matched_snippet,
            'matched_keyword': keyword
        }
        """
        if not keywords or self.df is None:
            # Check if table appeared on disk
            if self.df is None and self.ocr_path.is_file():
                self._load_table()
            if self.df is None:
                return {}

        results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for kw in keywords:
            clean_kw = kw.strip()
            if len(clean_kw) < 2:
                continue

            # Case-insensitive substring match in polars
            pattern = f"(?i){re.escape(clean_kw)}"
            try:
                matched_df = self.df.filter(pl.col("ocr_text").str.contains(pattern))
                if len(matched_df) == 0:
                    continue

                for row in matched_df.iter_rows(named=True):
                    v_id = row["video_id"]
                    k_name = str(row["keyframe_name"])
                    key = (v_id, k_name)

                    if key not in results:
                        results[key] = {
                            "boost": boost_score,
                            "matched_text": row["ocr_text"][:80],
                            "matched_keyword": clean_kw,
                            "pts_time": row["pts_time"],
                            "frame_id": row["frame_id"]
                        }
            except Exception as e:
                logger.warning(f"Regex error while searching OCR for '{clean_kw}': {e}")
                continue

        return results
