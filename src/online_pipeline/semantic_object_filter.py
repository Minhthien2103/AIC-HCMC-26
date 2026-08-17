"""
Semantic Object Filter for VQA pipeline - Improvement #9.
Replaces Regex exact-match with spaCy NP extraction + all-MiniLM-L6-v2 cosine similarity.
100% Lazy Loaded: No heavy C++/Torch libraries are imported at startup.
"""
import os
import re
import sys
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config


class SemanticObjectFilter:
    """
    Improvement #9: Semantic NP extraction + MiniLM similarity matching against OpenImages labels.
    Example: "man holding an automobile" -> ["Person", "Car"]
    """
    SBERT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, all_labels: list):
        self.all_labels = all_labels
        self._nlp = None
        self._sbert = None
        self._label_embeddings = None
        self._ready = False

    def _setup(self):
        if self._ready:
            return
        try:
            print("[SemanticFilter] Lazy-loading spaCy and SentenceTransformer on demand...")
            import spacy
            from sentence_transformers import SentenceTransformer

            self._nlp = spacy.load("en_core_web_sm")
            self._sbert = SentenceTransformer(self.SBERT_MODEL, device="cpu")
            self._label_embeddings = self._sbert.encode(
                self.all_labels, normalize_embeddings=True, show_progress_bar=False
            )
            self._ready = True
            print(f"[SemanticFilter] Ready. {len(self.all_labels)} OpenImages labels cached on CPU.")
        except Exception as e:
            print(f"[SemanticFilter] Setup failed: {e}. Falling back to Regex.")
            self._ready = False

    def extract_objects(self, english_query: str) -> list:
        """
        Returns OpenImages labels that semantically match noun phrases in the query.
        Returns [] if no concrete objects found (triggers w=0 dynamic weight).
        """
        if not self._ready:
            self._setup()

        if not self._ready or self._nlp is None or self._sbert is None:
            return self._regex_fallback(english_query)

        doc = self._nlp(english_query)
        phrases = list(set(
            [chunk.text.lower() for chunk in doc.noun_chunks]
            + [token.text.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")]
        ))

        if not phrases:
            return []

        phrase_embeddings = self._sbert.encode(
            phrases, normalize_embeddings=True, show_progress_bar=False
        )

        matched = set()
        for vec in phrase_embeddings:
            sims = np.dot(self._label_embeddings, vec)
            top_idxs = np.argsort(sims)[::-1][:config.SEMANTIC_OBJ_TOP_K]
            for idx in top_idxs:
                if sims[idx] >= config.SEMANTIC_OBJ_THRESHOLD:
                    matched.add(self.all_labels[idx])

        return list(matched)

    def _regex_fallback(self, query: str) -> list:
        query_lower = query.lower()
        matched = []
        for label in self.all_labels:
            pattern = r'\b' + re.escape(label.lower()) + r'(?:s|es)?\b'
            if re.search(pattern, query_lower):
                matched.append(label)
        return matched
