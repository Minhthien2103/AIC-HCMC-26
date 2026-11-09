"""Reproducible Vietnamese-to-English translation for retrieval.

The submission runner configures a local mBART translator with a
content-addressed cache.  The old network helper remains only as a compatible
default for the exploratory Streamlit app.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover - depends on the selected environment
    GoogleTranslator = None


class _Translator(Protocol):
    def translate(self, text: str) -> str: ...


class _GoogleFallback:
    def translate(self, text: str) -> str:
        if GoogleTranslator is None:
            return text
        try:
            return GoogleTranslator(source="vi", target="en").translate(text)
        except Exception:
            return text


class MBartCachedTranslator:
    """Lazy local translator whose final-run behaviour never uses the network."""

    def __init__(
        self,
        cache_path: str | Path,
        model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = "cuda",
        *,
        strict: bool = True,
    ):
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.device = device
        self.strict = strict
        self._cache: dict[str, str] | None = None
        self._tokenizer = None
        self._model = None

    def _load_cache(self) -> dict[str, str]:
        if self._cache is None:
            if self.cache_path.exists():
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(f"Translation cache must be a JSON object: {self.cache_path}")
                self._cache = {str(key): str(value) for key, value in payload.items()}
            else:
                self._cache = {}
        return self._cache

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(self._load_cache(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.cache_path)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            resolved_device = self.device if self.device == "cuda" and torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
            self._model.to(resolved_device)
            self._model.eval()
            self.device = resolved_device
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    "Offline translation requires cached mBART weights. Run asset preparation before final generation."
                ) from exc
            self._model = False

    def translate(self, text: str) -> str:
        normalized = str(text).strip()
        if not normalized:
            return normalized
        cache = self._load_cache()
        key = self._key(normalized)
        if key in cache:
            return cache[key]

        self._load_model()
        if self._model is False:
            return normalized

        import torch

        self._tokenizer.src_lang = "vi_VN"
        inputs = self._tokenizer(normalized, return_tensors="pt", truncation=True, max_length=512)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        forced_bos = self._tokenizer.lang_code_to_id["en_XX"]
        with torch.inference_mode():
            output = self._model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=256)
        translated = self._tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        if not translated:
            if self.strict:
                raise RuntimeError("Offline translator returned an empty result")
            return normalized
        cache[key] = translated
        self._save_cache()
        return translated


_translator: _Translator = _GoogleFallback()


def configure_translation(
    *,
    cache_path: str | Path | None = None,
    offline: bool = False,
    device: str = "cuda",
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
) -> None:
    """Configure process-wide translation before constructing submission tasks."""
    global _translator
    if offline:
        if cache_path is None:
            raise ValueError("offline translation requires a cache path")
        _translator = MBartCachedTranslator(cache_path, model_name=model_name, device=device, strict=True)
    elif cache_path is not None:
        _translator = MBartCachedTranslator(cache_path, model_name=model_name, device=device, strict=False)
    else:
        _translator = _GoogleFallback()


def translate_vi_to_en(text: str) -> str:
    return _translator.translate(text)
