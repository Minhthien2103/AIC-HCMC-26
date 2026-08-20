"""Optional translation helper.

Translation improves Vietnamese retrieval, but a missing network/package must
not prevent validators or English query packs from starting.
"""

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover - depends on the selected environment
    GoogleTranslator = None


def translate_vi_to_en(text: str) -> str:
    if GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source="vi", target="en").translate(text)
    except Exception:
        return text
