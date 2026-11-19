"""Auditable query preparation without GPU dependencies."""
import logging

from src.utils.translation import translate_vi_to_en

logger = logging.getLogger(__name__)


class QueryPreparationError(ValueError):
    pass


def prepare_english(query, expander):
    """Use main's optional expansion and translation fallback for all queries."""
    query = query.strip()
    if not query:
        return {"input": query, "source": "empty", "english": [], "warnings": []}

    variants = []
    source = "local_llm"
    if expander is not None:
        try:
            variants = expander.expand(query)
        except Exception as error:
            logger.warning("Query expansion failed for %r: %s", query, error)
    if not variants:
        source = "translator"
        try:
            translated = translate_vi_to_en(query)
            variants = [translated.strip()] if translated and translated.strip() else [query]
        except Exception as error:
            logger.warning("Translation fallback for %r: %s", query, error)
            variants = [query]
        if variants == [query]:
            source = "original"
    variants = [value.strip() for value in variants if value.strip()][:3]
    return {"input": query, "source": source, "english": variants, "warnings": []}


def split_for_encoder(text, count_tokens, limit):
    """Preserve every word, including the last constraint, without truncation."""
    if count_tokens(text) <= limit:
        return [text]
    chunks, current = [], ""
    for word in text.split():
        if count_tokens(word) > limit:
            raise QueryPreparationError("Một từ trong truy vấn vượt giới hạn model. Hãy rút gọn từ đó.")
        candidate = f"{current} {word}".strip()
        if current and count_tokens(candidate) > limit:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def prepare_chunks(plan, encoder, branch_weights):
    chunks, weights = [], []
    for index, text in enumerate(plan["english"]):
        parts = split_for_encoder(text, encoder.text_token_count, encoder.text_context_length)
        chunks.extend(parts)
        weights.extend([float(branch_weights[index]) / len(parts)] * len(parts))
    warnings = list(plan["warnings"])
    if len(chunks) > len(plan["english"]):
        warnings.append("Mô tả dài đã được chia thành nhiều đoạn để giữ cả chi tiết cuối câu.")
    return dict(plan, chunks=chunks, weights=weights, warnings=warnings,
                token_counts=[encoder.text_token_count(text) for text in chunks],
                token_limit=encoder.text_context_length)
