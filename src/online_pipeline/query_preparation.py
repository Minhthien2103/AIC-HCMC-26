"""Auditable query preparation without GPU dependencies."""
import re
import unicodedata


class QueryPreparationError(ValueError):
    pass


def has_vietnamese_marks(text):
    text = unicodedata.normalize("NFC", text)
    return bool(re.search(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", text, re.I))


def prepare_english(query, expander):
    query = " ".join(query.split())
    if not query:
        return {"input": query, "source": "empty", "english": [], "warnings": []}
    if not has_vietnamese_marks(query):
        return {"input": query, "source": "original", "english": [query], "warnings": []}

    if expander is None or getattr(expander, "model", None) is None:
        raise QueryPreparationError(
            "Mô hình dịch local chưa sẵn sàng. Hãy nhập tiếng Anh để tìm, hoặc cài đủ trọng số mô hình rồi khởi động lại backend (--llm 3b). Không gửi truy vấn tới dịch vụ ngoài.")
    try:
        variants = [str(value).strip() for value in expander.expand(query) if str(value).strip()]
    except Exception as error:
        raise QueryPreparationError("Mô hình local không dịch được truy vấn. Hãy dùng bản tiếng Anh và kiểm tra log backend.") from error
    if not variants or has_vietnamese_marks(variants[0]) or variants[0].casefold() == query.casefold():
        raise QueryPreparationError("Mô hình local chưa trả bản tiếng Anh hợp lệ. Đã dừng tìm kiếm; hãy sửa bằng bản tiếng Anh để tránh kết quả lệch.")
    english = list(dict.fromkeys(v for v in variants[:3] if not has_vietnamese_marks(v)))
    return {"input": query, "source": "local_llm", "english": english, "warnings": []}


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
