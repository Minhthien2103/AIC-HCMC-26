"""Translation instructions and strict output parsing, independent of model loading."""
import json
import re
import unicodedata


def translation_messages(query):
    normalized = unicodedata.normalize('NFC', query).casefold()
    hints = []
    # Supply only relevant definitions: unrelated examples can leak into output.
    if re.search(r'\b(con lân|múa lân|đầu lân)\b', normalized):
        hints.append('Vietnamese cultural term: con lân is a lion-dance costume/figure (qilin); múa lân is lion dancing; đầu lân is a lion-dance head. It is not a deer, horse or monkey. Do not add dancing or performers unless the input says so.')
    if 'kỳ lân' in normalized or 'kì lân' in normalized:
        hints.append('Interpret kỳ lân in context: a one-horned animal in a cartoon is a unicorn, not a lion-dance costume.')
    if 'áo dài' in normalized:
        hints.append('Áo dài is the Vietnamese ao dai dress. Do not infer the wearer is a woman unless stated.')
    if 'nón lá' in normalized:
        hints.append('Nón lá means Vietnamese conical hat. Preserve holding versus wearing.')
    system = (
        'Translate INTO ENGLISH ONLY. Never translate English into Vietnamese. '
        'You are a translator, not a scene description generator. '
        'Identify the SOURCE language and translate its exact meaning into natural English. '
        'Preserve subjects, colors, counts, actions, negation, spatial relationships and who does what. '
        'Do not add settings, objects, gender or actions. Translate idioms in context, not word by word. '
        'For English input, copy the input exactly; do not correct names or unfamiliar words. '
        'Return exactly one JSON object: {"source_language":"vi or en or other","translation":"English text"}. '
        'source_language identifies the USER INPUT language, not the output language. '
        'If input is English, source_language MUST be en and translation MUST equal the input.'
    )
    if hints:
        system += '\nRelevant terminology:\n' + '\n'.join(hints)
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Source text (translate INTO ENGLISH):\n' + query}]


def parse_json_output(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


def parse_translation(text, original):
    try:
        value = parse_json_output(text)
    except ValueError:
        # Some valid translations are emitted as a plain sentence despite the
        # JSON instruction. Accept one bare line, not broken JSON or explanations.
        plain = text.strip()
        if (not plain or '\n' in plain or any(c in plain for c in '{}[]<>:')
                or not re.search(r'[A-Za-z]', plain)):
            raise ValueError('Malformed translation output')
        return normalize_cultural_terms(plain, original)
    if not isinstance(value, dict) or value.get('source_language') not in ('vi', 'en', 'other'):
        raise ValueError('Invalid translation language/schema')
    translated = value.get('translation')
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError('Empty or invalid translation')
    # Even if the model rewrites a proper name, the English primary is preserved.
    return original.strip() if value['source_language'] == 'en' else normalize_cultural_terms(translated.strip(), original)


def normalize_cultural_terms(translated, original):
    if re.search(r'\b(con lân|múa lân|đầu lân)\b', original.casefold()):
        # Keep the established English term when the model leaves this Vietnamese
        # word inside an otherwise English translation; never replace "kỳ lân".
        translated = re.sub(r'(?<!kỳ )(?<!kì )\blân\b', 'lion-dance', translated, flags=re.IGNORECASE)
    return translated


def expansion_messages(primary):
    return [
        {'role': 'system', 'content': (
            'Create exactly two concise English paraphrases of the supplied image-search query. '
            'Preserve EVERY subject, color, count, action, negation and relationship. '
            'Do not guess gender, setting or objects. Do not replace unfamiliar names. '
            'When unsure, repeat the input. Return exactly ONE JSON object with two string values: '
            '{"paraphrase1":"first English paraphrase","paraphrase2":"second English paraphrase"}. '
            'Do not return arrays, multiple JSON objects, or explanations.'
        )},
        {'role': 'user', 'content': primary},
    ]


def parse_expansions(text, primary):
    values = parse_json_output(text)
    if isinstance(values, dict):
        values = [values.get('paraphrase1'), values.get('paraphrase2')]
    if not isinstance(values, list) or len(values) != 2 or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError('Expected two nonempty paraphrases')
    # The authoritative translation is never generated again by the expansion step.
    return [primary] + [v.strip() for v in values]
