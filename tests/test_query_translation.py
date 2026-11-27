import ast
import logging
from pathlib import Path
import unittest

from src.online_pipeline.query_translation import (
    translation_messages, parse_translation, expansion_messages, parse_expansions,
)


class TranslationTests(unittest.TestCase):
    def test_english_input_is_preserved_even_if_model_rewrites_name(self):
        self.assertEqual(parse_translation('{"source_language":"en","translation":"A white quail"}',
                                           'A white quilin'), 'A white quilin')

    def test_context_is_only_injected_for_relevant_terms(self):
        plain = translation_messages('Một con ngựa trắng')[0]['content']
        self.assertNotIn('qilin', plain)
        self.assertNotIn('cyclist', plain)
        self.assertIn('lion-dance', translation_messages('Con lân màu trắng')[0]['content'])
        self.assertNotIn('Relevant terminology', translation_messages('Một ngôi nhà ở vùng lân cận')[0]['content'])
        self.assertIn('unicorn', translation_messages('Kỳ lân một sừng')[0]['content'])
        self.assertEqual(parse_translation('The white lân costume is dancing', 'Con lân màu trắng đang múa'),
                         'The white lion-dance costume is dancing')

    def test_primary_is_immutable_and_invalid_paraphrases_are_rejected(self):
        primary = 'A white lion-dance costume'
        prompt = expansion_messages(primary)[0]['content']
        self.assertIn('SigLIP', prompt)
        self.assertIn('paraphrase1', prompt)
        self.assertIn('paraphrase2', prompt)
        self.assertEqual(parse_expansions('["variant1", "variant2"]', primary)[0], primary)
        self.assertEqual(parse_expansions('{"paraphrase1":"one","paraphrase2":"two"}', primary),
                         [primary, 'one', 'two'])
        for response in ('["only one"]', '{"query":"bad"}', '[1, "text"]', '["", "text"]'):
            with self.assertRaises(ValueError):
                parse_expansions(response, primary)

    def test_invalid_translation_is_not_used(self):
        self.assertEqual(parse_translation('Bike rider wearing a white shirt', 'Tay đua xe đạp mặc áo trắng'),
                         'Bike rider wearing a white shirt')
        for response in ('Explanation:\nHere is the answer', '{}', '{"source_language":"vi","translation":""}',
                         '{"source_language":"unknown","translation":"a horse"}'):
            with self.assertRaises(ValueError):
                parse_translation(response, 'query')

    def test_second_call_receives_translation_only_and_cache_avoids_generation(self):
        source = Path(__file__).resolve().parents[1] / 'src/online_pipeline/query_expander.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.ClassDef) and n.name == 'QueryExpander')
        scope = dict(translation_messages=translation_messages, parse_translation=parse_translation,
                     expansion_messages=expansion_messages, parse_expansions=parse_expansions,
                     logger=logging.getLogger('translation-test'))
        exec(compile('from __future__ import annotations\n' + ast.unparse(node), str(source), 'exec'), scope)
        expander = scope['QueryExpander'].__new__(scope['QueryExpander'])
        expander.model = expander.tokenizer = object()
        expander.cache = {}
        calls = []
        def generate(messages):
            calls.append(messages)
            if len(calls) == 1:
                return '{"source_language":"vi","translation":"A white lion-dance costume"}'
            raise ValueError('malformed expansion')
        expander._generate = generate
        with self.assertLogs('translation-test', level='WARNING'):
            result = expander.expand('Con lân trắng')
        self.assertEqual(result, ['A white lion-dance costume'])
        self.assertEqual(calls[1][-1]['content'], result[0])
        self.assertEqual(expander.expand('Con lân trắng'), result)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
