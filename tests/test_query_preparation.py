import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.online_pipeline.query_preparation import (
    QueryPreparationError, prepare_english, prepare_chunks, split_for_encoder,
)


VI = "Ba tay đua xe đạp trên đường bụi. Người cuối mang số 71."
EN = "Three cyclists on a dusty road. The last rider wears number 71."


class QueryPreparationTests(unittest.TestCase):
    @patch("src.online_pipeline.query_preparation.translate_vi_to_en", return_value=EN)
    def test_missing_local_model_uses_existing_translator(self, translate):
        plan = prepare_english(VI, None)
        translate.assert_called_once_with(VI)
        self.assertEqual(plan["english"], [EN])
        self.assertEqual(plan["source"], "translator")

    @patch("src.online_pipeline.query_preparation.translate_vi_to_en", return_value=EN)
    def test_english_also_uses_existing_translator(self, translate):
        self.assertEqual(prepare_english(EN, None)["english"], [EN])
        translate.assert_called_once_with(EN)

    @patch("src.online_pipeline.query_preparation.translate_vi_to_en")
    def test_optional_expansion_applies_to_both_languages(self, translate):
        for query in [VI, EN]:
            model = SimpleNamespace(expand=lambda _: [EN, "Bicycle race", "dusty road"])
            plan = prepare_english(query, model)
            self.assertEqual(plan["source"], "local_llm")
            self.assertEqual(plan["english"][0], EN)
        translate.assert_not_called()

    @patch("src.online_pipeline.query_preparation.translate_vi_to_en", return_value=EN)
    def test_failed_or_empty_expansion_falls_back(self, translate):
        def failure(_):
            raise RuntimeError("GPU error")
        for expand in [failure, lambda _: []]:
            plan = prepare_english(VI, SimpleNamespace(expand=expand))
            self.assertEqual(plan["english"], [EN])
            self.assertEqual(plan["source"], "translator")

    @patch("src.online_pipeline.query_preparation.translate_vi_to_en", side_effect=RuntimeError("offline"))
    def test_failed_translation_keeps_main_fallback(self, translate):
        plan = prepare_english(VI, None)
        self.assertEqual(plan["english"], [VI])
        self.assertEqual(plan["source"], "original")

    def test_chunks_keep_last_number_and_branch_weights(self):
        encoder = SimpleNamespace(text_context_length=7, text_token_count=lambda text: len(text.split()) + 2)
        plan = {"input": VI, "source": "local_llm", "english": [EN, "Bicycle race"], "warnings": []}
        output = prepare_chunks(plan, encoder, [0.7, 0.2, 0.1])
        self.assertTrue(all(n <= 7 for n in output["token_counts"]))
        self.assertIn("71.", " ".join(output["chunks"]))
        self.assertEqual(" ".join(output["chunks"][:-1]), EN)
        self.assertAlmostEqual(sum(output["weights"][:-1]), 0.7)
        self.assertAlmostEqual(output["weights"][-1], 0.2)
        self.assertTrue(output["warnings"])

    def test_unencodable_word_is_not_silently_truncated(self):
        with self.assertRaises(QueryPreparationError):
            split_for_encoder("abcdefgh", len, 3)

    def test_split_prefers_sentence_and_clause_boundaries(self):
        text = "A woman in yellow stands left of a white ambulance. She hands a red hat to a boy, not the man behind."
        count_tokens = lambda value: len(value.split()) + 2
        chunks = split_for_encoder(text, count_tokens, 12)
        self.assertEqual(chunks, [
            "A woman in yellow stands left of a white ambulance.",
            "She hands a red hat to a boy,",
            "not the man behind.",
        ])
        self.assertTrue(all(count_tokens(c) <= 12 for c in chunks))

    def test_token_counter_uses_unpadded_untruncated_encoding(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/online_pipeline/query_encoder.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'text_token_count')
        namespace = {}
        exec(compile(ast.unparse(node), 'text_token_count', 'exec'), namespace)
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [0] * len(text.split()) + ([1] if add_special_tokens else [])
        subject = SimpleNamespace(tokenizer=Tokenizer())
        self.assertEqual(namespace['text_token_count'](subject, 'x ' * 400), 401)


if __name__ == '__main__':
    unittest.main()
