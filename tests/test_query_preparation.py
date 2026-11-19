import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.online_pipeline.query_preparation import (
    QueryPreparationError, prepare_english, prepare_chunks, split_for_encoder,
)


VI = "Ba tay đua xe đạp trên đường bụi. Người cuối mang số 71."
EN = "Three cyclists on a dusty road. The last rider wears number 71."


class QueryPreparationTests(unittest.TestCase):
    def test_english_is_unchanged_and_does_not_call_translator(self):
        def forbidden(_):
            raise AssertionError("English must bypass expansion")
        plan = prepare_english(EN, SimpleNamespace(model=True, expand=forbidden))
        self.assertEqual(plan["english"], [EN])
        self.assertEqual(plan["source"], "original")

    def test_missing_model_blocks_vietnamese_instead_of_silent_fallback(self):
        for expander in [None, SimpleNamespace(model=None)]:
            with self.assertRaises(QueryPreparationError):
                prepare_english(VI, expander)

    def test_local_translation_used_and_reported(self):
        model = SimpleNamespace(model=True, expand=lambda _: [EN, "Bicycle race", "dusty road"])
        plan = prepare_english(VI, model)
        self.assertEqual(plan["source"], "local_llm")
        self.assertEqual(plan["english"][0], EN)

    def test_invalid_local_output_and_exception_block_search(self):
        for values in [[VI], [], [""]]:
            with self.assertRaises(QueryPreparationError):
                prepare_english(VI, SimpleNamespace(model=True, expand=lambda _: values))
        def failure(_):
            raise RuntimeError("GPU error")
        with self.assertRaises(QueryPreparationError):
            prepare_english(VI, SimpleNamespace(model=True, expand=failure))

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
