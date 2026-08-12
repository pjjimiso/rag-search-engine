import unittest

from pkg.search_utils import preprocess_text
from pkg.search_utils import tokenize_text
from pkg.search_utils import has_matching_token

class TestSearchUtils(unittest.TestCase):
    def test_preprocess_text(self):
        """
        Test text is converted to lowercase with punctuation removed.
        """
        processed_text = preprocess_text("Yo whaddup, this is Pat!")
        self.assertEqual(processed_text, "yo whaddup this is pat")

    def test_tokenize_text(self):
        tokenized_text = tokenize_text("Yo whaddup, this is Pat!")
        self.assertEqual(tokenized_text, ["yo", "whaddup", "this", "is", "pat"])

    def test_has_matching_token_true(self):
        """
        Test query token matches any title token
        """
        query_tokens = ["pat"]
        title_tokens = ["this", "is", "pat"]
        self.assertTrue(has_matching_token(query_tokens, title_tokens))

    def test_has_matching_token_false(self):
        """
        Test query token does not match any title token
        """
        query_tokens = ["notpat"]
        title_tokens = ["this", "is", "pat"]
        self.assertFalse(has_matching_token(query_tokens, title_tokens))

if __name__ == "__main__":
    unittest.main()

