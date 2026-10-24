import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.candidate_expansion.expander import CandidateExpander, TokenCandidates, tokenize
from src.candidate_expansion.variants import canonical_of, urduphone
from src.classifier.dataset import read_any
from src.eval.metrics import accuracy, cmi, macro_f1, summarize
from src.pipeline import LIDPipeline
from src.verification.rules import apply_rules

RESOURCES = os.path.join(ROOT, "data", "resources")


class TokenizerTest(unittest.TestCase):
    def test_tokenizes_words_and_skips_punctuation(self):
        self.assertEqual(tokenize("Delivery to time pe hui, but box band nhi tha."),
                         ["Delivery", "to", "time", "pe", "hui", "but", "box", "band", "nhi", "tha"])


class VariantTest(unittest.TestCase):
    def test_phonetic_codes_match_common_variant_pairs(self):
        self.assertEqual(urduphone("bar"), urduphone("baar"))
        self.assertEqual(urduphone("he"), urduphone("hai"))
        self.assertEqual(urduphone("main"), urduphone("mein"))

    def test_canonical_selection(self):
        self.assertEqual(canonical_of(["bar", "baar"]), "baar")
        self.assertEqual(canonical_of(["he", "hai"]), "hai")
        self.assertEqual(canonical_of(["main", "mein"]), "main")


class CandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expander = CandidateExpander.from_resources(RESOURCES)

    def category(self, word):
        return self.expander.expand_token(word).category

    def test_taxonomy_categories(self):
        self.assertEqual(self.category("main"), "ambiguous")
        self.assertEqual(self.category("band"), "ambiguous")
        self.assertEqual(self.category("is"), "ambiguous")
        self.assertEqual(self.category("bar"), "both")
        self.assertEqual(self.category("he"), "both")
        self.assertEqual(self.category("bht"), "variant")
        self.assertEqual(self.category("nhi"), "variant")
        self.assertEqual(self.category("delivery"), "standard")

    def test_candidates_keep_surface_form(self):
        info = self.expander.expand_token("Bar")
        self.assertEqual(info.surface, "Bar")
        self.assertEqual(info.normalized, "bar")
        self.assertTrue(info.english_reading)
        self.assertEqual(info.urdu_canonicals, ("baar",))


class RuleTest(unittest.TestCase):
    def test_variant_rule_flips_low_confidence_english(self):
        token = TokenCandidates("nhi", "nhi", "variant", False, ("nahi",))
        decisions = apply_rules([token], [("EN", 0.2)])
        self.assertEqual(decisions[0].label, "UR")
        self.assertIn("variant-consistency", decisions[0].rules)

    def test_confident_prediction_is_untouched(self):
        token = TokenCandidates("nhi", "nhi", "variant", False, ("nahi",))
        decisions = apply_rules([token], [("EN", 0.95)])
        self.assertEqual(decisions[0].label, "EN")
        self.assertEqual(decisions[0].rules, ())


class MetricsTest(unittest.TestCase):
    def test_accuracy_and_f1(self):
        self.assertAlmostEqual(accuracy(["UR", "EN"], ["UR", "EN"]), 1.0)
        self.assertAlmostEqual(macro_f1(["UR", "EN"], ["UR", "EN"]), 1.0)
        self.assertLess(accuracy(["UR", "EN"], ["EN", "EN"]), 1.0)

    def test_cmi(self):
        self.assertAlmostEqual(cmi(["UR", "EN"]), 100.0)
        self.assertAlmostEqual(cmi(["UR", "UR", "UR"]), 0.0)

    def test_error_attribution(self):
        rows = [
            {"gold": "UR", "pred": "EN", "category": "ambiguous"},
            {"gold": "EN", "pred": "EN", "category": "standard"},
        ]
        report = summarize(rows)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["error_share_by_cause"], {"ambiguous": 1.0})


class SeedSliceTest(unittest.TestCase):
    def test_seed_file_loads_and_pipeline_roundtrips(self):
        sentences = read_any(os.path.join(ROOT, "data", "demo", "seed_labels.csv"))
        self.assertEqual(len(sentences), 12)
        pipeline = LIDPipeline.from_resources(RESOURCES)
        total = 0
        for sentence in sentences:
            pairs = pipeline.run(" ".join(sentence.words))
            self.assertEqual(len(pairs), len(sentence.words))
            total += len(pairs)
        self.assertEqual(total, 51)


if __name__ == "__main__":
    unittest.main()
