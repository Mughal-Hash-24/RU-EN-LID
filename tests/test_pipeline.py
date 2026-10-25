import os
import sys
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.prepare_comi_lingua import ANNOTATOR_COLUMNS, Record, majority_record, retokenize
from src.candidate_expansion.expander import CandidateExpander, TokenCandidates, tokenize
from src.candidate_expansion.variants import canonical_of, urduphone
from src.classifier.dataset import read_any
from src.classifier.train import align_labels
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

    def test_ignored_tokens_are_not_scored(self):
        rows = [
            {"gold": "UR", "pred": "UR", "category": "standard"},
            {"gold": "IGN", "pred": "EN", "category": "ambiguous"},
        ]
        report = summarize(rows)
        self.assertEqual(report["tokens"], 1)
        self.assertEqual(report["ignored"], 1)
        self.assertEqual(report["errors"], 0)
        self.assertNotIn("ambiguous", report["per_category"])


class LabelAlignmentTest(unittest.TestCase):
    def test_only_first_pieces_of_labelled_words_are_trained(self):
        # <s> main (2 pieces) to (IGN) upset </s>
        word_ids = [None, 0, 0, 1, 2, None]
        labels = align_labels(word_ids, ["UR", "IGN", "EN"], {"UR": 0, "EN": 1, "OTH": 2})
        self.assertEqual(labels, [-100, 0, -100, -100, 1, -100])


class PrepareComiLinguaTest(unittest.TestCase):
    @staticmethod
    def row(words, *annotations):
        tags = [repr([{"key": w, "value": t, "index": i} for i, (w, t) in enumerate(zip(words, tags))])
                for tags in annotations]
        return {"Sentences": " ".join(words), **dict(zip(ANNOTATOR_COLUMNS, tags))}

    def test_majority_vote_maps_both_tag_styles_and_ignores_three_way_splits(self):
        words = ["ye", "to", "good"]
        row = self.row(words, ["hi", "en", "en"], ["hi", "ot", "en"], ["hi", "hi", "en"])
        record = majority_record(0, row, Counter(), Counter())
        self.assertEqual(record.labels, ["UR", "IGN", "EN"])
        short = majority_record(1, self.row(["kya", "pic", "!"], *[["h", "e", "u"]] * 3), Counter(), Counter())
        self.assertEqual(short.labels, ["UR", "EN", "OTH"])

    def test_retokenize_drops_punctuation_and_survives_rejoining(self):
        record = Record(0, "..aap , theek", ["..aap", ",", "theek"], ["UR", "OTH", "UR"])
        [result] = retokenize([record], {"tokens": {}, "steps": []})
        self.assertEqual((result.words, result.labels), (["aap", "theek"], ["UR", "UR"]))
        self.assertEqual(tokenize(" ".join(result.words)), result.words)


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
