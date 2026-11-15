import contextlib
import io
import math
import os
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.prepare_comi_lingua import ANNOTATOR_COLUMNS, Record, majority_record, retokenize
from src.candidate_expansion.expander import CandidateExpander, TokenCandidates, tokenize
from src.candidate_expansion.variants import canonical_of, urduphone
from src.classifier.dataset import read_any
from src.classifier.train import align_labels
from src.eval.metrics import accuracy, cmi, macro_f1, summarize
from src.pipeline import CHECKPOINT_RESOURCES, DEFAULT_RESOURCES, RESOURCE_FILES, LIDPipeline, resolve_resources
from src.verification.rules import DEFAULT_TAU, apply_rules
from src.verification.tau import (
    STAGE3_CONFIG,
    ScoredSentence,
    SweepPoint,
    candidate_taus,
    resolve_tau,
    save_tuned_tau,
    select_tau,
    sweep,
)

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


class TauSelectionTest(unittest.TestCase):
    @staticmethod
    def point(tau, macro_f1, accuracy=0.9):
        return SweepPoint(tau=tau, tokens=10, accuracy=accuracy, macro_f1=macro_f1, below_tau=0, fired=0, fixed=0,
                          broke=0)

    def test_picks_the_best_validation_macro_f1(self):
        points = [self.point(0.0, 0.90), self.point(0.4, 0.93), self.point(0.7, 0.91)]
        self.assertEqual(select_tau(points).tau, 0.4)

    def test_ties_go_to_accuracy_then_to_the_largest_tau_before_a_drop(self):
        points = [self.point(0.0, 0.90), self.point(0.6, 0.93, 0.95), self.point(0.3, 0.93, 0.95),
                  self.point(0.2, 0.93, 0.94), self.point(0.8, 0.92, 0.95)]
        self.assertEqual(select_tau(points).tau, 0.6)

    def test_keeps_the_last_tau_before_the_score_drops(self):
        points = [self.point(0.5, 0.93), self.point(0.0, 0.93), self.point(0.9, 0.92)]
        self.assertEqual(select_tau(points).tau, 0.5)
        recovers = [self.point(0.0, 0.93), self.point(0.3, 0.92), self.point(0.6, 0.93)]
        self.assertEqual(select_tau(recovers).tau, 0.0)  # the first drop ends the search
        with self.assertRaises(ValueError):
            select_tau(points[:1])  # without tau = 0 there is nothing to compare against

    def test_sweep_finds_the_largest_tau_with_the_best_outcome(self):
        nhi = TokenCandidates("nhi", "nhi", "variant", False, ("nahi",))
        he = TokenCandidates("he", "he", "both", True, ("hai",))
        sentences = [
            ScoredSentence((nhi,), (("EN", 0.30),), ("UR",)),  # variant-consistency fixes it once tau > 0.30
            ScoredSentence((he,), (("EN", 0.60),), ("EN",)),  # ... and breaks this English "he" once tau > 0.60
            ScoredSentence((nhi,), (("UR", 0.95),), ("UR",)),  # no rule applies to a UR "nhi"
        ]
        taus = candidate_taus(sentences, grid=None)
        up = math.inf
        self.assertEqual(taus, [0.0, 0.30, math.nextafter(0.30, up), 0.60, math.nextafter(0.60, up)])
        best = select_tau(sweep(sentences, candidate_taus(sentences)))
        self.assertEqual(best.tau, 0.60)  # the fix is in; one step higher breaks the English "he"
        self.assertEqual((best.fired, best.fixed, best.broke, best.accuracy), (1, 1, 0, 1.0))
        harmful = sentences[1:]  # only the firing that breaks a label: tau stops just below it
        chosen = select_tau(sweep(harmful, candidate_taus(harmful)))
        self.assertEqual((chosen.tau, chosen.fired), (0.60, 0))


class TauPrecedenceTest(unittest.TestCase):
    def test_explicit_beats_tuned_beats_default(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(resolve_tau(None, None), (DEFAULT_TAU, "default"))
            self.assertEqual(resolve_tau(None, folder), (DEFAULT_TAU, "default"))  # not tuned yet
            save_tuned_tau(Path(folder) / STAGE3_CONFIG, {"tau": 0.0})
            self.assertEqual(resolve_tau(None, folder)[0], 0.0)
            self.assertEqual(resolve_tau(0.7, folder), (0.7, "explicit"))
            self.assertEqual(resolve_tau(0.0, None), (0.0, "explicit"))

    def test_tuned_value_of_retrained_weights_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            config, weights = Path(folder) / STAGE3_CONFIG, Path(folder) / "model.safetensors"
            save_tuned_tau(config, {"tau": 0.0})
            weights.write_bytes(b"")
            os.utime(config, (1_000_000, 1_000_000))
            os.utime(weights, (2_000_000, 2_000_000))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(resolve_tau(None, folder)[0], DEFAULT_TAU)

    def test_hub_id_reads_the_tuned_value_from_the_hub(self):
        with tempfile.TemporaryDirectory() as folder:
            downloaded = Path(folder) / STAGE3_CONFIG
            save_tuned_tau(downloaded, {"tau": 0.6})
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = mock.Mock(return_value=str(downloaded))
            with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
                tau, source = resolve_tau(None, "someone/some-model")
                self.assertEqual(resolve_tau(0.7, "someone/some-model"), (0.7, "explicit"))
        self.assertEqual(tau, 0.6)
        self.assertIn("Hugging Face", source)
        hub.hf_hub_download.assert_called_once_with("someone/some-model", STAGE3_CONFIG)

    def test_hub_id_without_a_tuned_value_uses_the_default(self):
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = mock.Mock(side_effect=OSError("404: no stage3.json in this repository"))
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertEqual(resolve_tau(None, "someone/some-model"), (DEFAULT_TAU, "default"))
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):  # huggingface_hub not installed
            self.assertEqual(resolve_tau(None, "someone/some-model"), (DEFAULT_TAU, "default"))

    def test_pipeline_uses_the_tuned_value_unless_tau_is_given(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("src.classifier.predict.TokenLIDPredictor"):
            save_tuned_tau(Path(folder) / STAGE3_CONFIG, {"tau": 0.0})
            self.assertEqual(LIDPipeline.from_resources(RESOURCES, checkpoint=folder).tau, 0.0)
            self.assertEqual(LIDPipeline.from_resources(RESOURCES, checkpoint=folder, tau=0.8).tau, 0.8)
        self.assertEqual(LIDPipeline.from_resources(RESOURCES).tau, DEFAULT_TAU)  # fallback mode


class ResourcePrecedenceTest(unittest.TestCase):
    def test_explicit_beats_saved_with_the_checkpoint_beats_the_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(resolve_resources(None), (DEFAULT_RESOURCES, "default"))
            self.assertEqual(resolve_resources(folder), (DEFAULT_RESOURCES, "default"))  # plain checkpoint
            saved = os.path.join(folder, CHECKPOINT_RESOURCES)
            os.mkdir(saved)
            self.assertEqual(resolve_resources(folder), (saved, "saved with the checkpoint"))
            self.assertEqual(resolve_resources(folder, RESOURCES), (RESOURCES, "explicit"))
            self.assertEqual(resolve_resources("someone/some-model"), (DEFAULT_RESOURCES, "default"))  # hub id

    def test_pipeline_reads_the_lexicons_saved_with_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("src.classifier.predict.TokenLIDPredictor"):
            saved = os.path.join(folder, CHECKPOINT_RESOURCES)
            os.mkdir(saved)
            for name in RESOURCE_FILES:
                with open(os.path.join(RESOURCES, name), encoding="utf-8") as source:
                    text = source.read()
                if name == "english_seed.txt":
                    text += "\nwo\n"  # this copy also lists "wo" as English
                with open(os.path.join(saved, name), "w", encoding="utf-8") as target:
                    target.write(text)
            pipeline = LIDPipeline.from_resources(checkpoint=folder)
            self.assertEqual(pipeline.resources_source, "saved with the checkpoint")
            self.assertEqual(pipeline.expander.expand_token("wo").category, "ambiguous")
        self.assertEqual(LIDPipeline.from_resources().expander.expand_token("wo").category, "standard")


class MetricsTest(unittest.TestCase):
    def test_accuracy_and_f1(self):
        self.assertAlmostEqual(accuracy(["UR", "EN"], ["UR", "EN"]), 1.0)
        self.assertAlmostEqual(macro_f1(["UR", "EN"], ["UR", "EN"]), 1.0)
        self.assertLess(accuracy(["UR", "EN"], ["EN", "EN"]), 1.0)

    def test_cmi(self):
        self.assertAlmostEqual(cmi(["UR", "EN"]), 50.0)  # Gambäck & Das: two languages peak at 50
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
