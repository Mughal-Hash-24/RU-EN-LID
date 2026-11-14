"""Iteration 2 Stage 3 and evaluation: context decoder, selective report, lexicon induction.
Standard library only, so it runs in the CI job without torch."""

import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.candidate_expansion.induce import induce  # noqa: E402
from src.classifier.dataset import Sentence, read_any  # noqa: E402
from src.eval.selective import collect_rows, selective_report  # noqa: E402
from src.pipeline import LIDPipeline  # noqa: E402
from src.verification.context import decode  # noqa: E402

UR = {"UR": 0.9, "EN": 0.05, "OTH": 0.05}
EN = {"UR": 0.05, "EN": 0.9, "OTH": 0.05}
UNSURE = {"UR": 0.45, "EN": 0.55, "OTH": 0.0}
EXAMPLE = "Delivery to time pe hui but box band nhi tha main bht upset hoon is bar bhi same issue he"


class DecoderTest(unittest.TestCase):
    def test_zero_penalty_keeps_stage2_argmax(self):
        self.assertEqual(decode([UR, UNSURE, UR], [None, None, None], 0.0), ["UR", "EN", "UR"])

    def test_penalty_pulls_an_unsure_word_to_its_neighbours(self):
        self.assertEqual(decode([UR, UNSURE, UR], ["UR", None, "UR"], 1.0), ["UR", "UR", "UR"])

    def test_pinned_words_never_change(self):
        self.assertEqual(decode([UR, EN, UR], ["UR", "EN", "UR"], 100.0), ["UR", "EN", "UR"])

    def test_confident_stage2_beats_a_small_penalty(self):
        self.assertEqual(decode([UR, EN, UR], ["UR", None, "UR"], 0.5), ["UR", "EN", "UR"])

    def test_oth_does_not_break_the_language_run(self):
        oth = {"UR": 0.0, "EN": 0.0, "OTH": 1.0}
        self.assertEqual(decode([UR, oth, UNSURE], ["UR", "OTH", None], 1.0), ["UR", "OTH", "UR"])

    def test_empty_and_mismatched_input(self):
        self.assertEqual(decode([], [], 1.0), [])
        with self.assertRaises(ValueError):
            decode([UR], [], 1.0)


class PipelineContextTest(unittest.TestCase):
    def test_context_fixes_ambiguous_words_and_leaves_variants_alone(self):
        off = {r["token"]: r for r in LIDPipeline.from_resources().run_records(EXAMPLE)}
        on = {r["token"]: r for r in LIDPipeline.from_resources(context_penalty=1.0).run_records(EXAMPLE)}
        for word in ("band", "main", "is"):
            self.assertEqual(off[word]["label"], "EN")
            self.assertEqual(on[word]["label"], "UR")
            self.assertIn("context", on[word]["rules"])
            self.assertEqual(on[word]["stage2_label"], "EN")
        self.assertEqual(on["he"]["label"], "UR")  # variant: not re-decoded (insertion pattern)
        self.assertEqual(on["Delivery"]["label"], "EN")  # accepted words are pinned

    def test_selective_report_counts_fixes_and_breaks(self):
        sentences = read_any(os.path.join(ROOT, "data", "demo", "running_examples.csv"))
        off = selective_report(collect_rows(LIDPipeline.from_resources(), sentences))
        on = selective_report(collect_rows(LIDPipeline.from_resources(context_penalty=1.0), sentences))
        self.assertEqual(off["stage3_fixes"], 0)
        self.assertGreater(on["stage3_fixes"], 0)
        self.assertEqual(on["stage3_breaks"], 0)
        self.assertEqual(on["by_rule"]["context"]["fixed"], on["stage3_fixes"])
        self.assertGreater(on["final_accuracy"], off["final_accuracy"])
        self.assertEqual(on["accepted_accuracy"], off["accepted_accuracy"])  # Stage 3 never touches accepted words
        self.assertAlmostEqual(on["coverage"] * on["tokens"] + on["deferred_tokens"], on["tokens"])


class InductionTest(unittest.TestCase):
    def sentences(self):
        data = [("main ghar ja raha hoon", "UR UR UR UR UR"), ("main gate band hai", "EN EN UR UR"),
                ("main bhi aa raha", "UR UR UR UR"), ("the main road", "EN EN EN"), ("the box", "EN EN"),
                ("the gate", "EN EN"), ("ghar ghar", "UR UR")]
        return [Sentence(str(i), words.split(), labels.split()) for i, (words, labels) in enumerate(data)]

    def test_thresholds(self):
        lexicon = induce(self.sentences(), min_count=3, purity=0.9, ambiguous_share=0.2)
        self.assertEqual(lexicon.ambiguous, {"main"})  # 3 UR / 2 EN
        self.assertIn("the", lexicon.english)
        self.assertIn("ghar", lexicon.roman_urdu)
        self.assertNotIn("box", lexicon.english)  # seen once: too rare
        with self.assertRaises(ValueError):
            induce([], purity=0.5)

    def test_build_lexicon_script_writes_a_usable_resource_folder(self):
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts", "build_lexicon.py"),
                 "--train", os.path.join(ROOT, "data", "demo", "seed_labels.csv"), "--out", out,
                 "--min-count", "1", "--check", os.path.join(ROOT, "data", "demo", "running_examples.csv")],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("english_seed.txt", "roman_urdu_seed.txt", "variant_pairs.tsv", "induction_report.json"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            records = LIDPipeline.from_resources(out).run_records(EXAMPLE)
            self.assertEqual(len(records), len(EXAMPLE.split()))


if __name__ == "__main__":
    unittest.main()
