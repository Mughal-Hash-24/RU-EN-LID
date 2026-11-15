"""Edge cases, input validation and reference checks for the production review.
Standard library only, so it runs in the CI job without torch."""

import io
import itertools
import math
import os
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.candidate_expansion.lexicons import load_lexicons, read_wordlist  # noqa: E402
from src.candidate_expansion.variants import canonical_of, load_variant_pairs  # noqa: E402
from src.classifier.dataset import read_seed_csv  # noqa: E402
from src.eval.metrics import accuracy, cmi, macro_f1, mean_sentence_cmi, summarize  # noqa: E402
from src.pipeline import LIDPipeline, heuristic_distributions, heuristic_predictions  # noqa: E402
from src.pipeline import main as pipeline_main  # noqa: E402
from src.verification.context import LABELS, LANGUAGES, decode  # noqa: E402


def write(folder: str, name: str, text: str) -> str:
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class WordListTest(unittest.TestCase):
    def test_comment_lines_add_no_words(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write(folder, "w.txt", "# seeds + induced from train.csv\nhello  # trailing note\nbye, ok\n")
            self.assertEqual(read_wordlist(path), {"hello", "bye", "ok"})

    def test_false_friend_guard_and_cached_intersection(self):
        with tempfile.TemporaryDirectory() as folder:
            write(folder, "english_seed.txt", "main hai to box")
            write(folder, "roman_urdu_seed.txt", "main hai to tha")
            lexicons = load_lexicons(folder)
            self.assertNotIn("hai", lexicons.english)  # Roman Urdu function word removed from English
            self.assertEqual(lexicons.ambiguous, {"main", "to"})
            self.assertIs(lexicons.ambiguous, lexicons.ambiguous)  # computed once


class VariantPairsTest(unittest.TestCase):
    def test_comments_duplicates_and_short_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write(folder, "p.tsv", "# canonical\tvariant\nbohat\tbht\nbohat\tbht\nlonely\nnahi\tnhi # note\n")
            self.assertEqual(load_variant_pairs(path), {"bohat": ["bht"], "nahi": ["nhi"]})

    def test_a_variant_with_two_canonicals_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write(folder, "p.tsv", "hai\the\nhe\the\n")
            with self.assertRaisesRegex(ValueError, ":2: variant 'he' already belongs to 'hai'"):
                load_variant_pairs(path)

    def test_canonical_prefers_longest_then_alphabetical(self):
        self.assertEqual(canonical_of(["bar", "baar"]), "baar")
        self.assertEqual(canonical_of(["mein", "main"]), "main")
        with self.assertRaises(ValueError):
            canonical_of([])


class SeedCsvTest(unittest.TestCase):
    HEADER = "sentence_id,token,token_index,label\n"

    def read(self, body: str):
        with tempfile.TemporaryDirectory() as folder:
            return read_seed_csv(write(folder, "d.csv", self.HEADER + body))

    def test_mixed_numeric_and_text_ids_sort_without_crashing(self):
        sentences = self.read("s10,a,0,EN\nabc,b,0,UR\ns2,c,0,OTH\n")
        self.assertEqual([s.sentence_id for s in sentences], ["s2", "s10", "abc"])

    def test_tokens_are_ordered_by_index_and_labels_upper_cased(self):
        (sentence,) = self.read("s1,world,1,en\ns1,hello,0,ign\n")
        self.assertEqual(sentence.words, ["hello", "world"])
        self.assertEqual(sentence.labels, ["IGN", "EN"])

    def test_bad_rows_fail_with_the_line_number(self):
        with self.assertRaisesRegex(ValueError, ":2: unknown label 'NE'"):
            self.read("s1,Lahore,0,NE\n")
        with self.assertRaisesRegex(ValueError, ":3: token_index 'x' is not an integer"):
            self.read("s1,a,0,EN\ns1,b,x,EN\n")
        with self.assertRaisesRegex(ValueError, "repeats a token_index"):
            self.read("s1,a,0,EN\ns1,b,0,EN\n")


def reference_macro_f1(gold, pred):
    """The original three-pass implementation, kept as an oracle."""
    labels = sorted(set(gold) | set(pred))
    if not labels:
        return 0.0
    total = 0.0
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        total += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return total / len(labels)


class MetricsTest(unittest.TestCase):
    def test_macro_f1_matches_the_reference_on_random_data(self):
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(0, 40)
            gold = [rng.choice("ABCD") for _ in range(n)]
            pred = [rng.choice("ABCDE") for _ in range(n)]
            self.assertAlmostEqual(macro_f1(gold, pred), reference_macro_f1(gold, pred), places=12)

    def test_length_mismatch_is_an_error_not_a_silent_truncation(self):
        with self.assertRaises(ValueError):
            accuracy(["UR", "EN"], ["UR"])
        with self.assertRaises(ValueError):
            macro_f1(["UR"], [])
        self.assertEqual(accuracy([], []), 0.0)
        self.assertEqual(macro_f1([], []), 0.0)

    def test_cmi_follows_gamback_and_das(self):
        self.assertEqual(cmi([]), 0.0)
        self.assertEqual(cmi(["OTH", "IGN"]), 0.0)  # n == u
        self.assertEqual(cmi(["UR", "UR", "UR"]), 0.0)
        self.assertAlmostEqual(cmi(["UR", "EN"]), 50.0)
        self.assertAlmostEqual(cmi(["UR", "UR", "EN", "OTH"]), 100 * (1 - 2 / 3))  # OTH is language-independent
        self.assertAlmostEqual(cmi(["UR", "EN", "X"]), 100 * (1 - 1 / 3))  # three languages can exceed 50

    def test_corpus_cmi_is_the_mean_over_sentences(self):
        rows = [{"sentence_id": "a", "gold": g} for g in ("UR", "EN")] + [
            {"sentence_id": "b", "gold": g} for g in ("UR", "UR")
        ]
        self.assertAlmostEqual(mean_sentence_cmi(rows), 25.0)
        self.assertEqual(mean_sentence_cmi([]), 0.0)

    def test_summarize_handles_no_errors_and_ignored_tokens(self):
        rows = [
            {"sentence_id": "a", "gold": "UR", "pred": "UR", "category": "standard"},
            {"sentence_id": "a", "gold": "IGN", "pred": "EN", "category": "standard"},
        ]
        report = summarize(rows)
        self.assertEqual((report["tokens"], report["ignored"], report["errors"]), (1, 1, 0))
        self.assertEqual(report["error_share_by_cause"], {})
        self.assertEqual(report["accuracy"], 1.0)


def path_cost(probabilities, labels, penalty):
    """Cost the decoder minimises, written out directly from its definition."""
    cost, language = 0.0, None
    for probs, label in zip(probabilities, labels):
        cost += -math.log(max(probs.get(label, 0.0), 1e-6))
        if label in LANGUAGES:
            if language is not None and language != label:
                cost += penalty
            language = label
    return cost


class DecoderTest(unittest.TestCase):
    def test_decode_is_optimal_against_brute_force(self):
        rng = random.Random(3)
        for _ in range(400):
            n = rng.randint(1, 6)
            probabilities = []
            for _ in range(n):
                weights = [rng.random() + (0.0 if rng.random() < 0.8 else -1.0) for _ in LABELS]
                weights = [max(w, 0.0) for w in weights]
                total = sum(weights) or 1.0
                probabilities.append({label: w / total for label, w in zip(LABELS, weights)})
            pinned = [rng.choice(LABELS) if rng.random() < 0.4 else None for _ in range(n)]
            penalty = rng.choice([0.0, 0.3, 1.0, 5.0])
            got = decode(probabilities, pinned, penalty)
            self.assertTrue(all(pin is None or pin == label for pin, label in zip(pinned, got)))
            options = [LABELS if pin is None else (pin,) for pin in pinned]
            emitted = [p if pin is None else {pin: 1.0} for p, pin in zip(probabilities, pinned)]
            best = min(path_cost(emitted, labels, penalty) for labels in itertools.product(*options))
            self.assertAlmostEqual(path_cost(emitted, got, penalty), best, places=9)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            decode([{"UR": 1.0}], [None], -0.1)
        with self.assertRaises(ValueError):
            decode([{"UR": 1.0}], ["IGN"], 1.0)
        with self.assertRaises(ValueError):
            decode([{"UR": 1.0}], [], 1.0)
        self.assertEqual(decode([], [], 1.0), [])


class PipelineFallbackTest(unittest.TestCase):
    def test_fallback_numbers_are_unchanged(self):
        pipeline = LIDPipeline.from_resources()
        tokens = pipeline.expander.expand("bht main upset tha jio")
        self.assertEqual(
            heuristic_predictions(tokens), [("UR", 0.40), ("EN", 0.30), ("EN", 0.90), ("UR", 0.90), ("OTH", 0.0)]
        )
        first, second = heuristic_distributions(tokens)[:2]
        self.assertEqual(first, {"UR": 0.6, "EN": 0.4, "OTH": 0.0})
        self.assertEqual(second, {"UR": 0.5, "EN": 0.5, "OTH": 0.0})
        first["UR"] = 0.0  # callers get copies, so the shared table cannot be corrupted
        self.assertEqual(heuristic_distributions(tokens)[0]["UR"], 0.6)

    def test_empty_and_punctuation_only_text(self):
        pipeline = LIDPipeline.from_resources(context_penalty=1.0)
        self.assertEqual(pipeline.run_records(""), [])
        self.assertEqual(pipeline.run_records("?! ..."), [])

    def test_cli_rejects_out_of_range_numbers(self):
        for argv in (["--text", "x", "--tau", "1.5"], ["--text", "x", "--context-penalty", "-1"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pipeline_main(argv)


if __name__ == "__main__":
    unittest.main()
