"""Stage 1 -> Stage 2 conditioning. Needs torch/transformers/tokenizers; builds a tiny
XLM-R-shaped model and tokenizer locally, so it runs offline in a few seconds."""

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.candidate_expansion.expander import CandidateExpander  # noqa: E402
from src.classifier.features import CAND2ID, reading_of, stage1_tag, word_features  # noqa: E402

RESOURCES = os.path.join(ROOT, "data", "resources")
EXAMPLE = "Delivery to time pe hui but box band nhi tha main bht upset hoon is bar bhi same issue he".split()

try:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForTokenClassification

    HAVE_TORCH = True
except ImportError:  # the standard-library CI job skips this file
    HAVE_TORCH = False


def build_tiny_checkpoint(folder: str) -> None:
    """A plain (iteration-1 style) token classifier with XLM-R's special-token ids."""
    words = set(EXAMPLE)
    for name in ("english_seed.txt", "roman_urdu_seed.txt"):
        with open(os.path.join(RESOURCES, name), encoding="utf-8") as handle:
            words |= set(handle.read().split())
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
    tokenizer.train_from_iterator(sorted(words), trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
        cls_token="<s>", sep_token="</s>",
    )
    torch.manual_seed(0)
    config = XLMRobertaConfig(
        vocab_size=tokenizer.get_vocab_size(), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=130, pad_token_id=1, bos_token_id=0, eos_token_id=2,
        num_labels=3, id2label={0: "UR", 1: "EN", 2: "OTH"}, label2id={"UR": 0, "EN": 1, "OTH": 2},
    )
    XLMRobertaForTokenClassification(config).save_pretrained(folder)
    fast.save_pretrained(folder)


class TagTest(unittest.TestCase):
    def test_stage1_tags_for_the_running_example(self):
        expander = CandidateExpander.from_resources(RESOURCES)
        tags, readings = word_features(expander, EXAMPLE)
        got = dict(zip(EXAMPLE, zip(tags, readings)))
        self.assertEqual(got["Delivery"], ("EN", None))
        self.assertEqual(got["hui"], ("UR", None))
        self.assertEqual(got["to"], ("AMB", None))
        self.assertEqual(got["bht"], ("VAR", "bohat"))
        self.assertEqual(got["he"], ("BOTH", "hai"))
        self.assertEqual(stage1_tag(expander.expand_token("jio")), "UNK")
        self.assertIsNone(reading_of(expander.expand_token("main")))


@unittest.skipUnless(HAVE_TORCH, "torch/transformers/tokenizers not installed")
class CandidateModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        from src.classifier.candidate_model import CandidateCollator, CandidateXLMRForTokenClassification
        from src.classifier.features import piece_features

        cls.tmp = tempfile.TemporaryDirectory()
        cls.plain_dir = os.path.join(cls.tmp.name, "plain")
        build_tiny_checkpoint(cls.plain_dir)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.plain_dir)
        cls.expander = CandidateExpander.from_resources(RESOURCES)
        cls.model_class = CandidateXLMRForTokenClassification
        rows = []
        for words in (EXAMPLE, EXAMPLE[:6]):  # two lengths, so the batch is padded
            encoding = cls.tokenizer(words, is_split_into_words=True)
            tags, readings = word_features(cls.expander, words)
            row = {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}
            row.update(piece_features(encoding.word_ids(0), tags, readings, cls.tokenizer, use_readings=True))
            rows.append(row)
        cls.batch = CandidateCollator(cls.tokenizer)(rows)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self):
        model = self.model_class.from_pretrained(self.plain_dir)
        model.reset_candidate_parameters()
        return model.eval()

    def test_zeroed_candidates_reproduce_the_plain_model_exactly(self):
        plain = XLMRobertaForTokenClassification.from_pretrained(self.plain_dir).eval()
        with torch.no_grad():
            expected = plain(input_ids=self.batch["input_ids"], attention_mask=self.batch["attention_mask"]).logits
            got = self.load()(**self.batch).logits
        self.assertTrue(torch.allclose(expected, got, atol=1e-6), float((expected - got).abs().max()))

    def test_tag_and_reading_each_change_the_logits(self):
        model = self.load()
        with torch.no_grad():
            base = model(**self.batch).logits
            # A random vector: a constant one would be removed by the embedding LayerNorm.
            model.cand_embeddings.weight[CAND2ID["AMB"]] = torch.randn(model.config.hidden_size)
            with_tag = model(**self.batch).logits
            model.reading_gate.fill_(1.0)
            with_reading = model(**self.batch).logits
        self.assertGreater(float((with_tag - base).abs().max()), 1e-4)
        self.assertGreater(float((with_reading - with_tag).abs().max()), 1e-4)
        self.assertEqual(float(model.cand_embeddings.weight[CAND2ID["PAD"]].abs().sum()), 0.0)

    def test_train_cli_then_pipeline_end_to_end(self):
        from src.classifier.train import main as train_main
        from src.pipeline import LIDPipeline

        out = os.path.join(self.tmp.name, "cand")
        code = train_main([
            "--train", os.path.join(ROOT, "data", "demo", "seed_labels.csv"),
            "--valid", os.path.join(ROOT, "data", "demo", "running_examples.csv"),
            "--model", self.plain_dir, "--out", out, "--epochs", "2", "--candidates", "tag+reading",
        ])
        self.assertEqual(code, 0)
        reloaded = self.model_class.from_pretrained(out)
        self.assertEqual(reloaded.config.candidate_mode, "tag+reading")
        self.assertGreater(float(reloaded.cand_embeddings.weight.abs().sum()), 0.0)  # it trained
        saved = os.path.join(out, "stage1_resources")  # the lexicons its Stage-1 tags came from
        self.assertEqual(sorted(os.listdir(saved)), ["english_seed.txt", "roman_urdu_seed.txt", "variant_pairs.tsv"])
        pipeline = LIDPipeline.from_resources(checkpoint=out)
        self.assertEqual(pipeline.resources_source, "saved with the checkpoint")
        records = pipeline.run_records(" ".join(EXAMPLE))
        self.assertEqual([record["token"] for record in records], EXAMPLE)
        self.assertTrue(all(record["label"] in {"UR", "EN", "OTH"} for record in records))
        self.assertTrue(all(record["deferred"] == (record["confidence"] < 0.55) for record in records))

    def test_candidate_optimizer_matches_the_default_decay_rule(self):
        """The ablation arms must differ only in the input: same decay rule as the plain Trainer."""
        from transformers import Trainer, TrainingArguments

        from src.classifier.train import CANDIDATE_PARAMS, candidate_trainer

        model = self.load()
        args = TrainingArguments(output_dir=self.tmp.name, learning_rate=3e-5, weight_decay=0.01, report_to=[])
        plain = Trainer(model=model, args=args)
        plain.create_optimizer()
        cand = candidate_trainer(Trainer, cand_lr=1e-3)(model=model, args=args)
        cand.create_optimizer()

        def by_name(trainer):
            ids = {id(p): n for n, p in model.named_parameters()}
            return {ids[id(p)]: (g["lr"], g["weight_decay"]) for g in trainer.optimizer.param_groups for p in g["params"]}

        plain_groups, cand_groups = by_name(plain), by_name(cand)
        self.assertEqual(plain_groups.keys(), cand_groups.keys())
        self.assertIs(type(cand.optimizer), type(plain.optimizer))
        for name, (lr, decay) in cand_groups.items():
            if name.startswith(CANDIDATE_PARAMS):
                self.assertEqual((lr, decay), (1e-3, 0.0), name)
            else:
                self.assertEqual((lr, decay), plain_groups[name], name)
        self.assertTrue(any("LayerNorm" in name and cand_groups[name][1] == 0.0 for name in cand_groups))

    def test_plain_checkpoints_still_load_in_the_pipeline(self):
        from src.pipeline import LIDPipeline

        records = LIDPipeline.from_resources(checkpoint=self.plain_dir).run_records(" ".join(EXAMPLE))
        self.assertEqual(len(records), len(EXAMPLE))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class CalibrationTest(unittest.TestCase):
    def test_temperature_cools_an_over_confident_model(self):
        from src.classifier.calibrate import fit_temperature

        torch.manual_seed(0)
        logits = torch.randn(20000, 3) * 2.0
        labels = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)  # calibrated at T = 1
        temperature = fit_temperature(logits * 4.0, labels)  # sharpen x4 -> over-confident
        self.assertAlmostEqual(temperature, 4.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
