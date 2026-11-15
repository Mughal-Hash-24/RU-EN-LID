"""Stage-2 transformer training (token classification).

Requires torch + transformers (see requirements.txt). Use --smoke for a fast
end-to-end check with a tiny random checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..eval.metrics import accuracy, macro_f1
from .dataset import IGNORE_LABEL, Sentence, read_any
from .features import CAND2ID, CAND_MODES, CAND_TAGS, piece_features, word_features

LABELS = ["UR", "EN", "OTH"]
# Parameters the candidate model adds on top of XLM-R (see candidate_model.py).
CANDIDATE_PARAMS = ("cand_embeddings", "reading_gate")
SMOKE_MODEL = "hf-internal-testing/tiny-random-XLMRobertaForTokenClassification"
DEFAULT_MODEL = "xlm-roberta-base"


def align_labels(word_ids: list[int | None], word_labels: list[str], label2id: dict[str, int]) -> list[int]:
    """Label ids per subword piece: a word's first piece carries its label; special tokens,
    continuation pieces and IGN words get -100, which the loss and the metrics skip."""
    labels = []
    previous = None
    for word_id in word_ids:
        if word_id is None or word_id == previous or word_labels[word_id] == IGNORE_LABEL:
            labels.append(-100)
        else:
            labels.append(label2id[word_labels[word_id]])
        previous = word_id
    return labels


def candidate_trainer(trainer_base, cand_lr: float):
    """A Trainer whose optimizer differs from the default in one way only: the new candidate
    parameters get their own learning rate and no weight decay.

    They start at zero, so at the encoder's usual rate they would barely move. Everything else
    (which encoder weights are decayed, the optimizer class and its settings) comes from the
    Trainer itself, so the "none" and candidate arms of the ablation are optimized alike.
    """

    class CandidateTrainer(trainer_base):
        def create_optimizer(self):
            if self.optimizer is None:
                decay = set(self.get_decay_parameter_names(self.model))
                groups = {"decay": [], "no_decay": [], "candidate": []}
                for name, parameter in self.model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if name.startswith(CANDIDATE_PARAMS):
                        groups["candidate"].append(parameter)
                    else:
                        groups["decay" if name in decay else "no_decay"].append(parameter)
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
                self.optimizer = optimizer_cls(
                    [
                        {"params": groups["decay"], "weight_decay": self.args.weight_decay},
                        {"params": groups["no_decay"], "weight_decay": 0.0},
                        {"params": groups["candidate"], "weight_decay": 0.0, "lr": cand_lr},
                    ],
                    **optimizer_kwargs,
                )
            return self.optimizer

    return CandidateTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="training CSV (seed or proxy LID format)")
    parser.add_argument("--valid", help="optional validation CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="checkpoints/ruen-lid")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0, help="cap training examples (0 = all)")
    parser.add_argument("--smoke", action="store_true", help="tiny random checkpoint, 10 steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=128, help="max subword pieces per sentence")
    parser.add_argument("--bf16", action="store_true", help="bfloat16 mixed precision (GPU)")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument(
        "--select-best", action="store_true", help="with --valid: score every epoch, keep the best validation macro-F1"
    )
    parser.add_argument(
        "--candidates",
        choices=CAND_MODES,
        default="none",
        help="feed Stage 1 into Stage 2: none = plain encoder, tag = evidence tag, tag+reading = tag and canonical",
    )
    parser.add_argument("--resources", help="Stage-1 lexicon folder (default: data/resources)")
    parser.add_argument(
        "--cand-dropout", type=float, default=0.15, help="per word: hide Stage 1's evidence (tag -> UNK) while training"
    )
    parser.add_argument("--cand-lr", type=float, default=1e-3, help="learning rate of the new candidate parameters")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print("torch/transformers are not installed. Create a Python 3.10-3.12 venv and run:")
        print("  pip install -r requirements.txt")
        return 2

    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}

    train_sentences: list[Sentence] = read_any(args.train)
    if args.limit:
        train_sentences = train_sentences[: args.limit]
    valid_sentences = read_any(args.valid) if args.valid else None

    model_name = SMOKE_MODEL if args.smoke else args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    use_candidates = args.candidates != "none"
    use_readings = args.candidates == "tag+reading"
    expander = None
    if use_candidates:
        from ..candidate_expansion.expander import CandidateExpander
        from ..pipeline import DEFAULT_RESOURCES
        from .candidate_model import CandidateCollator, CandidateXLMRForTokenClassification

        expander = CandidateExpander.from_resources(args.resources or DEFAULT_RESOURCES)
        config = AutoConfig.from_pretrained(model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id)
        config.num_cand_tags = len(CAND_TAGS)
        config.candidate_mode = args.candidates  # read back by predict.py
        model, loading = CandidateXLMRForTokenClassification.from_pretrained(
            model_name, config=config, output_loading_info=True
        )
        if any("cand_embeddings" in key for key in loading["missing_keys"]):
            model.reset_candidate_parameters()  # plain checkpoint: step 0 == the plain model
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id
        )

    def encode(sentences: list[Sentence]):
        encodings = tokenizer(
            [sentence.words for sentence in sentences],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
        )
        encodings["labels"] = [
            align_labels(encodings.word_ids(index), sentence.labels, label2id)
            for index, sentence in enumerate(sentences)
        ]
        return encodings

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, index):
            return {key: torch.tensor(value[index]) for key, value in self.encodings.items()}

    class CandidateDataset(torch.utils.data.Dataset):
        """Plain lists per sentence plus Stage 1's features; ``dropout`` hides evidence per word."""

        def __init__(self, sentences: list[Sentence], dropout: float = 0.0, seed: int = 0):
            encodings = encode(sentences)
            self.rows = []
            self.word_ids = []
            for index, sentence in enumerate(sentences):
                word_ids = encodings.word_ids(index)
                tags, readings = word_features(expander, sentence.words)
                row = {key: encodings[key][index] for key in ("input_ids", "attention_mask", "labels")}
                row.update(piece_features(word_ids, tags, readings, tokenizer, use_readings))
                self.rows.append(row)
                self.word_ids.append(word_ids)
            self.dropout = dropout
            self.generator = torch.Generator().manual_seed(seed)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = dict(self.rows[index])
            if self.dropout <= 0:
                return row
            word_ids = self.word_ids[index]
            words = max((w for w in word_ids if w is not None), default=-1) + 1
            hidden = (torch.rand(words, generator=self.generator) < self.dropout).tolist()
            unknown = CAND2ID["UNK"]
            row["cand_ids"] = [
                unknown if w is not None and hidden[w] else tag
                for w, tag in zip(word_ids, row["cand_ids"], strict=True)
            ]
            if use_readings:
                row["reading_mask"] = [
                    [0] * len(mask) if w is not None and hidden[w] else mask
                    for w, mask in zip(word_ids, row["reading_mask"], strict=True)
                ]
            return row

    if use_candidates:
        train_dataset = CandidateDataset(train_sentences, dropout=args.cand_dropout, seed=args.seed)
        eval_dataset = CandidateDataset(valid_sentences) if valid_sentences else None
        collator = CandidateCollator(tokenizer)
    else:
        train_dataset = TokenDataset(encode(train_sentences))
        eval_dataset = TokenDataset(encode(valid_sentences)) if valid_sentences else None
        collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = logits.argmax(-1)
        mask = labels != -100
        gold = [LABELS[index] for index in labels[mask].tolist()]
        pred = [LABELS[index] for index in predictions[mask].tolist()]
        return {"token_accuracy": accuracy(gold, pred), "macro_f1": macro_f1(gold, pred)}

    per_epoch = args.select_best and eval_dataset is not None
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=4 if args.smoke else args.batch_size,
        per_device_eval_batch_size=4 if args.smoke else args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=0.01 if args.smoke else args.epochs,
        max_steps=10 if args.smoke else args.max_steps,
        seed=args.seed,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if per_epoch else "no",
        save_strategy="epoch" if per_epoch else "no",
        save_only_model=per_epoch,
        save_total_limit=1 if per_epoch else None,
        load_best_model_at_end=per_epoch,
        metric_for_best_model="macro_f1" if per_epoch else None,
        report_to=[],
    )
    trainer_class = candidate_trainer(Trainer, args.cand_lr) if use_candidates else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    final_eval = trainer.evaluate() if eval_dataset is not None else None
    if final_eval is not None:
        print(final_eval)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    summary = {
        "final_eval": final_eval,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "eval_history": [entry for entry in trainer.state.log_history if "eval_loss" in entry],
    }
    Path(args.out, "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved checkpoint to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
