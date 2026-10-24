"""Stage-2 transformer training (token classification).

Requires torch + transformers (see requirements.txt). Use --smoke for a fast
end-to-end check with a tiny random checkpoint.
"""

from __future__ import annotations

import argparse

from .dataset import Sentence, read_any

LABELS = ["UR", "EN", "OTH"]
SMOKE_MODEL = "hf-internal-testing/tiny-random-XLMRobertaForTokenClassification"
DEFAULT_MODEL = "xlm-roberta-base"


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        import torch
        from transformers import (
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
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    def encode(sentences: list[Sentence]):
        encodings = tokenizer(
            [sentence.words for sentence in sentences],
            is_split_into_words=True,
            truncation=True,
            max_length=128,
        )
        aligned = []
        for index, sentence in enumerate(sentences):
            word_ids = encodings.word_ids(index)
            labels = []
            previous = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != previous:
                    labels.append(label2id[sentence.labels[word_id]])
                else:
                    labels.append(-100)
                previous = word_id
            aligned.append(labels)
        encodings["labels"] = aligned
        return encodings

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, index):
            return {key: torch.tensor(value[index]) for key, value in self.encodings.items()}

    train_dataset = TokenDataset(encode(train_sentences))
    eval_dataset = TokenDataset(encode(valid_sentences)) if valid_sentences else None

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = logits.argmax(-1)
        mask = labels != -100
        correct = (predictions[mask] == labels[mask]).sum().item()
        total = int(mask.sum().item())
        return {"token_accuracy": correct / total if total else 0.0}

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=4 if args.smoke else args.batch_size,
        per_device_eval_batch_size=4 if args.smoke else args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=0.01 if args.smoke else args.epochs,
        max_steps=10 if args.smoke else args.max_steps,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    if eval_dataset is not None:
        print(trainer.evaluate())
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved checkpoint to {args.out}")
    return 0
