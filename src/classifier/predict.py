"""Stage-2 inference: word-level labels from a token-classification checkpoint."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WordPrediction:
    label: str
    confidence: float


class TokenLIDPredictor:
    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict_words(self, words: list[str]) -> list[WordPrediction]:
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in encoding.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0].to("cpu")
        probabilities = self.torch.softmax(logits, dim=-1)
        word_ids = encoding.word_ids(0)
        chosen: dict[int, WordPrediction] = {}
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id in chosen:
                continue
            confidence, label_id = probabilities[position].max(dim=-1)
            chosen[word_id] = WordPrediction(self.id2label[int(label_id)], float(confidence))
        return [chosen.get(index, WordPrediction("OTH", 0.0)) for index in range(len(words))]
