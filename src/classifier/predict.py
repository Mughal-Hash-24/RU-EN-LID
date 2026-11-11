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
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        config = AutoConfig.from_pretrained(checkpoint)
        # "none" for iteration-1 checkpoints; otherwise the model expects Stage 1's features.
        self.candidate_mode = getattr(config, "candidate_mode", "none")
        # Temperature fitted on validation (calibrate.py); 1.0 = raw softmax.
        self.temperature = float(getattr(config, "temperature", 1.0))
        if self.candidate_mode != "none":
            from .candidate_model import CandidateXLMRForTokenClassification

            self.model = CandidateXLMRForTokenClassification.from_pretrained(checkpoint)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def _logits(self, words: list[str], candidates=None):
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(0)
        inputs = {key: value.to(self.device) for key, value in encoding.items()}
        if self.candidate_mode != "none":
            if candidates is None or len(candidates) != len(words):
                raise ValueError("this checkpoint needs Stage 1 candidates, one per word")
            from .features import piece_features, reading_of, stage1_tag

            features = piece_features(
                word_ids,
                [stage1_tag(info) for info in candidates],
                [reading_of(info) for info in candidates],
                self.tokenizer,
                use_readings=self.candidate_mode == "tag+reading",
            )
            for key, value in features.items():
                inputs[key] = self.torch.tensor([value], dtype=self.torch.long, device=self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0].to("cpu")
        return logits, word_ids

    def word_logits(self, words: list[str], candidates=None) -> list:
        """Raw logits of each word's first piece (None if the word was truncated away)."""
        logits, word_ids = self._logits(words, candidates)
        first: dict[int, object] = {}
        for position, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in first:
                first[word_id] = logits[position]
        return [first.get(index) for index in range(len(words))]

    def predict_words(self, words: list[str], candidates=None) -> list[WordPrediction]:
        """``candidates``: Stage 1's TokenCandidates, one per word (required by candidate checkpoints)."""
        logits, word_ids = self._logits(words, candidates)
        probabilities = self.torch.softmax(logits / self.temperature, dim=-1)
        chosen: dict[int, WordPrediction] = {}
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id in chosen:
                continue
            confidence, label_id = probabilities[position].max(dim=-1)
            chosen[word_id] = WordPrediction(self.id2label[int(label_id)], float(confidence))
        return [chosen.get(index, WordPrediction("OTH", 0.0)) for index in range(len(words))]
