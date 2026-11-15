"""Stage-2 inference: word-level labels from a token-classification checkpoint."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WordPrediction:
    label: str
    confidence: float
    # Full Stage-2 distribution (label -> probability), used by the Stage-3 context decoder.
    probs: dict[str, float] | None = None


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

    def _first_piece_logits(self, words: list[str], candidates=None):
        """Logits of each word's first subword piece, and which words have one.

        Returns ``(logits [k, labels], present)`` where ``present[i]`` is the row of word i,
        or None when truncation (512 pieces) removed the word.
        """
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

        positions: list[int] = []
        present: list[int | None] = [None] * len(words)
        for position, word_id in enumerate(word_ids):
            if word_id is not None and present[word_id] is None:
                present[word_id] = len(positions)
                positions.append(position)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, positions].float().cpu()
        return logits, present

    def word_logits(self, words: list[str], candidates=None) -> list:
        """Raw logits of each word's first piece (None if the word was truncated away)."""
        logits, present = self._first_piece_logits(words, candidates)
        return [None if row is None else logits[row] for row in present]

    def predict_words(self, words: list[str], candidates=None) -> list[WordPrediction]:
        """``candidates``: Stage 1's TokenCandidates, one per word (required by candidate checkpoints)."""
        logits, present = self._first_piece_logits(words, candidates)
        # One softmax over the k first pieces, then one transfer to Python lists.
        rows = self.torch.softmax(logits / self.temperature, dim=-1).tolist()
        labels = [self.id2label[index] for index in range(len(self.id2label))]
        predictions = []
        for row in present:
            if row is None:
                predictions.append(WordPrediction("OTH", 0.0, {"OTH": 1.0}))
                continue
            probs = dict(zip(labels, rows[row], strict=True))
            label = max(probs, key=probs.get)
            predictions.append(WordPrediction(label, probs[label], probs))
        return predictions
