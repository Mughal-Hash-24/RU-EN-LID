"""MixLID baseline adapter (nlpai-lab/MixLID, MIT licence).

MixLID is a 2026 token-level LID model covering 111 language-script labels.
Its label inventory contains ``urd_Arab`` and ``eng_Latn`` but no ``urd_Latn``,
so Latin-script Roman Urdu cannot receive an Urdu label; predictions are mapped
to EN / UR / OTHER and the failure mode is reported with the evaluation.
"""

from __future__ import annotations

LABEL_MAP = {"eng_Latn": "EN", "urd_Arab": "UR", "urd_Latn": "UR"}


class MixLIDBaseline:
    def __init__(self, model_id: str = "nlpai-lab/MixLID", device: str = "cpu") -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def predict_words(self, words: list[str]) -> list[str]:
        text = " ".join(words)
        spans = self.model.predict_tokens(text)["spans"]
        labels: list[str] = []
        cursor = 0
        for word in words:
            start = text.find(word, cursor)
            if start < 0:
                labels.append("OTHER")
                continue
            end = start + len(word)
            cursor = end
            label = "OTHER"
            for span in spans:
                if span["start"] < end and span["end"] > start:
                    mapped = LABEL_MAP.get(span["label"], "OTHER")
                    if mapped != "OTHER":
                        label = mapped
                        break
            labels.append(label)
        return labels
