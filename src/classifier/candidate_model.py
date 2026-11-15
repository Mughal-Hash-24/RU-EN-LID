"""Candidate-conditioned XLM-R: Stage 1's evidence enters at the input embeddings.

    e(piece) = word_emb(piece) + cand_emb(tag) + gate * mean(word_emb(reading pieces))

The sum then goes through XLM-R's usual position / LayerNorm / 12 layers, so
every layer can weigh the lexicon evidence against the sentence context. With
``cand_embeddings`` zeroed and ``reading_gate`` at 0 the model is exactly the
plain token classifier, so an iteration-1 checkpoint can be loaded and
fine-tuned, and the ablation "plain vs + candidates" differs in nothing else.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import XLMRobertaForTokenClassification

from .features import CAND_TAGS


class CandidateXLMRForTokenClassification(XLMRobertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.cand_embeddings = nn.Embedding(
            getattr(config, "num_cand_tags", len(CAND_TAGS)), config.hidden_size, padding_idx=0
        )
        self.reading_gate = nn.Parameter(torch.zeros(1))
        self.post_init()

    def reset_candidate_parameters(self) -> None:
        """Call once when starting from a plain checkpoint: step 0 == plain model."""
        nn.init.zeros_(self.cand_embeddings.weight)
        nn.init.zeros_(self.reading_gate)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        cand_ids=None,
        reading_ids=None,
        reading_mask=None,
        labels=None,
        token_type_ids=None,  # accepted and unused: XLM-R has a single segment
    ):  # no **kwargs: the Trainer would then pass loss kwargs that XLM-R's forward rejects
        word_embeddings = self.roberta.embeddings.word_embeddings
        embeds = word_embeddings(input_ids)
        if cand_ids is not None:
            embeds = embeds + self.cand_embeddings(cand_ids)
        if reading_ids is not None:
            pieces = word_embeddings(reading_ids)  # [batch, seq, pieces, hidden]
            mask = reading_mask.unsqueeze(-1).to(pieces.dtype)
            reading = (pieces * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
            embeds = embeds + self.reading_gate * reading
        # Same position ids XLM-R derives from input_ids (padding keeps the padding index).
        pad = self.config.pad_token_id
        not_pad = input_ids.ne(pad).int()
        position_ids = torch.cumsum(not_pad, dim=1) * not_pad + pad
        return super().forward(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )


class CandidateCollator:
    """Pads input ids, labels and the candidate features to the longest sentence in the batch."""

    def __init__(self, tokenizer) -> None:
        self.pad = tokenizer.pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        longest = max(len(item["input_ids"]) for item in batch)
        fill = {"input_ids": self.pad, "attention_mask": 0, "labels": -100, "cand_ids": 0}
        out: dict[str, torch.Tensor] = {}
        for key in batch[0]:
            rows = []
            for item in batch:
                row = list(item[key])
                missing = longest - len(row)
                if key == "reading_ids":
                    row += [[self.pad] * len(row[0])] * missing
                elif key == "reading_mask":
                    row += [[0] * len(row[0])] * missing
                else:
                    row += [fill[key]] * missing
                rows.append(row)
            out[key] = torch.tensor(rows, dtype=torch.long)
        return out
