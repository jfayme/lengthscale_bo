"""
lsab/featurizers/lm.py
======================

Language-model embeddings of the SMILES string: a chemistry T5 encoder and
ChemBERTa. Both are pooled the SAME way, by a masked mean over the LAST hidden
state (padding tokens excluded), with no L2 normalisation. One pooler means a
difference between the two is a difference between the models, not the recipe.

Weights come from the HuggingFace cache, i.e. wherever `HF_HOME` points; no cache
directory is hard-coded. Batches of 16, truncated at 512 tokens.

Deterministic on CPU (eval mode, no_grad). GPU results may differ in the last
bits between runs.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

BATCH_SIZE = 16
MAX_LENGTH = 512


def masked_mean(hidden, attention_mask):
    """(B, L, D), (B, L) -> (B, D): the mean over real tokens only. The one pooler."""
    real = attention_mask.unsqueeze(-1).bool()
    return hidden.masked_fill(~real, 0.0).sum(dim=1) / real.sum(dim=1)


class _MaskedMeanEncoder:
    """Tokenise in batches, run the encoder, masked-mean the last hidden state."""

    name: str

    def __init__(self, tokenizer, model, device: str):
        self.tokenizer, self.device = tokenizer, device
        self.model = model.float().to(device).eval()

    def __call__(self, smiles: list[str]) -> np.ndarray:
        pooled = []
        for start in range(0, len(smiles), BATCH_SIZE):
            batch = self.tokenizer(list(smiles[start:start + BATCH_SIZE]), padding=True,
                                   truncation=True, max_length=MAX_LENGTH,
                                   return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"]).last_hidden_state
            pooled.append(masked_mean(hidden, batch["attention_mask"]).cpu().numpy())
        return np.concatenate(pooled).astype(np.float32)


class T5Featurizer(_MaskedMeanEncoder):
    """GT4SD's multitask text-and-chemistry T5 (base), encoder only.

    The published GOLLuM runs used `t5-base`, a generic English model, because
    this checkpoint would not download on that machine. The rewrite uses the
    chemistry checkpoint everywhere. The tokenizer is the slow SentencePiece one
    with legacy handling, as in the old runs, pinned so that a transformers
    upgrade cannot silently re-tokenise the SMILES.
    """

    name = "t5"

    def __init__(self, model_name: str = "GT4SD/multitask-text-and-chemistry-t5-base-augm",
                 device: str = "cpu"):
        super().__init__(AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True),
                         T5EncoderModel.from_pretrained(model_name), device)


class ChemBERTaFeaturizer(_MaskedMeanEncoder):
    """seyonec/ChemBERTa-zinc-base-v1 (RoBERTa), masked mean of the LAST hidden state.

    DECISION: the HSF-ChemBO baseline summed `hidden_states[0]`, the token-embedding
    layer BEFORE any transformer block, so no transformer block ever touched that
    vector. The rewrite pools the last hidden state, like T5. `add_pooling_layer=False`
    because the checkpoint has no pooler; building one would only initialise
    unused random weights.
    """

    name = "chemberta"

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", device: str = "cpu"):
        super().__init__(AutoTokenizer.from_pretrained(model_name),
                         AutoModel.from_pretrained(model_name, add_pooling_layer=False), device)
