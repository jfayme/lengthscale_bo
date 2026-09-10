"""
lsab/featurizers/aimnet2.py
===========================

Mean-pooled per-atom hidden states of AIMNet2 (isayevlab/aimnetcentral).

Inside `AIMNet2.forward` the network refines a per-atom vector over several MLP
passes (`model.mlps`; three in the released wB97M-D3 model). The LAST pass writes
the 256-d `aim` vector that the energy and charge heads read. The calculator
throws `aim` away, so the passes are read with forward hooks rather than by
patching the library. Hooks only fire on an eager `nn.Module`: the released v2
`.pt` models are one, a legacy TorchScript `.jpt` is not, and is rejected.

The model runs in DENSE mode: a batch axis of 1 and no neighbour matrix. That is
exact all-pairs message passing for one small molecule, and needs no
neighbour-list builder at run time.

    layers="last": the 256-d `aim` vector.
    layers="all" : every pass concatenated in order, 258 + 258 + 256 = 772 for the
                   released model (the first two passes carry two extra channels).

Deterministic on CPU (seeded conformer, eval mode, no_grad). GPU results may
differ in the last bits between runs.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from aimnet.calculators import AIMNet2Calculator

from lsab.featurizers.conformer import smiles_to_atoms


class AIMNet2Featurizer:
    """Mean-pooled per-atom hidden state of AIMNet2 (dense, all-pairs mode).
    layers="last": the 256-d `aim` vector that feeds the output heads.
    layers="all" : every MLP pass concatenated (the layer-ablation winner in the A/B)."""

    def __init__(self, layers: Literal["last", "all"] = "all", model_name: str = "aimnet2",
                 device: str = "cpu", seed: int = 42):
        if layers not in ("last", "all"):
            raise ValueError(f"layers must be 'last' or 'all', got {layers!r}")
        self.name = "aimnet2" if layers == "last" else "aimnet2_all"
        self.device, self.seed = device, seed
        self.model = AIMNet2Calculator(model_name, device=device).model.eval()
        if isinstance(self.model, torch.jit.ScriptModule):
            raise RuntimeError(f"{model_name!r} loaded as TorchScript, where forward hooks "
                               "never fire; use a v2 .pt model such as 'aimnet2'")
        self.mlps = list(self.model.mlps) if layers == "all" else [self.model.mlps[-1]]
        self.d = len(self._embed_one("C"))   # probe once: the width belongs to the checkpoint
        if layers == "last" and self.d != 256:
            raise RuntimeError(f"expected the 256-d `aim` vector, got d={self.d}")

    def _inputs(self, smiles: str) -> dict:
        numbers, coords, charge = smiles_to_atoms(smiles, seed=self.seed)
        return {  # batch axis of 1 -> dense mode, no padding atom
            "coord": torch.as_tensor(coords, dtype=torch.float32, device=self.device).unsqueeze(0),
            "numbers": torch.as_tensor(numbers, dtype=torch.long, device=self.device).unsqueeze(0),
            "charge": torch.as_tensor([float(charge)], dtype=torch.float32, device=self.device),
        }

    def _embed_one(self, smiles: str) -> np.ndarray:
        data = self._inputs(smiles)
        captured = {}
        handles = [mlp.register_forward_hook(
                       lambda _module, _input, output, i=i: captured.__setitem__(i, output.detach()))
                   for i, mlp in enumerate(self.mlps)]
        try:
            with torch.no_grad():
                try:
                    self.model(data)
                except Exception:
                    # The output heads run AFTER the last MLP. If every hook has
                    # already fired, a head that dislikes dense mode costs nothing.
                    if len(captured) < len(self.mlps):
                        raise
        finally:
            for handle in handles:
                handle.remove()
        per_atom = torch.cat([captured[i] for i in range(len(self.mlps))], dim=-1).squeeze(0)
        return per_atom.cpu().numpy().mean(axis=0, dtype=np.float64).astype(np.float32)

    def __call__(self, smiles: list[str]) -> np.ndarray:
        return np.asarray([self._embed_one(s) for s in smiles], dtype=np.float32)
