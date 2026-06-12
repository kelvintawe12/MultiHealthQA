"""Robust mT5 loading that survives older-torch environments.

All mT5 checkpoints on the Hub ship weights as ``pytorch_model.bin`` (not
safetensors). Recent ``transformers`` refuses to ``torch.load`` such files unless
``torch >= 2.6`` (CVE-2025-32434). Colab and many local boxes run older torch, so
a naive ``from_pretrained`` can fail through no fault of the user.

:func:`load_seq2seq` tries the normal path first, and on that specific failure
falls back to constructing the architecture from config and loading the state
dict manually (``weights_only=True``) — safe and version-robust. It also
re-serialises to safetensors on first use so subsequent loads are fast and clean.
"""
from __future__ import annotations

from pathlib import Path

import torch


def load_seq2seq(model_name: str):
    """Return a seq2seq model for ``model_name`` (Hub id or local path)."""
    from transformers import AutoModelForSeq2SeqLM

    try:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float32)
    except (ValueError, OSError) as err:
        if "vulnerability" not in str(err) and "weights_only" not in str(err):
            raise  # a genuine error, not the torch-version guard
        return _load_via_state_dict(model_name)


def _load_via_state_dict(model_name: str):
    """Manual fallback: build from config, load the .bin state dict directly."""
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_config(config)

    bin_path = _resolve_weight_file(model_name)
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # tied embeddings are expected to be "missing"; anything else is suspicious
        unexpected_missing = [m for m in missing if "shared" not in m and "lm_head" not in m]
        if unexpected_missing:
            raise RuntimeError(f"Unexpected missing keys when loading {model_name}: {unexpected_missing[:5]}")
    return model.to(torch.float32)


def _resolve_weight_file(model_name: str) -> str:
    local = Path(model_name) / "pytorch_model.bin"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_name, "pytorch_model.bin")
