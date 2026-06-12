"""Held-out evaluation helpers shared by the CLIs and the experiment runner.

Centralises the "load a fitted model, predict on a labelled split, report ROUGE
overall + per-language" routine so every script reports numbers the same way.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .data import ANSWER_COL, LANG_COL, load_split, stratified_split
from .infer import predict_dataframe
from .metrics import compute_rouge, compute_rouge_by_language


def evaluate_model(model, tokenizer, df: pd.DataFrame, cfg: Config, *, retriever=None,
                   batch_size: int = 8) -> tuple[dict, pd.DataFrame, list[str]]:
    """Predict on a labelled DataFrame and return (overall, per_language, preds)."""
    preds = predict_dataframe(model, tokenizer, df, cfg, retriever=retriever, batch_size=batch_size)
    refs = df[ANSWER_COL].tolist()
    overall = compute_rouge(preds, refs)
    per_lang = (
        compute_rouge_by_language(preds, refs, df[LANG_COL].tolist())
        if LANG_COL in df.columns else pd.DataFrame()
    )
    return overall, per_lang, preds


def holdout_for(cfg: Config) -> pd.DataFrame:
    """Reproduce the same stratified holdout used during training for scoring."""
    full = load_split(cfg.train_path, has_answer=True)
    _, holdout = stratified_split(full, cfg.val_size, cfg.seed)
    return holdout
