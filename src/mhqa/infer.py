"""Batched generation with mT5 sentinel cleanup and an optional hybrid fallback.

The same :func:`generate_answers` is used for the per-epoch sanity checks, the
held-out evaluation, and the final test prediction, so what we measure is exactly
what we submit.

Two robustness features:

* **Sentinel cleanup** — mT5 occasionally emits ``<extra_id_N>`` pretraining
  sentinels; we strip them so they never leak into a submission.
* **Hybrid fallback** — if generation collapses (empty / too short / pure
  sentinels), substitute the nearest canonical answer from the retriever. This
  protects the worst-case rows that otherwise score ~0 ROUGE and drag the mean.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import torch
from tqdm.auto import tqdm

from .config import Config
from .data import LANG_COL, QUESTION_COL, build_prompt

_SENTINEL_RE = re.compile(r"<extra_id_\d+>")


def _clean_generation(text: str) -> str:
    return _SENTINEL_RE.sub(" ", text or "").replace("  ", " ").strip()


def generate_answers(
    model,
    tokenizer,
    questions: Sequence[str],
    subsets: Optional[Sequence[str]],
    cfg: Config,
    *,
    retrieved: Optional[Sequence[str]] = None,
    batch_size: int = 8,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> list[str]:
    """Generate answers for ``questions`` with the same prompt format as training."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    if getattr(model.config, "use_cache", None) is False:
        model.config.use_cache = True  # re-enable for fast generation

    if subsets is None:
        subsets = [None] * len(questions)
    aug = retrieved if (retrieved is not None and cfg.retrieval_augment) else [None] * len(questions)

    outputs: list[str] = []
    n_batches = (len(questions) + batch_size - 1) // batch_size
    rng = range(n_batches)
    for b in tqdm(rng, disable=not show_progress, desc="generate"):
        s, e = b * batch_size, min((b + 1) * batch_size, len(questions))
        prompts = [
            build_prompt(q, sub, style=cfg.prompt_style, retrieved=r)
            for q, sub, r in zip(questions[s:e], subsets[s:e], aug[s:e])
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.max_input_length,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=cfg.max_target_length,
                num_beams=cfg.num_beams,
                length_penalty=cfg.length_penalty,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                early_stopping=True,
            )
        outputs.extend(_clean_generation(t) for t in tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def apply_hybrid_fallback(
    generations: Sequence[str],
    retrieved_answers: Sequence[str],
    cfg: Config,
) -> list[str]:
    """Replace collapsed generations with the retrieved canonical answer."""
    out = []
    for gen, ret in zip(generations, retrieved_answers):
        g = (gen or "").strip()
        out.append(ret if len(g) < cfg.hybrid_min_gen_chars else g)
    return out


def predict_dataframe(model, tokenizer, df, cfg: Config, *, retriever=None,
                      batch_size: int = 8) -> list[str]:
    """End-to-end prediction for a DataFrame: generate, then (optionally) hybridise.

    If ``cfg.retrieval_augment`` or ``cfg.hybrid_fallback`` is set, ``retriever``
    must be a fitted :class:`~mhqa.retrieval.PerLanguageRetriever`.
    """
    questions = df[QUESTION_COL].tolist()
    subsets = df[LANG_COL].tolist() if LANG_COL in df.columns else None

    retrieved = None
    if retriever is not None and (cfg.retrieval_augment or cfg.hybrid_fallback):
        retrieved, _, _ = retriever.predict(df)

    gens = generate_answers(
        model, tokenizer, questions, subsets, cfg,
        retrieved=retrieved, batch_size=batch_size,
    )
    if cfg.hybrid_fallback and retrieved is not None:
        gens = apply_hybrid_fallback(gens, retrieved, cfg)
    return gens
