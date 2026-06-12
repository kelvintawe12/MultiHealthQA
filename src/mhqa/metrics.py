"""
Evaluation metrics - script-safe ROUGE matching the platform's scoring.

DESIGN RATIONALE - WHITESPACE TOKENIZATION:
-------------------------------------------
TECHNICAL REQUIREMENT: The Zindi platform scores TargetRLF1 (ROUGE-L F1) and
TargetR1F1 (ROUGE-1 F1). Our evaluation must match the platform's tokenization
to ensure our local validation scores predict leaderboard performance.

WHY WHITESPACE TOKENIZATION:
1. LANGUAGE AGNOSTIC: Works uniformly across Latin, Ge'ez (Amharic), and
   diacritic scripts (Akan) without language-specific rules
2. SCRIPT COMPATIBILITY: Porter stemming, word tokenization, and other NLP
   preprocessing would silently mangle non-Latin scripts
3. EVALUATION ALIGNMENT: The platform likely uses whitespace tokenization for
   multilingual fairness; we match this for consistency
4. SIGNAL PRESERVATION: No linguistic information lost through aggressive
   tokenization that might work for English but fail for other scripts

RISK ANALYSIS - ALTERNATIVE APPROACHES:
- NLTK tokenizers: English-centric, would fail on Amharic Ge'ez script
- spaCy tokenizers: Require language-specific models, not available for all
  languages in our dataset (Akan, Luganda)
- Byte-pair encoding: Would create train-test mismatch unless platform uses
  identical tokenizer (unlikely)

WHY NOT ROUGE-S (STEAM): The platform evaluates ROUGE-1/L, not ROUGE-S.
Optimizing wrong metric would mislead our model selection.

TARGETLLM LIMITATION:
The LLM-as-a-Judge metric (TargetLLM) cannot be reproduced locally without
access to the judge model and evaluation rubric. This is a documented limitation
that should be discussed in the academic report as an evaluation strategy caveat.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer


class WhitespaceTokenizer:
    """Language-agnostic tokenizer: split on whitespace, no casing/stemming."""

    def tokenize(self, text):
        if text is None:
            return []
        return str(text).strip().split()


_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rougeL"], tokenizer=WhitespaceTokenizer(), use_stemmer=False
)


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> dict:
    """Mean ROUGE-1 / ROUGE-L F1 over aligned prediction/reference lists.

    Also returns the competition-style ``combined`` mean of the two ROUGE
    figures, a convenient single number for model selection / experiment ranking.
    """
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = _SCORER.score(str(ref), str(pred))
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    rouge1 = float(np.mean(r1)) if r1 else 0.0
    rougeL = float(np.mean(rl)) if rl else 0.0
    return {
        "rouge1_f1": rouge1,
        "rougeL_f1": rougeL,
        "combined": 0.5 * (rouge1 + rougeL),
    }


def compute_rouge_by_language(
    predictions: Sequence[str], references: Sequence[str], subsets: Sequence[str]
) -> pd.DataFrame:
    """Per-subset ROUGE table — surfaces the low-resource gap (Amh/Lug/Aka).

    Returns a DataFrame indexed by subset with rouge1_f1, rougeL_f1, combined and
    a row ``count``, sorted by combined score ascending so the weakest languages
    (where the leaderboard gap is won or lost) appear first.
    """
    df = pd.DataFrame(
        {"pred": list(predictions), "ref": list(references), "subset": list(subsets)}
    )
    rows = []
    for subset, grp in df.groupby("subset"):
        m = compute_rouge(grp["pred"].tolist(), grp["ref"].tolist())
        rows.append(
            {
                "subset": subset,
                "count": len(grp),
                "rouge1_f1": m["rouge1_f1"],
                "rougeL_f1": m["rougeL_f1"],
                "combined": m["combined"],
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("subset")
        .sort_values("combined")
    )


def build_compute_metrics(tokenizer):
    """Return a ``compute_metrics`` callable for ``Seq2SeqTrainer``.

    Decodes generated/label token IDs (handling the ``-100`` label mask) and
    reports ROUGE-1/L F1 each epoch, enabling ``metric_for_best_model=eval_rougeL``
    checkpoint selection and real learning curves.
    """

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.asarray(labels)
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        m = compute_rouge(decoded_preds, decoded_labels)
        return {"rouge1": m["rouge1_f1"], "rougeL": m["rougeL_f1"]}

    return compute_metrics
