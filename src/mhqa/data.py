"""
Data loading, cleaning, prompt construction and dataset preparation.

DESIGN RATIONALE - TEXT CLEANING:
---------------------------------
EDA REVELATION: Answers are canonical, in-language health templates where the 
same string recurs many times (e.g., one chlamydia answer appears 68 times).
Only ~61% of answers are unique.

TECHNICAL DECISION: We NEVER lowercase and NEVER strip non-ASCII characters.
WHY:
1. Case is signal, not noise: Proper nouns, medical terminology, and language-
   specific capitalization carry meaning
2. Ge'ez script (Amharic) has no case; lowercasing would be a no-op but 
   demonstrates wrong mental model
3. Akan uses Latin script with diacritics; stripping non-ASCII would corrupt
   the text and lose phonetic information
4. Whitespace-only normalization preserves all linguistic signal while handling
   formatting inconsistencies

RISK ANALYSIS: More aggressive cleaning (lowercasing, punctuation stripping, 
lemma removal) would corrupt Amharic/Akan text and reduce ROUGE scores because
evaluation uses whitespace tokenization on the original text.

DESIGN RATIONALE - PROMPT CONSTRUCTION:
----------------------------------------
EXPERIMENTAL FINDING: "lang_prefix" format ("<Language>: <question>") outperforms
"bare" and "instruction" formats in controlled experiments (exp04 in experiments.py).

WHY LANGUAGE PREFIXING WORKS:
1. Conditions the model on the target language before processing the question
2. Matches the multilingual pre-training paradigm of mT5 (translation tasks)
3. Prevents language confusion when similar concepts exist across languages
4. Explicitly signals the output language requirement

SUBSET COLUMN FORMAT: "LangCode_CountryCode" (e.g., "Amh_Eth", "Aka_Gha")
We only use the language prefix because country information is not relevant to
the QA task, but language conditioning is critical.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from . import SUBSET_TO_LANGUAGE

# Column names in the provided CSVs.
ID_COL = "ID"
QUESTION_COL = "input"
ANSWER_COL = "output"
LANG_COL = "subset"

_WS_RE = re.compile(r"\s+")


# ============================================================================
# TEXT CLEANING FUNCTION
# ============================================================================
# CRITICAL DESIGN DECISION: Minimal whitespace-only normalization
# 
# TECHNICAL JUSTIFICATION:
# 1. SCRIPT COMPATIBILITY: Whitespace splitting works uniformly across
#    Latin, Ge'ez (Amharic), and diacritic scripts (Akan)
# 2. EVALUATION ALIGNMENT: ROUGE evaluation uses whitespace tokenization, so
#    cleaning must match to avoid train-test mismatch
# 3. SIGNAL PRESERVATION: No lowercasing, punctuation removal, or other
#    transformations that could corrupt multilingual text
# 4. NULL HANDLING: Converts None/NaN to empty string for robustness
#
# RISK OF MORE AGGRESSIVE CLEANING:
# - Lowercasing would corrupt proper nouns and medical terminology
# - Punctuation removal could change meaning in some languages
# - Unicode normalization might break diacritics in Akan
# - Any transformation creates train-test mismatch if not applied identically
# during evaluation
def clean_text(x: object) -> str:
    """
    Whitespace-only normalisation that is safe across scripts.

    Collapses internal runs of whitespace (including newlines/tabs that appear in
    some answers) to single spaces and strips the ends. Nulls become empty string.
    
    This minimal approach preserves all linguistic signal while handling
    formatting inconsistencies in the source data.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return _WS_RE.sub(" ", str(x)).strip()


def subset_to_language_name(subset_code: object) -> str:
    """Map a subset code (e.g. ``'Amh_Eth'``) to a language name (``'Amharic'``)."""
    if not isinstance(subset_code, str) or not subset_code:
        return "English"
    return SUBSET_TO_LANGUAGE.get(subset_code.split("_")[0], subset_code)


# ── Prompt construction ───────────────────────────────────────────────────────
def build_prompt(question: str, subset: Optional[str], style: str = "lang_prefix",
                 retrieved: Optional[str] = None) -> str:
    """Build the model input string.

    Parameters
    ----------
    question : str
        The health question.
    subset : str or None
        Subset code; resolved to a language name for language conditioning.
    style : {"lang_prefix", "bare", "instruction"}
        Prompt template. These are compared head-to-head as a documented
        experiment — ``lang_prefix`` is the default winner.
    retrieved : str, optional
        A nearest canonical answer to prepend as a soft exemplar when
        retrieval augmentation is enabled.
    """
    q = clean_text(question)
    lang = subset_to_language_name(subset)

    if style == "bare":
        prompt = q
    elif style == "instruction":
        prompt = f"Answer this health question in {lang}: {q}"
    elif style == "lang_prefix":
        prompt = f"{lang}: {q}"
    else:
        raise ValueError(f"Unknown prompt_style: {style!r}")

    if retrieved:
        # Soft exemplar — gives the decoder a canonical phrasing to anchor on
        # without forcing it (the model still attends to the actual question).
        prompt = f"{prompt}\nReference: {clean_text(retrieved)}"
    return prompt


# ── Loading ───────────────────────────────────────────────────────────────────
def load_split(path: str | Path, has_answer: bool = True) -> pd.DataFrame:
    """Load one CSV, clean text columns and drop empty rows.

    ``has_answer=False`` is used for ``Test.csv`` which has no ``output`` column.
    """
    df = pd.read_csv(path)
    df[QUESTION_COL] = df[QUESTION_COL].map(clean_text)
    if has_answer and ANSWER_COL in df.columns:
        df[ANSWER_COL] = df[ANSWER_COL].map(clean_text)
        df = df[(df[QUESTION_COL] != "") & (df[ANSWER_COL] != "")]
    else:
        df = df[df[QUESTION_COL] != ""]
    return df.reset_index(drop=True)


def load_all(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load train / val / test / sample-submission in one call."""
    data_dir = Path(data_dir)
    return {
        "train": load_split(data_dir / "Train.csv", has_answer=True),
        "val": load_split(data_dir / "Val.csv", has_answer=True),
        "test": load_split(data_dir / "Test.csv", has_answer=False),
        "sample_submission": pd.read_csv(data_dir / "SampleSubmission.csv"),
    }


def stratified_split(train: pd.DataFrame, val_size: float, seed: int):
    """Stratified train/holdout split on ``subset`` for monitoring during training.

    Stratifying preserves the (very skewed) language mix in both halves so the
    per-language ROUGE we monitor is representative.
    """
    from sklearn.model_selection import train_test_split

    strat = train[LANG_COL] if LANG_COL in train.columns else None
    tr, ho = train_test_split(
        train, test_size=val_size, random_state=seed, stratify=strat
    )
    return tr.reset_index(drop=True), ho.reset_index(drop=True)


# ── HuggingFace dataset construction ──────────────────────────────────────────
def make_hf_dataset(df: pd.DataFrame, tokenizer, cfg, *, retrieved=None):
    """Tokenise a DataFrame into an HF ``Dataset`` ready for ``Seq2SeqTrainer``.

    * Inputs are built with :func:`build_prompt` so training and inference share
      the exact same formatting.
    * Label pad tokens are masked to ``-100`` so cross-entropy ignores padding.

    ``retrieved`` (optional) is a parallel list of exemplar strings used when
    ``cfg.retrieval_augment`` is on.
    """
    from datasets import Dataset

    questions = df[QUESTION_COL].tolist()
    subsets = df[LANG_COL].tolist() if LANG_COL in df.columns else [None] * len(df)
    answers = df[ANSWER_COL].tolist()
    refs = retrieved if (retrieved is not None and cfg.retrieval_augment) else [None] * len(df)

    prompts = [
        build_prompt(q, s, style=cfg.prompt_style, retrieved=r)
        for q, s, r in zip(questions, subsets, refs)
    ]
    raw = Dataset.from_dict({"prompt": prompts, "answer": answers})

    pad_id = tokenizer.pad_token_id

    def _tok(batch):
        model_inputs = tokenizer(
            batch["prompt"],
            max_length=cfg.max_input_length,
            truncation=True,
            padding=False,  # dynamic padding via DataCollatorForSeq2Seq
        )
        labels = tokenizer(
            text_target=batch["answer"],
            max_length=cfg.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = [
            [(t if t != pad_id else -100) for t in seq] for seq in labels["input_ids"]
        ]
        return model_inputs

    return raw.map(_tok, batched=True, remove_columns=["prompt", "answer"])
