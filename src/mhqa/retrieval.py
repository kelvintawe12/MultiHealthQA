"""Per-language TF-IDF char-ngram retrieval.

Two roles in this project:
1. **Baseline** (Experiment 1). For each test question, return the answer of the
   most similar *training* question. Measured ceiling on a CV holdout is
   ROUGE-1 ~0.44 / ROUGE-L ~0.38 — well below the leaderboard target, which is
   exactly why generation is the anchor and retrieval is only a support.
2. **Hybrid fallback / augmentation.** Supplies a nearest canonical answer to
   (a) fall back to when generation collapses (empty/degenerate output), or
   (b) prepend as a soft exemplar in retrieval-augmented prompting.

Character n-grams (``char_wb``) are used because they work across Latin,
Ge'ez and diacritic scripts without language-specific tokenisation, and casing
is preserved (non-Latin scripts carry meaning in case-like distinctions).

Matching happens in three tiers, falling back from most to least specific:
  subset (e.g. "Eng_Gha")  ->  language family (e.g. all "Eng_*" pooled)  ->  global.
Whichever tier returns the highest cosine similarity wins, so well-served
subsets keep their own precise index while sparse subsets can still borrow
signal from related data instead of being stuck with a small candidate pool.

Abugida/syllabary scripts (e.g. Amharic/Ge'ez) pack more phonetic information
per character than Latin letters, so the same ngram_range that works well for
English is often too coarse for them. Use ``ngram_overrides`` to set a smaller
range for a specific subset, e.g. ``ngram_overrides={"Amh_Eth": (1, 3)}``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .data import ANSWER_COL, LANG_COL, QUESTION_COL, clean_text, subset_to_language_name


class PerLanguageRetriever:
    """Nearest-neighbour answer retrieval with subset -> family -> global fallback."""

    def __init__(
        self,
        ngram_range=(3, 5),
        max_features: int = 200_000,
        ngram_overrides: Optional[dict] = None,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.ngram_overrides = ngram_overrides or {}
        self.models: dict[str, dict] = {}
        self.family_models: dict[str, dict] = {}
        self.global_model: Optional[dict] = None

    def _fit_one(self, df: pd.DataFrame, ngram_range=None) -> dict:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range or self.ngram_range,
            min_df=1,
            max_features=self.max_features,
            lowercase=False,  # preserve case for non-Latin scripts
        )
        # IMPORTANT: fit on the same clean_text() pipeline used at query time.
        # Previously the index was built on raw text while queries were
        # cleaned, which silently mismatched the character space used for
        # n-gram extraction on both sides.
        questions_raw = df[QUESTION_COL].fillna("").astype(str).tolist()
        questions_clean = [clean_text(q) for q in questions_raw]
        X = vec.fit_transform(questions_clean)
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)
        return {
            "vectorizer": vec,
            "nn": nn,
            "answers": df[ANSWER_COL].fillna("").astype(str).to_numpy(dtype=object),
            "questions": np.asarray(questions_raw, dtype=object),  # raw, for readable debug output
        }

    def fit(self, train: pd.DataFrame) -> "PerLanguageRetriever":
        self.global_model = self._fit_one(train)

        if LANG_COL in train.columns:
            # Tier 1: one index per exact subset (e.g. "Eng_Gha").
            for subset, grp in train.groupby(LANG_COL):
                if len(grp) >= 2:
                    self.models[subset] = self._fit_one(
                        grp, ngram_range=self.ngram_overrides.get(subset)
                    )

            # Tier 2: one index per language family, pooling sibling subsets
            # (e.g. all "Eng_*" variants together) so a thin subset can still
            # match against a related, larger pool.
            family = train[LANG_COL].map(subset_to_language_name)
            for fam, grp in train.groupby(family):
                if len(grp) >= 2:
                    self.family_models[fam] = self._fit_one(grp)

        return self

    def _query(self, question: str, model: dict):
        Xq = model["vectorizer"].transform([clean_text(question)])
        dist, idx = model["nn"].kneighbors(Xq, n_neighbors=1)
        i = int(idx[0][0])
        sim = 1.0 - float(dist[0][0])
        return model["answers"][i], sim, model["questions"][i]

    def predict_one(self, question: str, subset: Optional[str] = None):
        candidates = []

        if subset and subset in self.models:
            candidates.append(self._query(question, self.models[subset]))

        if subset:
            fam = subset_to_language_name(subset)
            if fam in self.family_models:
                candidates.append(self._query(question, self.family_models[fam]))

        candidates.append(self._query(question, self.global_model))

        # Tier 3 (global) is always a candidate; whichever tier has the
        # highest cosine similarity wins.
        return max(candidates, key=lambda c: c[1])

    def predict(self, df: pd.DataFrame):
        """Return (answers, similarities, matched_questions) aligned to ``df``."""
        answers, sims, matched = [], [], []
        subsets = df[LANG_COL] if LANG_COL in df.columns else [None] * len(df)
        for question, subset in zip(df[QUESTION_COL], subsets):
            a, s, q = self.predict_one(question, subset)
            answers.append(a)
            sims.append(s)
            matched.append(q)
        return answers, sims, matched
