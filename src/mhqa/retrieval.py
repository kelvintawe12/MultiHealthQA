"""Per-language TF-IDF char-ngram retrieval.

Two roles in this project:

1. **Baseline** (Experiment 1). For each test question, return the answer of the
   most similar *training* question. Measured ceiling on a CV holdout is
   ROUGE-1 ~0.44 / ROUGE-L ~0.38 — well below the leaderboard target, which is
   exactly why generation is the anchor and retrieval is only a support.
2. **Hybrid fallback / augmentation.** Supplies a nearest canonical answer to
   (a) fall back to when generation collapses (empty/degenerate output), or
   (b) prepend as a soft exemplar in retrieval-augmented prompting.

Character n-grams (``char_wb``, 3–5) are used because they work across Latin,
Ge'ez and diacritic scripts without language-specific tokenisation, and casing is
preserved (non-Latin scripts carry meaning in case-like distinctions).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .data import ANSWER_COL, LANG_COL, QUESTION_COL, clean_text


class PerLanguageRetriever:
    """Nearest-neighbour answer retrieval with a per-subset index + global fallback."""

    def __init__(self, ngram_range=(3, 5), max_features=200_000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.models: dict[str, dict] = {}
        self.global_model: Optional[dict] = None

    def _fit_one(self, df: pd.DataFrame) -> dict:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=1,
            max_features=self.max_features,
            lowercase=False,  # preserve case for non-Latin scripts
        )
        questions = df[QUESTION_COL].fillna("").astype(str).tolist()
        X = vec.fit_transform(questions)
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)
        return {
            "vectorizer": vec,
            "nn": nn,
            "answers": df[ANSWER_COL].fillna("").astype(str).to_numpy(dtype=object),
            "questions": np.asarray(questions, dtype=object),
        }

    def fit(self, train: pd.DataFrame) -> "PerLanguageRetriever":
        self.global_model = self._fit_one(train)
        if LANG_COL in train.columns:
            for subset, grp in train.groupby(LANG_COL):
                if len(grp) >= 2:
                    self.models[subset] = self._fit_one(grp)
        return self

    def _query(self, question: str, model: dict):
        Xq = model["vectorizer"].transform([clean_text(question)])
        dist, idx = model["nn"].kneighbors(Xq, n_neighbors=1)
        i = int(idx[0][0])
        sim = 1.0 - float(dist[0][0])
        return model["answers"][i], sim, model["questions"][i]

    def predict_one(self, question: str, subset: Optional[str] = None):
        model = self.models.get(subset, self.global_model) if subset else self.global_model
        return self._query(question, model)

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
