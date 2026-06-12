"""Exploratory data analysis for the Multilingual Health QA challenge.

Runs CPU-only and regenerates every statistic the project relies on, plus a set of
publication-ready figures saved to ``reports/figures/``. The findings here are
what motivate the modelling choices (canonical templated answers, the retrieval
ceiling, the low-resource language gap, length-driven ``max_target_length``).

Usage
-----
    python -m scripts.run_eda                 # uses ./data
    python -m scripts.run_eda --data-dir data --out reports
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / CPU-only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make ``src`` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mhqa.data import ANSWER_COL, LANG_COL, QUESTION_COL, load_all  # noqa: E402


def _latin_fraction(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA for Multilingual Health QA")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    fig_dir = Path(args.out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(args.data_dir)
    train, val, test = data["train"], data["val"], data["test"]

    print("=" * 70)
    print("DATASET SHAPES")
    print("=" * 70)
    for name in ("train", "val", "test"):
        print(f"  {name:<6}: {data[name].shape}")

    # ── 1. Language-country distribution ─────────────────────────────────────
    dist = pd.DataFrame(
        {
            "train": train[LANG_COL].value_counts(),
            "val": val[LANG_COL].value_counts(),
            "test": test[LANG_COL].value_counts(),
        }
    ).fillna(0).astype(int)
    dist = dist.sort_values("train", ascending=False)
    print("\nSUBSET DISTRIBUTION (Lang_Country):\n", dist.to_string())

    ax = dist[["train", "test"]].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Language-country distribution (train vs test)")
    ax.set_ylabel("rows")
    ax.set_xlabel("subset")
    plt.tight_layout()
    plt.savefig(fig_dir / "01_subset_distribution.png", dpi=150)
    plt.close()

    # ── 2. Answer / question length ──────────────────────────────────────────
    train = train.copy()
    train["q_words"] = train[QUESTION_COL].str.split().apply(len)
    train["a_words"] = train[ANSWER_COL].str.split().apply(len)
    print("\nLENGTH (words):")
    print(f"  question  mean={train.q_words.mean():.1f}  median={train.q_words.median():.0f}")
    print(
        f"  answer    mean={train.a_words.mean():.1f}  median={train.a_words.median():.0f}"
        f"  p90={train.a_words.quantile(.9):.0f}  p99={train.a_words.quantile(.99):.0f}"
        f"  max={train.a_words.max():.0f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train.q_words, bins=50, color="#4C72B0")
    axes[0].set_title("Question length (words)")
    axes[0].axvline(train.q_words.median(), color="k", ls="--", lw=1)
    axes[1].hist(train.a_words.clip(upper=300), bins=50, color="#55A868")
    axes[1].set_title("Answer length (words, clipped@300)")
    axes[1].axvline(train.a_words.median(), color="k", ls="--", lw=1)
    plt.tight_layout()
    plt.savefig(fig_dir / "02_length_distributions.png", dpi=150)
    plt.close()

    # ── 3. Answer templating (why ROUGE targets are high) ────────────────────
    uniq = train[ANSWER_COL].nunique()
    print("\nANSWER TEMPLATING:")
    print(f"  unique answers: {uniq}/{len(train)} ({uniq/len(train):.1%})")
    top = train[ANSWER_COL].value_counts().head(10)
    print("  most repeated answer appears", int(top.iloc[0]), "times")
    shared = train.groupby(ANSWER_COL)[LANG_COL].nunique()
    print(f"  answers reused across >1 subset: {(shared > 1).sum()} ({(shared > 1).mean():.1%})")

    ax = top[::-1].plot(kind="barh", figsize=(9, 5), color="#C44E52")
    ax.set_title("Top-10 most repeated answers (templating)")
    ax.set_xlabel("frequency")
    ax.set_yticklabels([f"#{i+1}" for i in range(len(top))][::-1])
    plt.tight_layout()
    plt.savefig(fig_dir / "03_answer_templating.png", dpi=150)
    plt.close()

    # ── 4. Script composition per subset ─────────────────────────────────────
    train["a_latin"] = train[ANSWER_COL].map(_latin_fraction)
    script = train.groupby(LANG_COL)["a_latin"].mean().sort_values()
    print("\nSCRIPT (mean Latin-letter fraction of answers) by subset:\n", script.round(3).to_string())
    ax = script.plot(kind="bar", figsize=(9, 4), color="#8172B3")
    ax.set_title("Mean Latin-letter fraction of answers (Amharic ~0 = Ge'ez script)")
    ax.set_ylabel("Latin fraction")
    plt.tight_layout()
    plt.savefig(fig_dir / "04_script_composition.png", dpi=150)
    plt.close()

    # ── 5. Retrieval ceiling (CV) — generation must beat this ────────────────
    _retrieval_ceiling(train)

    print(f"\n✅ EDA complete — figures written to {fig_dir}/")


def _retrieval_ceiling(train: pd.DataFrame) -> None:
    """Estimate the per-language nearest-neighbour answer ceiling via a CV split."""
    from sklearn.model_selection import train_test_split

    from mhqa.metrics import compute_rouge
    from mhqa.retrieval import PerLanguageRetriever

    trn, ho = train_test_split(
        train, test_size=0.1, random_state=42, stratify=train[LANG_COL]
    )
    retr = PerLanguageRetriever().fit(trn.reset_index(drop=True))
    preds, _, _ = retr.predict(ho.reset_index(drop=True))
    m = compute_rouge(preds, ho[ANSWER_COL].tolist())
    print("\nRETRIEVAL CEILING (per-language char-ngram, 10% CV holdout):")
    print(f"  ROUGE-1={m['rouge1_f1']:.4f}  ROUGE-L={m['rougeL_f1']:.4f}")
    print("  -> generation must clear this floor; retrieval is a fallback only.")


if __name__ == "__main__":
    main()
