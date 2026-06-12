"""
mhqa — Multilingual Health Question Answering in Low-Resource African Languages.

A small, modular toolkit for the Zindi MSRH challenge. The package separates the
pipeline into composable units so the notebook and the CLI scripts share exactly
one code path:

    config     — typed run configuration loaded from YAML
    data       — loading, cleaning, prompt construction, splits, HF datasets
    metrics    — script-safe whitespace ROUGE-1/L with per-language breakdown
    retrieval  — per-language TF-IDF char-ngram nearest-neighbour baseline/fallback
    train      — Seq2SeqTrainer wrapper with ROUGE-based model selection
    infer      — batched beam generation with sentinel cleanup + hybrid fallback
    submit     — validated 3-column submission writer
    experiments— registry of the documented leaderboard experiments
"""

__version__ = "0.1.0"

# Language prefix -> human-readable name, shared everywhere a prompt is built.
SUBSET_TO_LANGUAGE = {
    "Eng": "English",
    "Aka": "Akan",
    "Lug": "Luganda",
    "Swa": "Swahili",
    "Amh": "Amharic",
}

# The 8 language-country subsets present in the data (Lang_Country).
SUBSETS = [
    "Eng_Uga", "Aka_Gha", "Eng_Gha", "Eng_Eth",
    "Lug_Uga", "Eng_Ken", "Swa_Ken", "Amh_Eth",
]
