"""Documented experiment registry — the spine of the rubric's experiment track.

Each experiment is a named *delta* over a base :class:`~mhqa.config.Config`, plus a
one-line hypothesis. Running an experiment fine-tunes the model with the override,
evaluates ROUGE on the held-out split (overall + per-language), and appends a row
to ``reports/experiments.csv``. This keeps the *what changed / why / outcome /
insight* story machine-tracked and reproducible rather than hand-written.

The 12 registered experiments map 1:1 to the plan's experiment list and cover
prompting, model scale, decoding, length, optimisation, retrieval augmentation,
hybrid inference, low-resource handling, and ensembling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import Config


@dataclass
class Experiment:
    key: str
    title: str
    hypothesis: str
    # Override dict applied to the base config (None => special-cased runner).
    overrides: dict[str, Any]
    # Optional custom runner for non-finetune experiments (e.g. retrieval-only).
    runner: Optional[str] = None


# ── Registry ──────────────────────────────────────────────────────────────────
EXPERIMENTS: list[Experiment] = [
    Experiment(
        key="exp01_retrieval_baseline",
        title="TF-IDF retrieval baseline",
        hypothesis="Nearest canonical answer sets the non-generative floor (~0.44 R1).",
        overrides={},
        runner="retrieval",
    ),
    Experiment(
        key="exp02_zeroshot_mt5",
        title="Zero-shot mT5-base (no fine-tune)",
        hypothesis="Pretrained mT5 cannot do in-language health QA without fine-tuning.",
        overrides={"num_train_epochs": 0},
        runner="zeroshot",
    ),
    Experiment(
        key="exp03_mt5base_bare",
        title="mT5-base fine-tune, bare-question prompt",
        hypothesis="Fine-tuning on canonical answers clears the retrieval floor.",
        overrides={"prompt_style": "bare"},
    ),
    Experiment(
        key="exp04_mt5base_langprefix",
        title="mT5-base fine-tune, language-prefix prompt",
        hypothesis="Conditioning on the target language improves in-language fidelity.",
        overrides={"prompt_style": "lang_prefix"},
    ),
    Experiment(
        key="exp05_mt5base_instruction",
        title="mT5-base fine-tune, instruction-style prompt",
        hypothesis="A verbose instruction is not better than a compact language prefix.",
        overrides={"prompt_style": "instruction"},
    ),
    Experiment(
        key="exp06_mt5large",
        title="mT5-large fine-tune (scale up)",
        hypothesis="More capacity lifts the hard low-resource subsets (Amh/Lug/Aka).",
        overrides={
            "model_name": "google/mt5-large",
            "learning_rate": 3.0e-4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 4,
        },
    ),
    Experiment(
        key="exp07_decoding_beams6",
        title="Decoding sweep: beams=6, length_penalty=1.1",
        hypothesis="Wider beams + length favouring completeness raise ROUGE recall.",
        overrides={"num_beams": 6, "length_penalty": 1.1},
    ),
    Experiment(
        key="exp08_target_len_256",
        title="Target length 256 vs 512",
        hypothesis="Truncating answers to 256 tokens loses recall on long templates.",
        overrides={"max_target_length": 256},
    ),
    Experiment(
        key="exp09_lr_smoothing",
        title="LR + label smoothing sweep",
        hypothesis="Lower LR with mild label smoothing trades a little ROUGE for fluency.",
        overrides={"learning_rate": 2.0e-4, "label_smoothing_factor": 0.1, "num_train_epochs": 4},
    ),
    Experiment(
        key="exp10_retrieval_augment",
        title="Retrieval-augmented prompting",
        hypothesis="Prepending a canonical exemplar anchors phrasing and lifts ROUGE.",
        overrides={"retrieval_augment": True},
    ),
    Experiment(
        key="exp11_hybrid_fallback",
        title="Hybrid inference (retrieval fallback)",
        hypothesis="Replacing collapsed generations with retrieval removes near-zero rows.",
        overrides={"hybrid_fallback": True},
    ),
    Experiment(
        key="exp12_oversample_lowresource",
        title="Oversample low-resource subsets",
        hypothesis="Up-weighting Amh/Lug/Swa/Aka closes the per-language gap.",
        overrides={},
        runner="oversample",
    ),
]

EXPERIMENTS_BY_KEY = {e.key: e for e in EXPERIMENTS}


def resolve_config(base: Config, exp: Experiment) -> Config:
    """Apply an experiment's overrides on top of a base config."""
    cfg = base.with_overrides(**exp.overrides) if exp.overrides else base
    return cfg.with_overrides(
        run_name=exp.key,
        output_dir=f"artifacts/{exp.key}",
        submission_path=f"artifacts/{exp.key}/submission.csv",
        notes=exp.hypothesis,
    )
