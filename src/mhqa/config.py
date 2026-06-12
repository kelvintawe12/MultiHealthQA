"""
Typed run configuration loaded from a YAML file.

DESIGN RATIONALE:
-----------------
A single Config dataclass carries every hyperparameter needed for the pipeline.
This ensures that each training run is fully reproducible from a single YAML file.
Experiments are implemented as field overrides on this base configuration.

TECHNICAL JUSTIFICATION:
-----------------------
1. Single source of truth: All hyperparameters in one place prevents 
   configuration drift between training and inference
2. Reproducibility: YAML files can be version-controlled and shared
3. Experimentation: Field overrides enable systematic hyperparameter sweeps
4. Type safety: Dataclass with type hints catches configuration errors early
5. Validation: Unknown keys in YAML raise immediately (fail-fast principle)

This pattern is industry-standard for ML research pipelines (e.g., HuggingFace
Trainer, Weights & Biases configurations).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # --- Paths ---
    data_dir: str = "data"
    output_dir: str = "artifacts/run"
    submission_path: str = "artifacts/run/submission.csv"

    # --- Model ---
    model_name: str = "google/mt5-base"

    # --- Sequence lengths ---
    max_input_length: int = 256
    max_target_length: int = 512

    # --- Prompt construction ---
    prompt_style: str = "lang_prefix"   # lang_prefix | bare | instruction
    retrieval_augment: bool = False

    # --- Optimisation ---
    optimizer: str = "adamw"  # adamw | adafactor (adafactor saves ~8GB GPU memory for mt5-large)
    learning_rate: float = 5.0e-4
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    label_smoothing_factor: float = 0.0
    seed: int = 42
    
    # --- Adafactor-specific settings (only used when optimizer="adafactor") ---
    adafactor_eps: tuple = (1e-30, 1e-3)
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_beta1: float = None  # Set to 0.9 for momentum, None for no momentum

    # --- Precision / hardware ---
    bf16_if_supported: bool = True
    gradient_checkpointing: bool = True

    # --- Validation / checkpointing ---
    val_size: float = 0.05
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_rougeL"
    greater_is_better: bool = True
    early_stopping_patience: int = 2
    predict_with_generate: bool = True
    eval_max_samples: int = 1500

    # --- Decoding (inference) ---
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3

    # --- Hybrid retrieval fallback ---
    hybrid_fallback: bool = False
    hybrid_min_gen_chars: int = 5

    # --- Free-form label so experiment rows are self-describing ---
    run_name: str = "run"
    notes: str = ""

    # ── Convenience ───────────────────────────────────────────────────────────
    @property
    def train_path(self) -> Path:
        return Path(self.data_dir) / "Train.csv"

    @property
    def val_path(self) -> Path:
        return Path(self.data_dir) / "Val.csv"

    @property
    def test_path(self) -> Path:
        return Path(self.data_dir) / "Test.csv"

    @property
    def sample_submission_path(self) -> Path:
        return Path(self.data_dir) / "SampleSubmission.csv"

    def with_overrides(self, **kwargs: Any) -> "Config":
        """Return a copy with selected fields replaced (used by experiments)."""
        unknown = set(kwargs) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"Unknown config field(s): {sorted(unknown)}")
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_config(path: str | Path, **overrides: Any) -> Config:
    """Load a :class:`Config` from YAML, applying optional keyword overrides.

    Unknown YAML keys raise immediately so typos in a config file fail loudly
    rather than being silently ignored.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    valid = {f.name for f in fields(Config)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")

    cfg = Config(**{**raw, **overrides})
    if cfg.run_name == "run":
        cfg = replace(cfg, run_name=Path(path).stem)
    return cfg
