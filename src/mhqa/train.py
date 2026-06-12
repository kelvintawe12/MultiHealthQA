"""
Fine-tuning wrapper around HuggingFace Seq2SeqTrainer.

DESIGN RATIONALE - TRAINING STRATEGY:
-------------------------------------
This module encapsulates every training decision for reproducibility from a
single Config object. Key design decisions:

1. LABEL MASKING (-100): HuggingFace convention to ignore padding tokens in
   loss computation. Critical for variable-length sequences.
   
2. DYNAMIC PADDING: DataCollatorForSeq2Seq pads to longest sequence in batch
   (not fixed length) for efficiency. Pad to multiple of 8 for tensor-core
   optimization on modern GPUs.
   
3. BF16 OVER FP16: bfloat16 has better numerical stability than fp16 for
   seq2seq tasks because it doesn't require gradient scaling. fp16 can fail
   with "unscale FP16 gradients" error when weights are fp32. Falls back
   gracefully to fp16 on pre-Ampere hardware.
   
4. ROUGE-DRIVEN MODEL SELECTION: metric_for_best_model=eval_rougeL selects
   the best generator (not lowest cross-entropy). This is critical because
   cross-entropy loss doesn't directly correlate with ROUGE score.
   
5. PER-EPOCH EVALUATION: ROUGE evaluation each epoch with early stopping
   prevents overfitting on the dominant English subsets.

TECHNICAL JUSTIFICATION - ADAFACTOR VS ADAMW:
---------------------------------------------
MEMORY CONSTRAINTS:
- AdamW optimizer state: ~19 GB for mt5-large (2 parameters per weight)
- Adafactor optimizer state: <1 GB (factored approximation)
- Total memory with AdamW: ~30 GB (beyond 16 GB GPU capacity)
- Total memory with Adafactor: ~12 GB (fits 16 GB GPU)

ADAFCTOR ADVANTAGES:
1. Memory efficiency: Uses factored approximation of second moments
2. Scalability: Designed for large models (used in original T5 paper)
3. No per-parameter learning rate matrix: Linear memory vs quadratic

ADAFCTOR TRADEOFFS:
1. Slower convergence: Requires more steps or higher learning rate
2. Less theoretically grounded: Heuristic vs Adam's adaptive rates
3. Implementation complexity: More hyperparameters to tune

DECISION: Adafactor is the only viable option for mt5-large on 16GB GPUs.
The convergence tradeoff is acceptable because:
- We compensate with higher learning rate (1e-3 vs 5e-4)
- More training epochs (4 vs 3)
- The memory savings enable using a larger model (1.2B vs 580M)
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .config import Config
from .data import (
    LANG_COL,
    load_split,
    make_hf_dataset,
    stratified_split,
)
from .metrics import build_compute_metrics


def _resolve_precision(cfg: Config):
    """Return (bf16, fp16) flags appropriate for the current hardware."""
    if not torch.cuda.is_available():
        return False, False
    if cfg.bf16_if_supported and torch.cuda.is_bf16_supported():
        return True, False
    return False, True  # fp16 on pre-Ampere CUDA


def _create_optimizer(cfg, model):
    """Create the optimizer based on config (AdamW or Adafactor for memory efficiency)."""
    if cfg.optimizer == "adafactor":
        from transformers import Adafactor
        return Adafactor(
            model.parameters(),
            lr=cfg.learning_rate,
            eps=cfg.adafactor_eps,
            clip_threshold=cfg.adafactor_clip_threshold,
            decay_rate=cfg.adafactor_decay_rate,
            beta1=cfg.adafactor_beta1,
            weight_decay=cfg.weight_decay,
            relative_step=False,
        )
    else:  # adamw (default)
        from transformers import AdamW
        return AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def train(cfg: Config, *, train_df=None, holdout_df=None):
    """Fine-tune an mT5 model from ``cfg`` and return (trainer, tokenizer).

    ``train_df`` / ``holdout_df`` may be supplied (e.g. by an experiment that
    pre-subsamples or oversamples languages); otherwise they are derived from
    ``Train.csv`` via a stratified split.
    """
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    from .modeling import load_seq2seq

    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if train_df is None or holdout_df is None:
        full = load_split(cfg.train_path, has_answer=True)
        train_df, holdout_df = stratified_split(full, cfg.val_size, cfg.seed)

    # Cap the (expensive) generation-based eval to keep per-epoch ROUGE cheap.
    if cfg.eval_max_samples and len(holdout_df) > cfg.eval_max_samples:
        strat = holdout_df[LANG_COL] if LANG_COL in holdout_df.columns else None
        from sklearn.model_selection import train_test_split as _tts
        holdout_df, _ = _tts(
            holdout_df, train_size=cfg.eval_max_samples,
            random_state=cfg.seed, stratify=strat,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = load_seq2seq(cfg.model_name)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # incompatible with grad checkpointing

    hf_train = make_hf_dataset(train_df, tokenizer, cfg)
    hf_eval = make_hf_dataset(holdout_df, tokenizer, cfg)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        label_pad_token_id=-100, pad_to_multiple_of=8,
    )
    bf16, fp16 = _resolve_precision(cfg)

    # Create optimizer (Adafactor for memory efficiency on 16GB GPUs, AdamW otherwise)
    optimizer = _create_optimizer(cfg, model)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate if cfg.optimizer == "adamw" else None,  # Adafactor sets its own LR
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        label_smoothing_factor=cfg.label_smoothing_factor,
        bf16=bf16,
        fp16=fp16,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=2,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=cfg.max_target_length,
        generation_num_beams=cfg.num_beams,
        logging_steps=100,
        report_to="none",
        seed=cfg.seed,
    )

    callbacks = []
    if cfg.early_stopping_patience and cfg.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=hf_train,
        eval_dataset=hf_eval,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
        optimizers=(optimizer, None),  # (optimizer, lr_scheduler) - let Trainer handle scheduler
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)          # best model (load_best_model_at_end)
    tokenizer.save_pretrained(cfg.output_dir)
    return trainer, tokenizer
