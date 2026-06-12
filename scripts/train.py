"""Fine-tune an mT5 model and report held-out ROUGE.

    python -m scripts.train --config configs/mt5_large.yaml
    python -m scripts.train --config configs/mt5_base.yaml --epochs 1   # quick run

Saves the best checkpoint to ``cfg.output_dir`` and prints overall + per-language
ROUGE on the stratified holdout.
"""
from __future__ import annotations

import argparse

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from mhqa.config import load_config
from mhqa.evaluate import evaluate_model, holdout_for
from mhqa.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune mT5 for Multilingual Health QA")
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=float, default=None, help="override num_train_epochs")
    ap.add_argument("--model", default=None, help="override model_name")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    args = ap.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.model is not None:
        overrides["model_name"] = args.model
    cfg = load_config(args.config, **overrides)

    print(f"▶ Training {cfg.model_name}  (run={cfg.run_name})")
    trainer, tokenizer = train(cfg)

    print("\n▶ Scoring held-out split...")
    holdout = holdout_for(cfg)
    overall, per_lang, _ = evaluate_model(
        trainer.model, tokenizer, holdout, cfg, batch_size=args.eval_batch_size
    )
    print(f"\n📊 Holdout ROUGE-1={overall['rouge1_f1']:.4f}  "
          f"ROUGE-L={overall['rougeL_f1']:.4f}  combined={overall['combined']:.4f}")
    if not per_lang.empty:
        print("\nPer-language (weakest first):")
        print(per_lang.round(4).to_string())


if __name__ == "__main__":
    main()
