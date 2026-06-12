"""End-to-end smoke test: prove the train -> evaluate -> predict -> submit path.

Uses google/mt5-small on a 200-row subset for 1 epoch on CPU. This is NOT a
quality run — it only verifies the wiring (tokenisation, label masking, Trainer,
ROUGE compute_metrics, generation, sentinel cleanup, submission validation).

    python -m scripts.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mhqa.config import load_config
from mhqa.data import ID_COL, load_split, stratified_split
from mhqa.evaluate import evaluate_model
from mhqa.infer import predict_dataframe
from mhqa.submit import make_submission
from mhqa.train import train


def main() -> None:
    cfg = load_config(
        "configs/mt5_base.yaml",
        model_name="google/mt5-small",
        num_train_epochs=1,
        max_input_length=64,
        max_target_length=96,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        eval_max_samples=40,
        num_beams=1,
        early_stopping_patience=0,
        output_dir="artifacts/smoke",
        submission_path="artifacts/smoke/submission.csv",
    )

    full = load_split(cfg.train_path, has_answer=True)
    tr, ho = stratified_split(full, cfg.val_size, cfg.seed)
    tr = tr.head(200).reset_index(drop=True)
    ho = ho.head(40).reset_index(drop=True)
    print(f"smoke: train={len(tr)} holdout={len(ho)}")

    trainer, tokenizer = train(cfg, train_df=tr, holdout_df=ho)

    overall, per_lang, _ = evaluate_model(trainer.model, tokenizer, ho, cfg, batch_size=4)
    print(f"smoke ROUGE-1={overall['rouge1_f1']:.4f} ROUGE-L={overall['rougeL_f1']:.4f}")

    test = load_split(cfg.test_path, has_answer=False).head(20).reset_index(drop=True)
    preds = predict_dataframe(trainer.model, tokenizer, test, cfg, batch_size=4)
    make_submission(test[ID_COL].tolist(), preds, cfg.submission_path)
    print(f"smoke: wrote {cfg.submission_path} with {len(preds)} rows")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
