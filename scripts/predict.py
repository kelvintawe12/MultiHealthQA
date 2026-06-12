"""Load a fine-tuned checkpoint and write a validated test submission.

    python -m scripts.predict --config configs/mt5_large.yaml
    python -m scripts.predict --config configs/mt5_base.yaml \
        --checkpoint artifacts/mt5_base --out artifacts/mt5_base/submission.csv

If the config enables retrieval augmentation or the hybrid fallback, a
``PerLanguageRetriever`` is fitted on Train.csv and threaded through inference.
"""
from __future__ import annotations

import argparse

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from mhqa.config import load_config
from mhqa.data import ID_COL, load_split
from mhqa.infer import predict_dataframe
from mhqa.submit import make_submission


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a test submission")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="defaults to cfg.output_dir")
    ap.add_argument("--out", default=None, help="defaults to cfg.submission_path")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg.output_dir
    out_path = args.out or cfg.submission_path

    from transformers import AutoTokenizer

    from mhqa.modeling import load_seq2seq

    print(f"▶ Loading checkpoint {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = load_seq2seq(checkpoint)

    retriever = None
    if cfg.retrieval_augment or cfg.hybrid_fallback:
        from mhqa.retrieval import PerLanguageRetriever
        print("▶ Fitting retriever for augmentation/fallback...")
        retriever = PerLanguageRetriever().fit(load_split(cfg.train_path, has_answer=True))

    test = load_split(cfg.test_path, has_answer=False)
    print(f"▶ Generating {len(test)} answers...")
    preds = predict_dataframe(model, tokenizer, test, cfg, retriever=retriever, batch_size=args.batch_size)

    make_submission(
        ids=test[ID_COL].tolist(),
        predictions=preds,
        output_path=out_path,
        sample_submission_path=cfg.sample_submission_path,
    )
    print(f"✅ Wrote validated submission -> {out_path}")


if __name__ == "__main__":
    main()
