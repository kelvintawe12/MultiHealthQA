"""Validated submission writer.

The platform expects a CSV with columns ``ID,TargetRLF1,TargetR1F1,TargetLLM``
where **the same generated answer fills all three target columns** (the platform
computes ROUGE-L, ROUGE-1 and the LLM-judge from that single answer). This module
writes that file and validates it against ``SampleSubmission.csv`` so a
mis-shaped submission fails locally rather than on the leaderboard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

SUBMISSION_COLUMNS = ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]


def make_submission(
    ids: Sequence[str],
    predictions: Sequence[str],
    output_path: str | Path,
    *,
    sample_submission_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build, validate and save a submission DataFrame.

    Empty predictions are replaced with a single space so no cell is blank
    (a blank cell can be rejected by the grader and always scores 0 ROUGE).
    """
    if len(ids) != len(predictions):
        raise ValueError(f"ids ({len(ids)}) and predictions ({len(predictions)}) length mismatch")

    answers = [(str(p).strip() or " ") for p in predictions]
    df = pd.DataFrame(
        {
            "ID": list(ids),
            "TargetRLF1": answers,
            "TargetR1F1": answers,
            "TargetLLM": answers,
        }
    )

    if sample_submission_path is not None:
        _validate_against_sample(df, sample_submission_path)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _validate_against_sample(df: pd.DataFrame, sample_submission_path: str | Path) -> None:
    """Assert column names, row count and ID coverage match the sample exactly."""
    sample = pd.read_csv(sample_submission_path)

    if list(df.columns) != list(sample.columns):
        raise ValueError(
            f"Column mismatch.\n  expected: {list(sample.columns)}\n  got     : {list(df.columns)}"
        )
    if len(df) != len(sample):
        raise ValueError(f"Row count mismatch: expected {len(sample)}, got {len(df)}")

    missing = set(sample["ID"]) - set(df["ID"])
    extra = set(df["ID"]) - set(sample["ID"])
    if missing or extra:
        raise ValueError(
            f"ID mismatch — missing {len(missing)}, unexpected {len(extra)} "
            f"(e.g. missing={list(missing)[:3]}, extra={list(extra)[:3]})"
        )
