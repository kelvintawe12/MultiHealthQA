"""Fast, CPU-only unit tests for the non-GPU parts of the pipeline.

Run with:  pytest -q   (from the repo root)

These guard the contracts that a wrong submission or a silent prompt/format
regression would otherwise only reveal on the leaderboard.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mhqa.config import load_config  # noqa: E402
from mhqa.data import build_prompt, clean_text, subset_to_language_name  # noqa: E402
from mhqa.metrics import compute_rouge, compute_rouge_by_language  # noqa: E402
from mhqa.submit import SUBMISSION_COLUMNS, make_submission  # noqa: E402

DATA = ROOT / "data"


# ── data ──────────────────────────────────────────────────────────────────────
def test_clean_text_whitespace_only_and_script_safe():
    assert clean_text("  a\n\t b  ") == "a b"
    assert clean_text(None) == ""
    # Ge'ez characters must survive untouched.
    amh = "የጤና ጥያቄ"
    assert clean_text(amh) == amh


def test_subset_to_language_name():
    assert subset_to_language_name("Amh_Eth") == "Amharic"
    assert subset_to_language_name("Aka_Gha") == "Akan"
    assert subset_to_language_name("Eng_Uga") == "English"
    assert subset_to_language_name(None) == "English"  # safe default


@pytest.mark.parametrize(
    "style,expected_contains",
    [("bare", "What"), ("lang_prefix", "Swahili:"), ("instruction", "in Swahili")],
)
def test_build_prompt_styles(style, expected_contains):
    p = build_prompt("What is malaria?", "Swa_Ken", style=style)
    assert expected_contains in p


def test_build_prompt_retrieval_augment_appends_reference():
    p = build_prompt("Q?", "Eng_Uga", style="lang_prefix", retrieved="canonical answer")
    assert "Reference: canonical answer" in p


# ── metrics ───────────────────────────────────────────────────────────────────
def test_rouge_perfect_and_zero():
    perfect = compute_rouge(["a b c"], ["a b c"])
    assert perfect["rouge1_f1"] == pytest.approx(1.0)
    assert perfect["rougeL_f1"] == pytest.approx(1.0)
    zero = compute_rouge(["x y z"], ["a b c"])
    assert zero["rouge1_f1"] == pytest.approx(0.0)


def test_rouge_known_partial_value():
    # pred "a b" vs ref "a b c d": precision 2/2, recall 2/4 -> F1 = 2/3
    m = compute_rouge(["a b"], ["a b c d"])
    assert m["rouge1_f1"] == pytest.approx(2 / 3, abs=1e-6)


def test_rouge_by_language_shape():
    df = compute_rouge_by_language(
        ["a b", "x"], ["a b", "x y"], ["Eng_Uga", "Amh_Eth"]
    )
    assert set(df.index) == {"Eng_Uga", "Amh_Eth"}
    assert {"rouge1_f1", "rougeL_f1", "combined", "count"} <= set(df.columns)


# ── submission ────────────────────────────────────────────────────────────────
def test_make_submission_format(tmp_path):
    ids = ["ID_1", "ID_2"]
    preds = ["answer one", ""]  # empty must be backfilled, never blank
    out = tmp_path / "sub.csv"
    df = make_submission(ids, preds, out)
    assert list(df.columns) == SUBMISSION_COLUMNS
    # same answer in all three target columns
    assert (df["TargetRLF1"] == df["TargetR1F1"]).all()
    assert (df["TargetR1F1"] == df["TargetLLM"]).all()
    assert df.loc[1, "TargetLLM"] != ""  # backfilled to a non-blank cell (space)


def test_make_submission_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        make_submission(["a"], ["x", "y"], tmp_path / "s.csv")


@pytest.mark.skipif(not (DATA / "SampleSubmission.csv").exists(), reason="data not present")
def test_submission_matches_sample_ids(tmp_path):
    sample = pd.read_csv(DATA / "SampleSubmission.csv")
    df = make_submission(
        ids=sample["ID"].tolist(),
        predictions=["x"] * len(sample),
        output_path=tmp_path / "s.csv",
        sample_submission_path=DATA / "SampleSubmission.csv",
    )
    assert len(df) == len(sample) == 2618


# ── config ────────────────────────────────────────────────────────────────────
def test_configs_load_and_override():
    cfg = load_config(ROOT / "configs" / "mt5_base.yaml")
    assert cfg.model_name == "google/mt5-base"
    assert cfg.max_target_length == 512
    cfg2 = cfg.with_overrides(num_train_epochs=1)
    assert cfg2.num_train_epochs == 1 and cfg.num_train_epochs == 3  # immutable copy


def test_config_rejects_unknown_field():
    cfg = load_config(ROOT / "configs" / "mt5_base.yaml")
    with pytest.raises(ValueError):
        cfg.with_overrides(does_not_exist=1)
