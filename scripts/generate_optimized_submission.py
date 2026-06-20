"""Generate submission with optimized retrieval parameters."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mhqa.data import load_split
from mhqa.retrieval import PerLanguageRetriever
from mhqa.submit import make_submission


def generate_optimized_submission(ngram_range=(3, 5)):
    """Generate submission with optimized ngram_range parameter."""
    print(f"Generating submission with ngram_range={ngram_range}...")
    
    # Load data
    print("Loading data...")
    project_root = Path(__file__).parent.parent
    train = load_split(project_root / "data" / "Train.csv", has_answer=True)
    test = load_split(project_root / "data" / "Test.csv", has_answer=False)
    
    # Train retriever with optimized ngram range
    print(f"Training retriever with ngram_range={ngram_range}...")
    retriever = PerLanguageRetriever(ngram_range=ngram_range).fit(train)
    
    # Generate predictions
    print("Generating predictions...")
    test_ids = test['ID'].tolist()
    preds, _, _ = retriever.predict(test)
    
    # Create submission
    output_path = project_root / f"retrieval_submission_{ngram_range[0]}_{ngram_range[1]}.csv"
    make_submission(
        ids=test_ids,
        predictions=preds,
        output_path=output_path,
        sample_submission_path=project_root / "data" / "SampleSubmission.csv"
    )
    
    print(f"✅ Submission created: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram-min", type=int, default=3, help="ngram range minimum")
    parser.add_argument("--ngram-max", type=int, default=5, help="ngram range maximum")
    args = parser.parse_args()
    
    generate_optimized_submission((args.ngram_min, args.ngram_max))
