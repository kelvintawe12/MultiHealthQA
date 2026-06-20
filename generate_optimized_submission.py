"""Generate submission with optimized retrieval parameters."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mhqa.data import load_split
from mhqa.retrieval import PerLanguageRetriever
from mhqa.submit import make_submission


def generate_optimized_submission(k_neighbors=5):
    """Generate submission with optimized k_neighbors parameter."""
    print(f"Generating submission with k_neighbors={k_neighbors}...")
    
    # Load data
    print("Loading data...")
    train = load_split("data/Train.csv", has_answer=True)
    test = load_split("data/Test.csv", has_answer=False)
    
    # Train retriever with optimized k value
    print(f"Training retriever with k_neighbors={k_neighbors}...")
    retriever = PerLanguageRetriever(k_neighbors=k_neighbors).fit(train)
    
    # Generate predictions
    print("Generating predictions...")
    test_ids = test['ID'].tolist()
    preds, _, _ = retriever.predict(test)
    
    # Create submission
    output_path = f"retrieval_submission_k{k_neighbors}.csv"
    make_submission(
        ids=test_ids,
        predictions=preds,
        output_path=output_path,
        sample_submission_path="data/SampleSubmission.csv"
    )
    
    print(f"✅ Submission created: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="k_neighbors value")
    args = parser.parse_args()
    
    generate_optimized_submission(args.k)
