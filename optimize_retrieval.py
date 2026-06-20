"""Optimize retrieval baseline parameters for better ROUGE scores."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from mhqa.data import load_split, LANG_COL, ANSWER_COL
from mhqa.retrieval import PerLanguageRetriever
from mhqa.metrics import compute_rouge, compute_rouge_by_language


def test_k_neighbors():
    """Test different k_neighbors values for retrieval."""
    print("Loading data...")
    train = load_split("data/Train.csv", has_answer=True)
    val = load_split("data/Val.csv", has_answer=True)
    
    # Test different k values
    k_values = [1, 3, 5, 7, 10, 15]
    results = []
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Testing k_neighbors={k}")
        print(f"{'='*50}")
        
        # Train retriever with this k value
        retriever = PerLanguageRetriever(k_neighbors=k).fit(train)
        
        # Generate predictions on validation set
        preds, _, _ = retriever.predict(val)
        references = val[ANSWER_COL].tolist()
        languages = val[LANG_COL].tolist()
        
        # Compute overall score
        overall = compute_rouge(preds, references)
        combined = (overall['rouge1_f1'] + overall['rougeL_f1']) / 2
        
        print(f"Overall ROUGE-1: {overall['rouge1_f1']:.4f}")
        print(f"Overall ROUGE-L: {overall['rougeL_f1']:.4f}")
        print(f"Combined: {combined:.4f}")
        
        # Per-language breakdown
        per_lang = compute_rouge_by_language(preds, references, languages)
        print(f"\nPer-language scores:")
        print(per_lang.round(4))
        
        results.append({
            'k': k,
            'rouge1': overall['rouge1_f1'],
            'rougeL': overall['rougeL_f1'],
            'combined': combined
        })
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*50}")
    results_df = pd.DataFrame(results)
    print(results_df.round(4))
    
    # Find best k
    best = results_df.loc[results_df['combined'].idxmax()]
    print(f"\n🏆 Best k_neighbors: {best['k']} (combined: {best['combined']:.4f})")
    
    return results_df, best


if __name__ == "__main__":
    results_df, best = test_k_neighbors()
    print(f"\n✅ Optimization complete!")
    print(f"Recommended k_neighbors: {int(best['k'])}")
