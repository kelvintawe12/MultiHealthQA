"""Optimize retrieval baseline parameters for better ROUGE scores."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from mhqa.data import load_split, LANG_COL, ANSWER_COL
from mhqa.retrieval import PerLanguageRetriever
from mhqa.metrics import compute_rouge, compute_rouge_by_language


def test_ngram_ranges():
    """Test different ngram_range values for retrieval optimization."""
    print("Loading data...")
    # Get project root directory
    project_root = Path(__file__).parent.parent
    train = load_split(project_root / "data" / "Train.csv", has_answer=True)
    val = load_split(project_root / "data" / "Val.csv", has_answer=True)
    
    # Test different ngram ranges (char n-grams work better across languages)
    ngram_ranges = [(2, 4), (3, 5), (3, 6), (4, 7), (2, 6)]
    results = []
    
    for ngram_range in ngram_ranges:
        print(f"\n{'='*50}")
        print(f"Testing ngram_range={ngram_range}")
        print(f"{'='*50}")
        
        # Train retriever with this ngram range
        retriever = PerLanguageRetriever(ngram_range=ngram_range).fit(train)
        
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
            'ngram_range': str(ngram_range),
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
    
    # Find best ngram range
    best_idx = results_df['combined'].idxmax()
    best = results_df.loc[best_idx]
    print(f"\n🏆 Best ngram_range: {best['ngram_range']} (combined: {best['combined']:.4f})")
    
    return results_df, best


if __name__ == "__main__":
    results_df, best = test_ngram_ranges()
    print(f"\n✅ Optimization complete!")
    print(f"Recommended ngram_range: {best['ngram_range']}")
