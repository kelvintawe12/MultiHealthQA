# Multilingual Health Question Answering in Low-Resource African Languages

This project implements a multilingual question answering system for health-related questions in low-resource African languages (Akan, Luganda, Swahili, Amharic, and English) as part of the Zindi MSRH Challenge.

## Project Overview

The goal of this project is to build models capable of answering health questions in multiple African languages, with a focus on handling low-resource languages effectively. The system uses:

- **Base Model**: mT5 (multilingual T5) for sequence-to-sequence generation
- **Retrieval Augmentation**: TF-IDF character n-gram retrieval as fallback mechanism
- **Multilingual Processing**: Script-safe text handling for Latin, Ge'ez (Amharic), and diacritic scripts (Akan)
- **Evaluation**: ROUGE-1 F1, ROUGE-L F1 metrics matching competition standards

## Competition Details

- **Languages**: English, Akan, Luganda, Swahili, Amharic
- **Task**: Generate health answers in the appropriate language based on the question
- **Metrics**: ROUGE-1 F1, ROUGE-L F1, LLM-as-a-Judge
- **Current Leaderboard Score**: 0.499179 (ROUGE-L F1)

## Leaderboard Progression

The following table shows the progression of leaderboard scores throughout the development process:

| Submission ID | Score | Description | Timestamp |
|--------------|-------|-------------|-----------|
| G277DbTd | 0.401853 | Retrieval-augmented generation | ~2 hours ago |
| dSNVqjYn | 0.499179 | Fine-tuned mT5-base with retrieval fallback | ~3 hours ago |
| o2wUNrFy | 0.499179 | Optimized decoding parameters | ~10 hours ago |
| EdnvUnVW | 0.493546 | Test training run | ~13 hours ago |
| VeB7mAdy | 0.079416 | Initial CPU training baseline | 3 days ago |

The current best score of 0.499179 represents a significant improvement from the initial baseline of 0.079416, demonstrating the effectiveness of systematic experimentation and hyperparameter optimization.

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/kelvintawe12/MultiHealthQA.git
cd MultiHealthQA

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

1. 
2. Run the setup cells to clone the repository and install dependencies
3. Upload your data files to the `data/` directory
4. Execute the notebook cells sequentially

## Project Structure

```
MultiHealthQA/
├── configs/                 # Configuration files for different experiments
│   ├── mt5_base.yaml       # Base configuration (580M parameters)
│   ├── mt5_large.yaml      # Large model configuration (1.2B parameters)
│   └── mt5_base_cpu_test.yaml
├── data/                    # Data files (not included in repo)
│   ├── Train.csv           # Training data
│   ├── Val.csv             # Validation data
│   ├── Test.csv            # Test data
│   └── SampleSubmission.csv
├── notebooks/              # Jupyter notebooks for experimentation
│   ├── MultiHealthQA_end_to_end.ipynb
│   └── starter_notebook_reference.ipynb
├── reports/               # Analysis and experiment results
│   ├── experiments.csv    # Experiment tracking
│   ├── leaderboard_progression.csv
│   └── figures/           # EDA visualizations
├── scripts/               # Command-line utilities
│   ├── train.py          # Training script
│   ├── predict.py        # Inference and submission generation
│   ├── run_experiments.py # Experiment suite
│   ├── smoke_test.py     # End-to-end test
│   ├── run_eda.py        # Exploratory data analysis
│   ├── build_notebook.py # Notebook generation
│   ├── optimize_retrieval.py # Retrieval optimization
│   └── check_gpu.py      # GPU detection
├── src/                   # Core package
│   └── mhqa/
│       ├── __init__.py   # Language definitions
│       ├── config.py     # Configuration management
│       ├── data.py       # Data loading and preprocessing
│       ├── metrics.py    # ROUGE evaluation
│       ├── retrieval.py  # TF-IDF retrieval system
│       ├── train.py      # Training pipeline
│       ├── modeling.py   # Model loading utilities
│       ├── infer.py      # Inference pipeline
│       ├── evaluate.py   # Evaluation helpers
│       ├── submit.py     # Submission generation
│       └── experiments.py # Experiment registry
├── tests/                # Unit tests
│   └── test_pipeline.py
├── artifacts/            # Model checkpoints and outputs (gitignored)
├── requirements.txt       # Python dependencies
└── README.md            # This file
```

## Usage

### Training a Model

Train the base mT5 model:
```bash
python -m scripts.train --config configs/mt5_base.yaml
```

Train with overrides:
```bash
python -m scripts.train --config configs/mt5_base.yaml --epochs 1 --model google/mt5-small
```

### Generating Predictions

Generate submission from a trained checkpoint:
```bash
python -m scripts.predict --config configs/mt5_base.yaml
```

### Running Experiments

List available experiments:
```bash
python -m scripts.run_experiments --list
```

Run a single experiment:
```bash
python -m scripts.run_experiments --config configs/mt5_base.yaml --only exp04_mt5base_langprefix
```

Run the full experiment suite:
```bash
python -m scripts.run_experiments --config configs/mt5_base.yaml --all
```

### Running Tests

Run the test suite:
```bash
pytest -q tests/test_pipeline.py
```

Run smoke test (end-to-end verification):
```bash
python -m scripts.smoke_test
```

### Optimizing Retrieval Parameters

Test different n-gram ranges for retrieval optimization:
```bash
python -m scripts.optimize_retrieval
```

Generate submission with specific retrieval parameters:
```bash
python -m scripts.generate_optimized_submission --ngram-min 3 --ngram-max 5
```

## Configuration

The project uses YAML configuration files for reproducibility. Key configuration options:

- **Model**: `model_name` (e.g., `google/mt5-base`, `google/mt5-large`)
- **Training**: `learning_rate`, `num_train_epochs`, `batch_size`
- **Architecture**: `max_input_length`, `max_target_length`
- **Prompting**: `prompt_style` (lang_prefix, bare, instruction)
- **Retrieval**: `retrieval_augment`, `hybrid_fallback`
- **Hardware**: `bf16_if_supported`, `gradient_checkpointing`

See `configs/mt5_base.yaml` for detailed configuration options and rationales.

## Experiments

The project includes a systematic experiment suite with 12 documented experiments:

1. **Retrieval Baseline**: TF-IDF retrieval-only baseline
2. **Zero-shot mT5**: Pretrained model without fine-tuning
3. **Bare Prompt**: Basic question format
4. **Language Prefix**: Language-conditioned prompting (default)
5. **Instruction Prompt**: Verbose instruction format
6. **Model Scale**: mT5-large vs mT5-base
7. **Decoding Strategy**: Beam search variations
8. **Target Length**: Sequence length ablation
9. **Optimization**: Learning rate and label smoothing
10. **Retrieval Augmentation**: Soft exemplar prompting
11. **Hybrid Fallback**: Retrieval fallback for collapsed generations
12. **Oversampling**: Low-resource language up-weighting

Results are tracked in `reports/experiments.csv`.

## Evaluation

### Local Evaluation

The project uses script-safe ROUGE evaluation with whitespace tokenization:

```python
from mhqa.metrics import compute_rouge, compute_rouge_by_language

# Overall ROUGE
scores = compute_rouge(predictions, references)

# Per-language breakdown
per_lang = compute_rouge_by_language(predictions, references, subsets)
```

### Competition Metrics

The Zindi platform evaluates:
- **TargetRLF1**: ROUGE-L F1 (primary metric)
- **TargetR1F1**: ROUGE-1 F1
- **TargetLLM**: LLM-as-a-Judge score

## Key Design Decisions

### Multilingual Text Processing
- **No lowercasing**: Preserves linguistic signal for proper nouns and medical terminology
- **No ASCII stripping**: Maintains diacritics (Akan) and Ge'ez script (Amharic)
- **Whitespace-only normalization**: Script-safe cleaning that matches evaluation tokenization

### Architecture Choices
- **mT5 over monolingual models**: Better cross-lingual transfer and resource efficiency
- **Character n-grams for retrieval**: Language-agnostic similarity across scripts
- **Adafactor optimizer**: Memory-efficient for large models on 16GB GPUs
- **BF16 precision**: Better numerical stability than FP16 for seq2seq tasks

### Evaluation Strategy
- **ROUGE-driven model selection**: Selects best generator, not lowest loss
- **Per-language monitoring**: Tracks low-resource language performance separately
- **Stratified validation**: Preserves language distribution in holdout sets

## Troubleshooting

### GPU Issues
Check GPU availability:
```bash
python -m scripts.check_gpu
```

### Memory Issues
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing` (default)
- Use `optimizer: adafactor` for large models
- Consider `mt5-base` instead of `mt5-large`

### Data Issues
- Ensure all CSV files are in the `data/` directory
- Verify data files match the expected format
- Check for encoding issues (use UTF-8)

## Performance

### Actual Results Achieved
- **Retrieval-only baseline**: ROUGE-L ~0.38, ROUGE-1 ~0.44
- **Initial CPU training**: ROUGE-L 0.079, ROUGE-1 0.082 (very poor due to hardware constraints)
- **Fine-tuned mT5-base with retrieval**: ROUGE-L 0.499, ROUGE-1 0.550 (current best)
- **Zero-shot mT5**: ROUGE-L ~0.25, ROUGE-1 ~0.30

### Resource Requirements
- **mT5-base**: 12GB GPU VRAM minimum, 2-3 hours training
- **mT5-large**: 16GB GPU VRAM recommended, 4-6 hours training
- **CPU training**: Not recommended (very poor results as shown in initial submission)

## Contributing

This project is part of an academic course submission. For improvements or issues:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code for your research, please cite:

```bibtex
@software{multilingual_health_qa_2024,
  title={Multilingual Health Question Answering in Low-Resource African Languages},
  author={Kelvin Tawe},
  year={2024},
  url={https://github.com/kelvintawe12/MultiHealthQA}
}
```

## License

This project is part of an academic submission. Please contact the author for usage permissions.

## Acknowledgments

- **Zindi**: For hosting the competition and providing the dataset
- **HuggingFace**: For the transformers library and mT5 model
- **Google**: For the mT5 model architecture

## Contact

For questions about this project, please contact Kelvin Tawe.

---
