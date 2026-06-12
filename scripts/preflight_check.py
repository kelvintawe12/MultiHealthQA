#!/usr/bin/env python3
"""Hardware preflight check before training.

Run this before starting a real training run to verify:
1. GPU availability and memory
2. Whether mt5-large will fit with current configuration
3. Data file integrity
4. Basic import sanity check

Usage:
    python -m scripts.preflight_check
    python -m scripts.preflight_check --config configs/mt5_large.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def check_gpu():
    """Check GPU availability and memory."""
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch")
        return False, None, None

    if not torch.cuda.is_available():
        print("❌ No GPU detected. Training mt5-large on CPU is not feasible.")
        print("   Expected training time: ~2 days for mt5-large on CPU")
        print("   Please use a GPU (Colab T4/V100/A100 or local 16GB+ GPU)")
        return False, None, None

    device_name = torch.cuda.get_device_name(0)
    device_cap = torch.cuda.get_device_capability(0)
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_supported = torch.cuda.is_bf16_supported()

    print(f"✅ GPU detected: {device_name}")
    print(f"   Memory: {total_memory_gb:.1f} GB")
    print(f"   Capability: {device_cap}")
    print(f"   bf16 support: {bf16_supported}")

    return True, total_memory_gb, bf16_supported


def check_config_memory_fit(gb_memory, config_path=None):
    """Check if the chosen config fits in available GPU memory."""
    if config_path is None:
        print("ℹ️  No config specified, assuming mt5-base with Adafactor")
        # mt5-base needs ~8GB with Adafactor
        return gb_memory >= 8

    try:
        from mhqa.config import load_config
        cfg = load_config(config_path)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

    # Estimate memory needs based on model and optimizer
    model_name = cfg.model_name
    optimizer = cfg.optimizer

    # Rough memory estimates (bf16, gradient checkpointing)
    if "large" in model_name.lower():
        model_mem = 2.3  # ~2.3GB for mt5-large weights in bf16
        gradient_mem = 2.3  # ~2.3GB for gradients
        activation_mem = 5.0  # ~5GB for activations with checkpointing
        
        if optimizer == "adafactor":
            optimizer_mem = 0.8  # Adafactor uses minimal optimizer state
        else:  # adamw
            optimizer_mem = 19.0  # AdamW uses ~19GB for mt5-large
            
        total_needed = model_mem + gradient_mem + activation_mem + optimizer_mem
    elif "base" in model_name.lower():
        model_mem = 1.0  # ~1GB for mt5-base weights in bf16
        gradient_mem = 1.0
        activation_mem = 3.0
        
        if optimizer == "adafactor":
            optimizer_mem = 0.3
        else:
            optimizer_mem = 8.0
            
        total_needed = model_mem + gradient_mem + activation_mem + optimizer_mem
    else:  # small
        model_mem = 0.3
        gradient_mem = 0.3
        activation_mem = 2.0
        
        if optimizer == "adafactor":
            optimizer_mem = 0.1
        else:
            optimizer_mem = 2.0
            
        total_needed = model_mem + gradient_mem + activation_mem + optimizer_mem

    fits = gb_memory >= total_needed

    print(f"\n📊 Memory fit analysis for {config_path}:")
    print(f"   Model: {model_name}")
    print(f"   Optimizer: {optimizer}")
    print(f"   Estimated memory needed: {total_needed:.1f} GB")
    print(f"   Available: {gb_memory:.1f} GB")
    
    if fits:
        print(f"   ✅ Fits with {(gb_memory - total_needed):.1f} GB to spare")
    else:
        print(f"   ❌ Does NOT fit - short by {(total_needed - gb_memory):.1f} GB")
        if optimizer == "adamw" and "large" in model_name.lower():
            print(f"   💡 Try switching optimizer: adafactor (saves ~18GB)")
        if "large" in model_name.lower():
            print(f"   💡 Try smaller model: mt5-base (saves ~6GB)")

    return fits


def check_data_files(data_dir="data"):
    """Check that all required data files exist and are non-empty."""
    required_files = ["Train.csv", "Val.csv", "Test.csv", "SampleSubmission.csv"]
    all_ok = True

    print(f"\n📁 Checking data files in {data_dir}/:")
    for filename in required_files:
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            print(f"   ❌ {filename} - MISSING")
            all_ok = False
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename} - {size_mb:.1f} MB")
            if size_mb == 0:
                print(f"      ⚠️  File is empty!")
                all_ok = False

    return all_ok


def check_imports():
    """Check that all required packages can be imported."""
    required = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn"),
        ("rouge_score", "rouge-score"),
    ]

    print(f"\n📦 Checking package imports:")
    all_ok = True
    for module_name, display_name in required:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} - NOT INSTALLED")
            all_ok = False

    # Check mhqa package
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        import mhqa
        print(f"   ✅ mhqa package")
    except ImportError as e:
        print(f"   ❌ mhqa package - {e}")
        all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Hardware preflight check")
    parser.add_argument("--config", help="Path to config file to check")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    args = parser.parse_args()

    print("=" * 60)
    print("MultiHealthQA Hardware Preflight Check")
    print("=" * 60)

    # Check GPU
    has_gpu, memory_gb, bf16_supported = check_gpu()
    
    # Check config memory fit if GPU available
    if has_gpu and memory_gb:
        fits = check_config_memory_fit(memory_gb, args.config)
    else:
        fits = False

    # Check data files
    data_ok = check_data_files(args.data_dir)

    # Check imports
    imports_ok = check_imports()

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_checks = [
        ("GPU available", has_gpu),
        ("Config fits in memory", fits if has_gpu else False),
        ("Data files present", data_ok),
        ("Package imports", imports_ok),
    ]

    for check_name, check_ok in all_checks:
        status = "✅" if check_ok else "❌"
        print(f"{status} {check_name}")

    if all(ok for _, ok in all_checks):
        print("\n✅ All checks passed! Ready to train.")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())