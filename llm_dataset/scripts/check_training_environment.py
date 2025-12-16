"""
Pre-Training Checklist & Environment Verification
Verifies system is ready for fine-tuning before starting training.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("\n📌 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro}")
        print("   Required: Python 3.8 or higher")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
        return True
    except ImportError:
        print(f"   ❌ {package_name} (not installed)")
        return False

def check_required_packages():
    """Check all required packages."""
    print("\n📦 Checking required packages...")
    
    packages = [
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("torch", "torch"),
        ("evaluate", "evaluate"),
        ("rouge-score", "rouge_score"),
    ]
    
    all_installed = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_installed = False
    
    if not all_installed:
        print("\n   💡 Install missing packages with:")
        print("      pip install -r requirements.txt")
    
    return all_installed

def check_gpu():
    """Check GPU availability."""
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"      Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"      CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("   ⚠️  No GPU detected")
            print("      Training will use CPU (very slow, not recommended)")
            print("      💡 Consider using Google Colab with GPU runtime")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_datasets():
    """Check if cleaned datasets exist."""
    print("\n📊 Checking datasets...")
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "model_training"
    
    required_files = [
        "clean_train.jsonl",
        "clean_val.jsonl",
        "clean_test.jsonl",
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {filename} (not found)")
            all_exist = False
    
    if not all_exist:
        print("\n   💡 Run data preparation first:")
        print("      python scripts/prepare_data_for_finetuning.py")
    
    return all_exist

def check_disk_space():
    """Check available disk space."""
    print("\n💾 Checking disk space...")
    
    try:
        import shutil
        base_dir = Path(__file__).parent.parent
        stat = shutil.disk_usage(base_dir)
        
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        
        print(f"   Free: {free_gb:.2f} GB / {total_gb:.2f} GB")
        
        if free_gb >= 10:
            print(f"   ✅ Sufficient space (need ~5-10 GB for checkpoints)")
            return True
        else:
            print(f"   ⚠️  Low disk space (recommended: >10 GB)")
            return False
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return True

def check_model_access():
    """Check if can access Hugging Face model."""
    print("\n🤗 Checking Hugging Face model access...")
    
    try:
        from transformers import AutoTokenizer
        print("   Attempting to download tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        print("   ✅ Successfully accessed google/flan-t5-base")
        return True
    except Exception as e:
        print(f"   ❌ Failed to access model: {e}")
        print("   💡 Check internet connection or Hugging Face status")
        return False

def estimate_training_time():
    """Estimate training time based on hardware."""
    print("\n⏱️  Training time estimates...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if "A100" in gpu_name:
                time_est = "1-2 hours"
            elif "V100" in gpu_name:
                time_est = "2-3 hours"
            elif "T4" in gpu_name:
                time_est = "3-4 hours"
            elif "P100" in gpu_name:
                time_est = "3-5 hours"
            else:
                time_est = "2-6 hours"
            
            print(f"   GPU: {gpu_name} ({gpu_mem:.0f} GB)")
            print(f"   Estimated time: {time_est} for 3 epochs")
        else:
            print("   CPU training: 20-30+ hours (not recommended)")
    except:
        print("   Unable to estimate")

def print_next_steps():
    """Print next steps."""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Install missing packages (if any):")
    print("   pip install -r requirements.txt")
    print("\n2. Start training:")
    print("   python scripts/train_model.py")
    print("\n3. Monitor progress:")
    print("   - Check terminal output for training metrics")
    print("   - Watch GPU usage: nvidia-smi")
    print("   - Training logs in: model_training/logs/")
    print("\n4. After training:")
    print("   - Check test results: model_training/fine_tuned_model/test_results.json")
    print("   - Run evaluation: python scripts/evaluate_model.py")
    print("   - Test inference: python scripts/claim_report_generation.py")
    print("="*70 + "\n")

def main():
    """Run all checks."""
    print("\n" + "="*70)
    print("PRE-TRAINING ENVIRONMENT CHECK")
    print("="*70)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Required Packages", check_required_packages()),
        ("GPU Availability", check_gpu()),
        ("Datasets", check_datasets()),
        ("Disk Space", check_disk_space()),
        ("Model Access", check_model_access()),
    ]
    
    # Estimate training time
    estimate_training_time()
    
    # Summary
    print("\n" + "="*70)
    print("CHECK SUMMARY")
    print("="*70)
    
    all_passed = True
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
        if not result and check_name not in ["GPU Availability", "Disk Space"]:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All critical checks passed! Ready to start training.")
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Some checks failed. Please resolve issues before training.")
        print_next_steps()
        return 1

if __name__ == "__main__":
    sys.exit(main())
