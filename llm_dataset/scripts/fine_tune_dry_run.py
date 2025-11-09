"""
fine_tune_dry_run.py
--------------------
Dry-run test for verifying dataset structure and formatting before LLM fine-tuning.
This script checks the JSONL datasets (train/val/test) and prints sample entries.
"""

import os
import json
from datasets import load_dataset

# ===== Paths =====
DATA_DIR = "data/llm_jsonl"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "val.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")

# ====== Validate dataset existence ======
print("🔍 Checking dataset files...")
for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file: {path}")
    else:
        print(f"✅ Found: {path}")

# ====== Load dataset using Hugging Face Datasets ======
print("\n📦 Loading datasets...")
dataset = load_dataset("json", data_files={
    "train": TRAIN_PATH,
    "validation": VAL_PATH,
    "test": TEST_PATH
})

# ====== Quick sanity checks ======
print("\n🔎 Dataset summary:")
for split in dataset.keys():
    print(f" - {split}: {len(dataset[split])} samples")

# Check sample format
sample = dataset["train"][0]
print("\n🧩 Sample structure:")
for k, v in sample.items():
    print(f"{k}: {str(v)[:150]}")

# ====== Verify required fields ======
required_fields = ["input", "output"]
print("\n⚙️ Verifying required fields...")
missing_fields = [f for f in required_fields if f not in sample]
if missing_fields:
    raise ValueError(f"❌ Missing required fields in dataset: {missing_fields}")
else:
    print("✅ All required fields found: ", required_fields)

# ====== Optional: preview few samples ======
print("\n📋 Preview few samples:")
for i, row in enumerate(dataset["train"].select(range(3))):
    print(f"\n🖼️ Sample {i+1}")
    print("Input:", row["input"])
    print("Output:", row["output"][:300], "..." if len(row["output"]) > 300 else "")

# ====== Dry-run passed ======
print("\n🎯 Dry run successful — dataset is ready for fine-tuning!")
