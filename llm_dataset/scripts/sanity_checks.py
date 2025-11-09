import os
import json
import pandas as pd

# ---------------------------------------------
# 🗂 Path Setup (auto-detects JSONL folder)
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))

# Prefer llm_jsonl folder if it exists
JSONL_DIR = os.path.join(DATA_DIR, "llm_jsonl")
if not os.path.exists(JSONL_DIR):
    JSONL_DIR = DATA_DIR  # fallback if files saved directly in /data

ENRICHED_PATH = os.path.join(DATA_DIR, "enriched_metadata.csv")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
TRAIN_PATH = os.path.join(JSONL_DIR, "train.jsonl")
VAL_PATH = os.path.join(JSONL_DIR, "val.jsonl")
TEST_PATH = os.path.join(JSONL_DIR, "test.jsonl")

print("🔎 Running Sanity Checks for Week 4 Dataset...\n")

# ---------------------------------------------
# 🧠 Helper: Validate JSONL file structure
# ---------------------------------------------
def check_jsonl_file(path):
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return 0

    print(f"🔍 Checking {os.path.basename(path)} ...")
    valid, invalid = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "input" in obj and "output" in obj:
                    valid += 1
                else:
                    invalid += 1
            except json.JSONDecodeError:
                invalid += 1

    print(f"✅ Valid: {valid}, ⚠️ Invalid: {invalid}\n")
    return valid

# ---------------------------------------------
# 1️⃣ Check Enriched Metadata
# ---------------------------------------------
print(f"📋 Checking metadata file: {ENRICHED_PATH}")

if not os.path.exists(ENRICHED_PATH):
    print("❌ enriched_metadata.csv not found.")
    df = None
else:
    df = pd.read_csv(ENRICHED_PATH)
    print(f"✅ Loaded {len(df)} rows")

    required_cols = [
        "image_id", "shipment_id", "damage_type",
        "severity", "damage_area_pct", "likely_cause",
        "liability", "estimated_cost", "action"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print("✅ All required columns present")

    # Check for nulls
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print("⚠️ Null values found:")
        print(nulls)
        # Optional: auto-fill missing liability
        if "liability" in nulls.index:
            df["liability"] = df["liability"].fillna("Pending Assessment")
            df.to_csv(ENRICHED_PATH, index=False)
            print("🩹 Filled missing 'liability' with 'Pending Assessment'")
    else:
        print("✅ No null values")

    # Check for duplicate image_ids
    duplicate_ids = df[df["image_id"].duplicated(keep=False)]["image_id"].unique()
    if len(duplicate_ids) > 0:
        print(f"⚠️ Duplicate image_ids found: {len(duplicate_ids)}")
        dup_log = os.path.join(DATA_DIR, "duplicate_image_ids.csv")
        df[df["image_id"].isin(duplicate_ids)].to_csv(dup_log, index=False)
        print(f"📁 Duplicates logged at: {dup_log}")
    else:
        print("✅ No duplicate image_ids")

# ---------------------------------------------
# 2️⃣ Check Reports Folder
# ---------------------------------------------
print(f"\n🧾 Checking reports folder: {REPORTS_DIR}")

if not os.path.exists(REPORTS_DIR):
    print("❌ Reports folder missing!")
else:
    report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".txt") or f.endswith(".jsonl") or f.endswith(".json")]
    print(f"✅ Found {len(report_files)} report files")

    if len(report_files) > 0:
        empty_reports = [rf for rf in report_files if os.path.getsize(os.path.join(REPORTS_DIR, rf)) == 0]
        if len(empty_reports) > 0:
            print(f"⚠️ Empty reports found: {len(empty_reports)}")
        else:
            print("✅ No empty reports")
    else:
        print("ℹ️ No individual report files detected (may be combined in claim_reports.jsonl)")

# ---------------------------------------------
# 3️⃣ Check JSONL Datasets
# ---------------------------------------------
print("\n🔍 Checking JSONL files...")
train_valid = check_jsonl_file(TRAIN_PATH)
val_valid = check_jsonl_file(VAL_PATH)
test_valid = check_jsonl_file(TEST_PATH)

# ---------------------------------------------
# 📊 Summary
# ---------------------------------------------
print("\n📊 Sanity Check Summary:")
print(f"   Enriched metadata rows: {len(df) if df is not None else 'N/A'}")
print(f"   Valid train records: {train_valid}")
print(f"   Valid val records: {val_valid}")
print(f"   Valid test records: {test_valid}")

print("\n🎯 Sanity checks complete.")
