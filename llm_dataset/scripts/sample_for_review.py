import pandas as pd
import json
import random
import os

# === CONFIG ===
REPORTS_PATH = "data/reports/claim_reports.jsonl"
OUTPUT_SAMPLE_PATH = "data/reports/sample_for_review.jsonl"
OUTPUT_CSV_LOG = "data/review_log.csv"
SAMPLE_FRACTION = 0.1  # 10%

def load_jsonl(filepath):
    """Load JSONL file as list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, filepath):
    """Save list of dicts as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def safe_parse_json(data):
    """Parse JSON string safely if needed."""
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except Exception:
        return {"raw_input": data}

def main():
    # === Step 1: Load all reports ===
    if not os.path.exists(REPORTS_PATH):
        print(f"❌ Error: File not found → {REPORTS_PATH}")
        return
    reports = load_jsonl(REPORTS_PATH)
    total_reports = len(reports)
    print(f"✅ Loaded {total_reports} claim reports.")

    # === Step 2: Sample 10% of reports ===
    sample_size = max(1, int(total_reports * SAMPLE_FRACTION))
    sample_reports = random.sample(reports, sample_size)
    print(f"✅ Selected {sample_size} reports for manual review.")

    # === Step 3: Save sample to JSONL ===
    os.makedirs(os.path.dirname(OUTPUT_SAMPLE_PATH), exist_ok=True)
    save_jsonl(sample_reports, OUTPUT_SAMPLE_PATH)
    print(f"✅ Saved sample subset → {OUTPUT_SAMPLE_PATH}")

    # === Step 4: Create empty review log CSV ===
    review_data = []
    for i, r in enumerate(sample_reports):
        input_data = safe_parse_json(r.get("input", {}))
        pkg_id = input_data.get("package_id", f"PKG-{i+1}")
        review_data.append({
            "report_id": pkg_id,
            "status": "Pending",
            "comments": ""
        })

    review_df = pd.DataFrame(review_data)
    review_df.to_csv(OUTPUT_CSV_LOG, index=False)
    print(f"✅ Created review log → {OUTPUT_CSV_LOG}")

    print("\n🎯 Next Step: Open `sample_for_review.jsonl` and update `review_log.csv` after review.")

if __name__ == "__main__":
    main()
