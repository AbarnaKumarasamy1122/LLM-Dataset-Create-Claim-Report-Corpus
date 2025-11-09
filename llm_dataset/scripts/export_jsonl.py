import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ===== Paths =====
DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "enriched_metadata.csv")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
OUTPUT_DIR = os.path.join(DATA_DIR, "llm_jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Load Metadata =====
print("🔹 Loading enriched metadata...")
metadata = pd.read_csv(METADATA_PATH)
print(f"✅ Loaded {len(metadata)} records from {METADATA_PATH}")

# Normalize image_id column
if "image_id" not in metadata.columns:
    possible_cols = [c for c in metadata.columns if "image" in c.lower()]
    if possible_cols:
        metadata.rename(columns={possible_cols[0]: "image_id"}, inplace=True)
        print(f"ℹ️ Renamed column {possible_cols[0]} → image_id")
    else:
        raise ValueError("❌ No image_id column found in metadata.")

# ===== Load Claim Reports =====
print("🔹 Loading claim reports...")
reports = []
for root, _, files in os.walk(REPORTS_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        reports.append(json.loads(line.strip()))
                    except Exception:
                        pass
        elif file.endswith(".json"):
            try:
                data = json.load(open(file_path, "r", encoding="utf-8"))
                if isinstance(data, list):
                    reports.extend(data)
                else:
                    reports.append(data)
            except Exception:
                pass
        elif file.endswith(".txt"):
            image_id = os.path.splitext(file)[0]
            text = open(file_path, "r", encoding="utf-8").read().strip()
            reports.append({"image_id": image_id, "output": text})

print(f"✅ Loaded {len(reports)} claim reports from {REPORTS_DIR}")
if reports:
    print("🧩 Sample report keys:", list(reports[0].keys()))

# ===== Extract image_id from reports =====
def extract_image_id(report):
    """Try multiple strategies to extract image_id or filename from report."""
    # 1️⃣ Direct key search
    for key in ["image_id", "file_name", "filename", "image_path", "evidence"]:
        if key in report:
            val = str(report[key]).strip()
            if val:
                return os.path.basename(val)

    # 2️⃣ From input field text
    input_text = str(report.get("input", ""))
    # Match patterns like 'image_id=000_mask.png' or 'evidence=data/images/test/001.png'
    match = re.search(r'([A-Za-z0-9_\-]+\.png|[A-Za-z0-9_\-]+\.jpg)', input_text)
    if match:
        return match.group(1)

    # 3️⃣ If JSON-like embedded data exists
    if isinstance(report.get("input"), dict):
        for v in report["input"].values():
            if isinstance(v, str) and (".png" in v or ".jpg" in v):
                return os.path.basename(v)

    return None


# ===== Build lookup dictionary =====
report_dict = {}
for r in reports:
    image_id = extract_image_id(r)
    if not image_id:
        continue
    text = r.get("output") or r.get("claim_text") or ""
    if text.strip():
        report_dict[image_id] = text.strip()

print(f"✅ Built lookup dictionary for {len(report_dict)} reports")

# Debug preview
meta_ids = metadata["image_id"].astype(str).apply(os.path.basename).tolist()
print(f"💡 Debug: First 5 metadata IDs → {meta_ids[:5]}")
report_ids = list(report_dict.keys())
print(f"💡 Debug: First 5 report IDs → {report_ids[:5]}")

# ===== Merge Metadata & Reports =====
print("🔹 Merging metadata with claim reports...")
dataset_entries = []

for _, row in metadata.iterrows():
    image_id = os.path.basename(str(row["image_id"]).strip())
    claim_text = report_dict.get(image_id)
    if not claim_text:
        continue

    input_text = (
        f"shipment_id={row.get('shipment_id', '')}; "
        f"image_id={image_id}; "
        f"damage_type={row.get('damage_type', '')}; "
        f"severity={row.get('severity', '')}; "
        f"vendor={row.get('vendor', '')}; "
        f"shipment_stage={row.get('shipment_stage', '')}"
    )

    dataset_entries.append({"input": input_text, "output": claim_text})

print(f"✅ Created {len(dataset_entries)} combined entries (missing: {len(metadata) - len(dataset_entries)})")

if not dataset_entries:
    print("⚠️ No dataset entries found! Check if report text or input field contains filenames like `.png` or `.jpg`.")
    print("💡 Tip: Open one report file and confirm how images are referenced (e.g., `evidence=data/images/...`).")
    exit()

# ===== Split & Export =====
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
train_data, temp_data = train_test_split(dataset_entries, test_size=(1 - train_ratio), random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

def save_jsonl(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"💾 Saved {filename} ({len(data)} entries)")

save_jsonl(train_data, "train.jsonl")
save_jsonl(val_data, "val.jsonl")
save_jsonl(test_data, "test.jsonl")

print("🎯 Export complete. Ready for LLM fine-tuning.")
