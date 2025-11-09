import os
import json
import random
import pandas as pd
import yaml
from datetime import datetime

# --- Paths ---
DATA_DIR = "data"
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
ENRICHED_METADATA_PATH = os.path.join(DATA_DIR, "enriched_metadata.csv")
TEMPLATES_PATH = os.path.join(DATA_DIR, "templates.yaml")

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Load data ---
print("🔹 Loading enriched metadata...")
metadata_df = pd.read_csv(ENRICHED_METADATA_PATH)
print(f"✅ Loaded {len(metadata_df)} rows from enriched_metadata.csv")

print("🔹 Loading templates...")
with open(TEMPLATES_PATH, "r") as f:
    templates = yaml.safe_load(f)["templates"]
print(f"✅ Loaded {len(templates)} templates")

# --- Separate templates by type ---
damaged_templates = [t for t in templates if t["type"] == "damaged"]
no_damage_templates = [t for t in templates if t["type"] == "no_damage"]

# --- Helper function to fill a template ---
def fill_template(template_text, record):
    """Replace placeholders with actual metadata values."""
    try:
        return template_text.format(**record)
    except KeyError as e:
        print(f"⚠️ Missing key {e} in record {record.get('shipment_id', 'unknown')}")
        return template_text

# --- Generate claim reports ---
jsonl_data = []
for _, row in metadata_df.iterrows():
    record = row.to_dict()

    # Pick template based on damage condition
    if record.get("damage_type", "").lower() in ["none", "no_damage", "non_damaged"]:
        template = random.choice(no_damage_templates)
        record["confidence"] = round(random.uniform(0.90, 0.99), 2)
    else:
        template = random.choice(damaged_templates)

    report_text = fill_template(template["text"], record)

    # Create LLM fine-tuning entry
    json_entry = {
        "input": json.dumps(record, indent=2),
        "output": report_text
    }
    jsonl_data.append(json_entry)

# --- Save outputs ---
jsonl_path = os.path.join(REPORTS_DIR, "claim_reports.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Generated {len(jsonl_data)} claim reports → {jsonl_path}")

# Optional: Save 10% sample for manual review
sample_df = pd.DataFrame(random.sample(jsonl_data, max(1, len(jsonl_data)//10)))
sample_path = os.path.join(DATA_DIR, "review_log.csv")
sample_df.to_csv(sample_path, index=False)
print(f"✅ Saved sample review log → {sample_path}")
