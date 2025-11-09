# synth_metadata.py
import os, csv, random
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)

# Base image folder (will recursively include train/val/test)
IMAGES_DIR = "data/images"
OUT_CSV = "data/metadata.csv"

VENDORS = ["GlobalPack Ltd","PolyChem Co","FastShip Logistics","TransMover Inc","VendorA"]
STAGES = ["warehouse","transit","delivered","handling"]
DAMAGE_TYPES = ["dent","scratch","crushed","wet","torn_tape","none"]

def rand_timestamp():
    base = datetime(2025,10,1,8,0,0)
    delta = random.randint(0, 60*24*30)  # up to 30 days
    return (base + timedelta(minutes=delta)).isoformat()

def list_images(root_dir):
    """Recursively collect all image file paths."""
    all_imgs = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff')):
                all_imgs.append(os.path.join(root, f))
    return all_imgs

def main():
    imgs = list_images(IMAGES_DIR)
    if not imgs:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")
    
    rows = []
    for i, img_path in enumerate(sorted(imgs)):
        fn = os.path.basename(img_path)
        shipment_id = f"SHP-{100000 + i}"
        vendor = random.choice(VENDORS)
        stage = random.choice(STAGES)

        # Label damaged/non_damaged from folder name
        lower_path = img_path.lower()
        if "non_damaged" in lower_path:
            damage_type = "none"
            severity = "none"
            confidence = round(random.uniform(0.9, 0.99), 2)
        else:
            damage_type = random.choice([d for d in DAMAGE_TYPES if d != "none"])
            severity = random.choices(["low","medium","high"], weights=[0.5,0.35,0.15])[0]
            confidence = round(random.uniform(0.6, 0.98), 2)

        bbox = "[0,0,0,0]"
        img_w, img_h = 512, 512
        rows.append({
            "image_id": fn,
            "image_path": img_path.replace("\\", "/"),
            "shipment_id": shipment_id,
            "damage_type": damage_type,
            "severity": severity,
            "bbox": bbox,
            "confidence": confidence,
            "shipment_stage": stage,
            "vendor": vendor,
            "timestamp": rand_timestamp(),
            "image_width": img_w,
            "image_height": img_h
        })

    os.makedirs("../data", exist_ok=True)
    with open(OUT_CSV, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Wrote {len(rows)} metadata rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
