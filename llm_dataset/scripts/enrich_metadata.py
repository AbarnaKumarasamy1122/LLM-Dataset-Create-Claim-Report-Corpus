# enrich_metadata.py (final, warning-free)
import pandas as pd, ast, os, random, warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
SEED = 42
random.seed(SEED)

# consistent paths with synth_metadata.py
RULES = "rules/enrichment_rules.csv"
META_IN = "data/metadata.csv"
META_OUT = "data/enriched_metadata.csv"

# configuration
SEVERITY_FACTOR = {"low": 0.8, "medium": 1.0, "high": 1.6, "none": 0.0}
ACTION_MAP = {
    "low": "Minor repair / accept",
    "medium": "Repair or partial refund",
    "high": "Reject and replace",
    "none": "No action required",
}


def bbox_area_pct(bbox, img_w, img_h):
    """Compute % of image area covered by bbox"""
    try:
        if isinstance(bbox, str):
            bbox = ast.literal_eval(bbox)
        bbox = list(map(float, bbox))
    except Exception:
        return 0.0

    if len(bbox) == 4:
        x, y, w, h = bbox
        area = abs(w * h)
    elif len(bbox) == 2:
        area = abs(bbox[0] * bbox[1])
    else:
        try:
            x1, y1, x2, y2 = bbox
            area = abs((x2 - x1) * (y2 - y1))
        except Exception:
            return 0.0

    img_area = img_w * img_h if img_w and img_h else 1
    return round(100.0 * (area / img_area), 2)


def find_rule(df, dmg, stage):
    """Lookup rule based on damage_type and shipment_stage"""
    row = df[
        (df["damage_type"] == dmg)
        & ((df["shipment_stage"] == stage) | (df["shipment_stage"] == "*"))
    ]
    if not row.empty:
        return row.iloc[0]
    row = df[df["damage_type"] == dmg]
    if not row.empty:
        return row.iloc[0]
    return None


def main():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(META_IN):
        raise FileNotFoundError(f"{META_IN} not found — run synth_metadata.py first")

    if not os.path.exists(RULES):
        raise FileNotFoundError(f"{RULES} not found — please create enrichment_rules.csv")

    meta = pd.read_csv(META_IN)
    rules = pd.read_csv(RULES)

    # initialize new columns
    meta["likely_cause"] = ""
    meta["liability"] = ""
    meta["base_cost"] = 0.0
    meta["damage_area_pct"] = 0.0
    meta["estimated_cost"] = 0.0
    meta["action"] = ""
    meta["report_status"] = "draft"

    enriched_rows = []
    for idx, row in meta.iterrows():
        dmg = str(row.get("damage_type", "none")).lower()
        stage = str(row.get("shipment_stage", "*")).lower()

        rule = find_rule(rules, dmg, stage)
        if rule is not None:
            likely_cause = rule.get("likely_cause", "")
            liability = rule.get("liability", "")
            base_cost = float(rule.get("base_cost", 0))
        else:
            likely_cause = "Unknown - manual review"
            liability = "Unknown"
            base_cost = 400.0

        # compute damage area %
        img_w = int(row.get("image_width", 0)) if pd.notna(row.get("image_width")) else 0
        img_h = int(row.get("image_height", 0)) if pd.notna(row.get("image_height")) else 0
        area_pct = bbox_area_pct(row.get("bbox"), img_w, img_h)

        sev = str(row.get("severity", "medium")).lower()
        sf = SEVERITY_FACTOR.get(sev, 1.0)
        est = round(base_cost * (1 + area_pct / 100.0) * sf, 2)
        action = ACTION_MAP.get(sev, "Review")

        enriched_rows.append({
            **row.to_dict(),
            "likely_cause": likely_cause,
            "liability": liability,
            "base_cost": base_cost,
            "damage_area_pct": area_pct,
            "estimated_cost": est,
            "action": action,
            "report_status": "draft",
        })

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv(META_OUT, index=False)
    print(f"✅ Wrote enriched metadata to {META_OUT} — rows={len(enriched_df)}")


if __name__ == "__main__":
    main()
