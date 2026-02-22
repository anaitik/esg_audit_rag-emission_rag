"""
Seed the ESG Audit Platform with sample data for investor/client demos.
Run from project root: python scripts/seed_sample_data.py

Creates:
- Framework: CSRD
- Reporting entity: GreenTech Industries (2025), status In progress
- Material topics (IROs): Climate/Scope 1, Energy/Consumption, Water
- Sample evidence files (CSVs) in Evidence Vault
- One calculator module and one reported metric
"""

import shutil
import sys
from pathlib import Path

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import db

SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
EVIDENCE_STORE = config.EVIDENCE_STORE_DIR


def ensure_sample_data():
    if not SAMPLE_DATA_DIR.exists():
        raise FileNotFoundError(f"Sample data folder not found: {SAMPLE_DATA_DIR}")
    csvs = list(SAMPLE_DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {SAMPLE_DATA_DIR}")
    return csvs


def seed_framework():
    """Create CSRD framework if not exists."""
    frameworks = db.list_frameworks()
    for f in frameworks:
        if f["name"] == "CSRD":
            return f["id"]
    fid = db.create_framework(
        "CSRD",
        "EU Corporate Sustainability Reporting Directive (ESRS)."
    )
    print(f"  Created framework: CSRD (id={fid})")
    return fid


def seed_entity(framework_id: int):
    """Create GreenTech Industries 2025."""
    entities = db.list_reporting_entities(framework_id=framework_id)
    for e in entities:
        if e["entity_name"] == "GreenTech Industries" and e["reporting_year"] == 2025:
            return e["id"]
    eid = db.create_reporting_entity(
        "GreenTech Industries",
        2025,
        framework_id,
        status="in_progress"
    )
    print(f"  Created entity: GreenTech Industries (2025) (id={eid})")
    return eid


def seed_iros(framework_id: int):
    """Create material topics (IROs) with disclosure codes. Returns dict iro_label -> id."""
    existing = {f"{r['topic']}|{r['sub_topic']}": r["id"] for r in db.list_iros(framework_id=framework_id)}
    iros_to_create = [
        ("Climate Change", "Scope 1 emissions", "Both", "Investors, Regulators, Communities", "GHG Scope 1", "ESRS E1"),
        ("Energy", "Energy consumption", "Both", "Investors, Regulators", "Electricity and fuel consumption", "ESRS E1"),
        ("Water", "Water withdrawal", "Impact", "Communities, Regulators", "Water use", "ESRS E3"),
    ]
    result = {}
    for topic, sub_topic, scope, stakeholders, desc, disclosure in iros_to_create:
        key = f"{topic}|{sub_topic}"
        if key in existing:
            result[key] = existing[key]
            continue
        iid = db.create_iro(
            framework_id=framework_id,
            topic=topic,
            sub_topic=sub_topic,
            materiality_scope=scope,
            stakeholder_groups=stakeholders,
            description=desc,
            disclosure_code=disclosure,
        )
        result[key] = iid
        print(f"  Created IRO: {topic} — {sub_topic} ({disclosure})")
    return result


def seed_evidence(iro_by_key: dict):
    """Copy sample_data CSVs into evidence_store and register with tags + IRO link."""
    import uuid
    EVIDENCE_STORE.mkdir(parents=True, exist_ok=True)
    # Map file -> (tag, iro_key)
    mapping = [
        ("scope1_emissions_2025.csv", "CSRD_2025_ClimateChange_Scope1", "Climate Change|Scope 1 emissions"),
        ("energy_consumption_2025.csv", "CSRD_2025_Energy_Consumption", "Energy|Energy consumption"),
        ("emission_factors_ref.csv", "CSRD_2025_EmissionFactors", None),  # reference, no IRO
    ]
    for filename, tag, iro_key in mapping:
        src = SAMPLE_DATA_DIR / filename
        if not src.exists():
            print(f"  Skip (file not found): {filename}")
            continue
        dest_dir = EVIDENCE_STORE / tag
        dest_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        dest_file = dest_dir / unique_name
        shutil.copy2(src, dest_file)
        iro_id = iro_by_key.get(iro_key) if iro_key else None
        db.add_evidence_file(
            evidence_tag=tag,
            file_path=str(dest_file),
            original_name=filename,
            iro_id=iro_id,
        )
        print(f"  Evidence: {filename} → tag {tag}" + (f" (IRO: {iro_key})" if iro_key else ""))


def seed_calculator_and_report(entity_id: int):
    """Add one calculator module and one report value for Scope 1 (idempotent)."""
    tag = "CSRD_2025_ClimateChange_Scope1"
    existing = db.list_report_values(reporting_entity_id=entity_id)
    if any(r["evidence_tag"] == tag for r in existing):
        print(f"  Calculator + report value already exist for {tag}; skip.")
        return
    # Simple calculator: sum of Emissions (tCO2e) column
    code = '''def calculate_metric(df):
    """Total Scope 1 GHG emissions in tCO2e (from Emissions column)."""
    if "Emissions (tCO2e)" not in df.columns:
        return None
    total = df["Emissions (tCO2e)"].replace("", float("nan")).astype(float).sum()
    return round(total, 2)
'''
    db.save_calculator_module(tag, code, "Total Scope 1 GHG emissions (tCO2e)")
    mod = db.get_calculator_module(tag)
    evidence_rows = db.list_evidence_by_tag(tag)
    evidence_ids = [r["id"] for r in evidence_rows]
    db.save_report_value(
        evidence_tag=tag,
        metric_name="Total Scope 1 GHG emissions (tCO2e)",
        calculated_value=149.58,
        unit="tCO2e",
        calculator_module_id=mod["id"],
        evidence_file_ids=evidence_ids,
        reporting_entity_id=entity_id,
    )
    print(f"  Calculator + report value: {tag} = 149.58 tCO2e (linked to entity {entity_id})")


def main():
    print("Seeding sample data for ESG Audit Platform demo...")
    ensure_sample_data()

    print("\n1. Framework & entity")
    fid = seed_framework()
    entity_id = seed_entity(fid)

    print("\n2. Material topics (IROs)")
    iro_by_key = seed_iros(fid)

    print("\n3. Evidence files")
    seed_evidence(iro_by_key)

    print("\n4. Calculator & reported metric")
    seed_calculator_and_report(entity_id)

    print("\nDone. Start the app: streamlit run app.py")
    print("  → Home shows Audit readiness 100% and workflow.")
    print("  → Report & Audit shows disclosure coverage and audit trail for GreenTech Industries 2025.")


if __name__ == "__main__":
    main()
