# Sample data for ESG Audit Platform demo

These files are used by `scripts/seed_sample_data.py` to populate the platform for investor/client demos.

| File | Purpose |
|------|--------|
| `scope1_emissions_2025.csv` | Scope 1 GHG by source/activity; used for CSRD_2025_ClimateChange_Scope1 |
| `energy_consumption_2025.csv` | Electricity consumption by site; used for CSRD_2025_Energy_Consumption |
| `emission_factors_ref.csv` | Reference emission factors; used for CSRD_2025_EmissionFactors (Document Q&A / reference) |

**Load sample data (from project root):**
```bash
python scripts/seed_sample_data.py
```

Then start the app: `streamlit run app.py`

See **DEMO_SCRIPT.md** in the project root for the full presenter script.
