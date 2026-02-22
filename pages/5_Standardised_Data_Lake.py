"""Module E: Standardised Data Lake — normalize uploaded files to JSON and tag for Agentic Auditor."""

import sys
from pathlib import Path
import tempfile
import json
import uuid

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

import config
import db
from auditor_agent import load_file_to_dataframe


st.set_page_config(page_title="Standardised Data Lake", layout="wide")
st.title("5. Standardised Data Lake (beta)")
st.caption("Advanced — Normalize messy files to JSON, tag them, and reuse them in the Agentic Auditor.")
st.markdown(
    "Upload **any supported file** (CSV, Excel, PDF-as-text, TXT). "
    "We convert it to a **standard JSON structure** and save it under a **tag**. "
    "Those JSON files show up under the same tag in **Agentic Auditor**, where you can "
    "generate calculators and run metrics across all standardised files."
)

# Resolve current framework and IROs for dropdowns (optional link)
frameworks = db.list_frameworks()
framework_id = frameworks[0]["id"] if frameworks else None
iros = db.list_iros(framework_id=framework_id) if framework_id else []
iro_options = [
    (
        f"{iro['topic']} — {iro['sub_topic']}"
        + (f" ({iro.get('disclosure_code')})" if iro.get("disclosure_code") else ""),
        iro["id"],
    )
    for iro in iros
]

st.header("Standardise & tag files")

col_tag, col_iro = st.columns([2, 2])
with col_tag:
    std_tag = st.text_input(
        "Standardisation tag",
        placeholder="e.g. CSRD_2025_PurchasedGoods_JSON",
        key="std_tag_input",
    )
with col_iro:
    link_iro_id = None
    if iro_options:
        iro_labels = ["— Optional: do not link"] + [o[0] for o in iro_options]
        iro_selected = st.selectbox(
            "Link to material topic (IRO)",
            options=iro_labels,
            key="std_iro_select",
        )
        if iro_selected != "— Optional: do not link":
            link_iro_id = next(o[1] for o in iro_options if o[0] == iro_selected)

uploaded_files = st.file_uploader(
    "Upload files to standardise (CSV, Excel, PDF, TXT)",
    type=["csv", "xlsx", "xls", "pdf", "txt"],
    accept_multiple_files=True,
    key="std_upload",
)

if st.button("Standardise & save to JSON") and uploaded_files:
    tag = (std_tag or "").strip().replace(" ", "_")
    if not tag:
        st.error("Enter a standardisation tag first.")
    else:
        saved_count = 0
        previews = []
        for f in uploaded_files:
            # Write uploaded file to a temp path with correct suffix
            suffix = Path(f.name).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            df = load_file_to_dataframe(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            if df is None or df.empty:
                st.warning(f"Could not parse {f.name} into a table; skipped.")
                continue
            rows = df.to_dict(orient="records")
            data = {
                "tag": tag,
                "source_file": f.name,
                "rows": rows,
            }
            dest_dir = config.EVIDENCE_STORE_DIR / tag
            dest_dir.mkdir(parents=True, exist_ok=True)
            unique_name = f"{uuid.uuid4().hex}_{Path(f.name).stem}.json"
            dest_path = dest_dir / unique_name
            with open(dest_path, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=False)
            db.add_evidence_file(
                evidence_tag=tag,
                file_path=str(dest_path),
                original_name=f.name,
                iro_id=link_iro_id,
            )
            saved_count += 1
            # Keep a tiny preview (first 2 rows)
            previews.append((f.name, df.head(2)))

        if saved_count:
            st.success(
                f"Standardised and saved {saved_count} file(s) under tag **{tag}** "
                "as JSON. You can now go to **Agentic Auditor**, pick this tag, "
                "select a JSON file as the sample, and generate a metric calculator."
            )
            with st.expander("Preview of standardised data (first 2 rows per file)"):
                for name, df_preview in previews:
                    st.markdown(f"**{name}**")
                    st.dataframe(df_preview)
        else:
            st.info("No files were standardised. Check parsing warnings above.")

