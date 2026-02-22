"""Module B: Evidence Vault — Tag-based document upload (no vectorization)."""

import streamlit as st

import db
from evidence_store import save_uploaded_file

st.set_page_config(page_title="Evidence Vault", layout="wide")
st.title("Evidence Vault")
st.markdown("Upload documents and assign them to a **tag**. Files are stored in a folder per tag (no vectorization). Use tags in the Agentic Auditor to pick a sample file and run calculators.")

# Tag input and file upload
st.header("Upload with tag")
st.markdown("Use a consistent tag format, e.g. **CSRD_ClimateChange_Scope1_2025** or **CSRD_2025_ClimateChange_Energy_Consumption**. Files are saved under `evidence_store/<tag>/`.")
evidence_tag = st.text_input(
    "Evidence tag",
    placeholder="e.g. CSRD_2025_ClimateChange_Energy_Consumption",
    key="evidence_tag_input",
)
uploaded_files = st.file_uploader(
    "Select documents (PDF, Excel, CSV, TXT)",
    type=["pdf", "xlsx", "xls", "csv", "txt"],
    accept_multiple_files=True,
    key="evidence_upload",
)

if evidence_tag and evidence_tag.strip() and uploaded_files:
    tag = evidence_tag.strip().replace(" ", "_")
    if st.button("Upload"):
        for f in uploaded_files:
            try:
                save_uploaded_file(f, tag)
            except Exception as e:
                st.error(f"Error saving {f.name}: {e}")
        st.success(f"Uploaded {len(uploaded_files)} file(s) with tag **{tag}**.")
        st.rerun()

# List evidence by tag
st.header("Evidence by tag")
all_tags = db.list_all_evidence_tags()
if not all_tags:
    st.info("No evidence tags yet. Upload documents with a tag above.")
else:
    selected_tag = st.selectbox("Select tag to view files", options=all_tags, key="tag_select")
    if selected_tag:
        files = db.list_evidence_by_tag(selected_tag)
        st.write(f"**{len(files)} file(s)** with tag `{selected_tag}`:")
        for row in files:
            st.write(f"- {row['original_name']} _(uploaded {row['uploaded_at']})_")
