"""Module B: Evidence Vault — Tag-based upload linked to material topics (IROs) for audit traceability."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from datetime import datetime

import db
from evidence_store import save_uploaded_file
from suggestions import get_evidence_tag_suggestions

st.set_page_config(page_title="Evidence Vault", layout="wide")
st.title("2. Evidence Vault")
st.caption("Step 2 of 4 — Attach source documents to material topics.")
st.markdown("Upload **source documents** (policies, spreadsheets, reports) and assign a **tag**. Link files to a **material topic (IRO)** so the audit trail shows which disclosure each file supports. Tags are used in the Agentic Auditor to generate calculators.")

# Resolve current framework and IROs for dropdowns
frameworks = db.list_frameworks()
framework_id = frameworks[0]["id"] if frameworks else None
iros = db.list_iros(framework_id=framework_id) if framework_id else []
iro_options = [(f"{iro['topic']} — {iro['sub_topic']}" + (f" ({iro.get('disclosure_code')})" if iro.get("disclosure_code") else ""), iro["id"]) for iro in iros]
current_year = datetime.now().year

st.header("Upload with tag")
st.markdown("Link evidence to a **material topic (IRO)** so the audit trail shows which disclosure each file supports.")

# Tag suggestions from existing tags and IROs
iro_topics = [iro["topic"] for iro in iros]
tag_suggestions = get_evidence_tag_suggestions(
    existing_tags=db.list_all_evidence_tags(),
    iro_topics=iro_topics if iro_topics else None,
    year=str(current_year),
)
if tag_suggestions:
    st.caption("Suggested tags (click to use):")
    tcols = st.columns(min(3, len(tag_suggestions)))
    for i, tag in enumerate(tag_suggestions[:9]):
        with tcols[i % 3]:
            if st.button(tag[:40] + ("..." if len(tag) > 40 else ""), key=f"ev_tag_{i}", help=tag):
                st.session_state.evidence_tag_prefill = tag
                st.rerun()

# Optional: link to IRO — when selected, suggest tag from IRO
link_iro_id = None
if iro_options:
    iro_labels = ["— Do not link"] + [o[0] for o in iro_options]
    iro_selected = st.selectbox("Link to material topic (IRO)", options=iro_labels, key="iro_link")
    if iro_selected != "— Do not link":
        link_iro_id = next(o[1] for o in iro_options if o[0] == iro_selected)
        iro = next(i for i in iros if i["id"] == link_iro_id)
        suggested_tag = f"CSRD_{current_year}_{iro['topic'].replace(' ', '_')}_{iro['sub_topic'].replace(' ', '_')}"
        if not st.session_state.get("evidence_tag_prefill") and not st.session_state.get("evidence_tag_input"):
            if st.button("Use tag from IRO", key="use_iro_tag"):
                st.session_state.evidence_tag_prefill = suggested_tag
                st.rerun()
        st.caption(f"Suggested tag for this IRO: `{suggested_tag}`")

evidence_tag = st.text_input(
    "Evidence tag",
    value=st.session_state.pop("evidence_tag_prefill", "") or "",
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
                save_uploaded_file(f, tag, iro_id=link_iro_id)
            except Exception as e:
                st.error(f"Error saving {f.name}: {e}")
        st.success(f"Uploaded {len(uploaded_files)} file(s) with tag **{tag}**" + (" (linked to IRO)" if link_iro_id else "") + ".")
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
