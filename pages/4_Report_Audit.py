"""Report & Audit Trail — View calculated metrics and full audit chain (evidence → code → value)."""

import streamlit as st

import db

st.set_page_config(page_title="Report & Audit", layout="wide")
st.title("Report & Audit Trail")
st.markdown("Audit-ready view: for each metric see the calculated value, the evidence files used, and the calculator code.")

report_values = db.list_report_values()
if not report_values:
    st.info("No reported values yet. Use **Agentic Auditor** to run a calculator on a tag and save to report.")
    st.stop()

for rv in report_values:
    with st.expander(f"**{rv['metric_name'] or rv['evidence_tag']}** — {rv['calculated_value']} {rv['unit'] or ''}".strip()):
        st.write("**Evidence tag:**", rv["evidence_tag"])
        st.write("**Value:**", rv["calculated_value"], rv["unit"] or "")
        st.write("**Computed at:**", rv["computed_at"])

        # Evidence files (all files with this tag were used for this report value)
        evidence_rows = db.list_evidence_by_tag(rv["evidence_tag"])
        if evidence_rows:
            st.write("**Evidence files:**")
            for row in evidence_rows:
                st.write(f"- {row['original_name']}")

        # Calculator code (the module that was used for this report value)
        mid = rv.get("calculator_module_id")
        mod = db.get_calculator_module_by_id(mid) if mid else db.get_calculator_module(rv["evidence_tag"])
        if mod:
            st.write("**Calculator module (code used):**")
            st.code(mod["code_text"], language="python")
