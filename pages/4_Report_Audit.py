"""Report & Audit — Disclosure coverage and full audit trail (value → evidence → code)."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

import db
from esg_standards import AUDIT_STATUSES

st.set_page_config(page_title="Report & Audit", layout="wide")
st.title("4. Report & Audit")
st.caption("Step 4 of 4 — Assurance-ready disclosure coverage and audit trail.")
st.markdown("View **disclosure coverage** (which material topics have evidence and metrics) and the full **audit trail** for each reported metric: value → source evidence → calculation code.")

# ----- Reporting period filter -----
entities = db.list_reporting_entities()
entity_options = {f"{e['entity_name']} ({e['reporting_year']})": e["id"] for e in entities}
filter_entity_id = None
if entity_options:
    filter_label = st.selectbox(
        "Reporting period",
        options=["All periods"] + list(entity_options.keys()),
        key="report_entity_filter",
    )
    if filter_label != "All periods":
        filter_entity_id = entity_options[filter_label]
        ent = db.get_reporting_entity(filter_entity_id)
        if ent:
            status_label = next((s[1] for s in AUDIT_STATUSES if s[0] == ent.get("status")), ent.get("status"))
            st.caption(f"**Status:** {status_label} · **Framework:** {ent.get('framework_name', '')}")

# ----- Disclosure coverage (material topics × evidence) -----
st.markdown("### Disclosure coverage")
st.caption("Material topics (IROs) and whether they have evidence and reported metrics.")
frameworks = db.list_frameworks()
if frameworks:
    fid = frameworks[0]["id"]
    iros_with_evidence = db.list_iros_with_evidence_count(framework_id=fid)
    all_report_values = db.list_report_values(reporting_entity_id=filter_entity_id) if filter_entity_id is not None else db.list_report_values()
    tags_with_metrics = {rv["evidence_tag"] for rv in all_report_values}

    if iros_with_evidence:
        for iro in iros_with_evidence:
            tags_for_iro = db.get_evidence_tags_for_iro(iro["id"])
            has_metric = any(t in tags_with_metrics for t in tags_for_iro) if tags_for_iro else False
            disp = iro.get("disclosure_code") or "—"
            ev_cnt = iro.get("evidence_count") or 0
            st.markdown(
                f"**{iro['topic']}** — {iro['sub_topic']} · "
                f"Disclosure: {disp} · Evidence: **{ev_cnt}** file(s) · "
                f"Reported metric: **{'Yes' if has_metric else 'No'}**"
            )
    else:
        st.info("No material topics yet. Define them in **Materiality & Scope**.")
else:
    st.info("Add a framework and material topics in **Materiality & Scope**.")

st.divider()
st.markdown("### Audit trail: reported metrics")
st.caption("Each metric shows: value, source evidence files, and the calculation (code) used — for auditor review.")

report_values = db.list_report_values(reporting_entity_id=filter_entity_id) if filter_entity_id is not None else db.list_report_values()
if not report_values:
    st.info("No reported values yet. Use **Agentic Auditor** to run a calculator on a tag and save to report.")
    st.stop()

for rv in report_values:
    title = f"{rv['metric_name'] or rv['evidence_tag']} — **{rv['calculated_value']}** {rv['unit'] or ''}".strip()
    with st.expander(title):
        st.write("**Metric:**", rv["metric_name"] or rv["evidence_tag"])
        st.write("**Value:**", rv["calculated_value"], rv["unit"] or "")
        st.write("**Computed at:**", rv["computed_at"])
        if rv.get("reporting_entity_id"):
            ent = db.get_reporting_entity(rv["reporting_entity_id"])
            if ent:
                st.write("**Reporting period:**", f"{ent['entity_name']} ({ent['reporting_year']})")

        evidence_rows = db.list_evidence_by_tag(rv["evidence_tag"])
        if evidence_rows:
            st.write("**Source evidence (files):**")
            for row in evidence_rows:
                st.write(f"- {row['original_name']}")

        mid = rv.get("calculator_module_id")
        mod = db.get_calculator_module_by_id(mid) if mid else db.get_calculator_module(rv["evidence_tag"])
        if mod:
            st.write("**Calculation (code used):**")
            st.code(mod["code_text"], language="python")
