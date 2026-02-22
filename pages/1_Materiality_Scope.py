"""Module A: Materiality & Scope — Reporting entity, framework, Double Materiality Assessment (DMA) with disclosure mapping."""

import sys
from pathlib import Path

# Ensure project root is on path when running as Streamlit page
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from datetime import datetime

import config
import db
from esg_standards import get_disclosures_for_framework, AUDIT_STATUSES
from suggestions import (
    FRAMEWORK_SUGGESTIONS,
    TOPIC_SUGGESTIONS,
    SUB_TOPIC_SUGGESTIONS,
    STAKEHOLDER_GROUPS_SUGGESTIONS,
)

st.set_page_config(page_title="Materiality & Scope", layout="wide")
st.title("1. Materiality & Scope")
st.caption("Step 1 of 4 — Define what you report and to which standards.")
st.markdown("Set your **reporting entity** and **period**, then conduct a **Double Materiality Assessment (DMA)** and map topics to **ESRS/GRI** disclosure codes. This is the foundation for evidence and metrics.")

# ---- 1. Reporting framework ----
st.header("1. Reporting framework")
frameworks = db.list_frameworks()
framework_options = {f["name"]: f["id"] for f in frameworks}
framework_id = None

if not frameworks:
    st.subheader("Create a framework")
    st.caption("Suggestions (click to fill framework name):")
    fw_cols = st.columns(min(3, len(FRAMEWORK_SUGGESTIONS)))
    for i, (name, desc) in enumerate(FRAMEWORK_SUGGESTIONS):
        with fw_cols[i % 3]:
            if st.button(f"{name}", key=f"fw_sugg_{i}", help=desc):
                st.session_state.framework_suggestion = (name, desc)
                st.rerun()
    with st.form("new_framework"):
        fw_name = st.text_input(
            "Framework name",
            value=st.session_state.pop("framework_suggestion", (None, None))[0] or "",
            placeholder="e.g. CSRD, GRI, TCFD, custom",
        )
        fw_desc = st.text_area("Description", placeholder="Optional")
        if st.form_submit_button("Create"):
            if fw_name and fw_name.strip():
                db.create_framework(fw_name.strip(), fw_desc or "")
                st.success(f"Created framework: {fw_name}")
                st.rerun()
            else:
                st.warning("Enter a framework name.")
else:
    selected_fw_name = st.selectbox(
        "Select reporting framework",
        options=list(framework_options.keys()),
        index=0,
        key="sel_framework",
    )
    framework_id = framework_options[selected_fw_name]

    with st.expander("Create another framework"):
        with st.form("add_framework"):
            fw_name = st.text_input("Framework name", key="fw_name")
            fw_desc = st.text_area("Description", key="fw_desc")
            if st.form_submit_button("Add framework"):
                if fw_name and fw_name.strip():
                    db.create_framework(fw_name.strip(), fw_desc or "")
                    st.success(f"Created: {fw_name}")
                    st.rerun()

# ---- 2. Reporting entity (audit period) ----
if framework_id is not None:
    st.header("2. Reporting entity & period")
    entities = db.list_reporting_entities(framework_id=framework_id)
    current_year = datetime.now().year

    if not entities:
        with st.form("new_entity"):
            st.subheader("Create reporting entity")
            entity_name = st.text_input("Entity name", placeholder="e.g. Acme Corp, Group Europe")
            reporting_year = st.number_input("Reporting year", min_value=2000, max_value=2030, value=current_year, step=1)
            status = st.selectbox(
                "Audit status",
                options=[s[0] for s in AUDIT_STATUSES],
                format_func=lambda x: next(d[1] for d in AUDIT_STATUSES if d[0] == x),
                index=0,
            )
            if st.form_submit_button("Create entity"):
                if entity_name and entity_name.strip():
                    db.create_reporting_entity(entity_name.strip(), int(reporting_year), framework_id, status)
                    st.success(f"Created: {entity_name} ({reporting_year})")
                    st.rerun()
                else:
                    st.warning("Enter entity name.")
    else:
        entity_options = {f"{e['entity_name']} ({e['reporting_year']})": e["id"] for e in entities}
        selected_entity_label = st.selectbox(
            "Select reporting entity / period",
            options=list(entity_options.keys()),
            index=0,
            key="sel_entity",
        )
        entity_id = entity_options[selected_entity_label]
        current_entity = db.get_reporting_entity(entity_id)
        if current_entity:
            st.caption(f"Framework: **{current_entity.get('framework_name', '')}**")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col2:
                new_status = st.selectbox(
                    "Audit status",
                    options=[s[0] for s in AUDIT_STATUSES],
                    format_func=lambda x: next(d[1] for d in AUDIT_STATUSES if d[0] == x),
                    index=next(i for i, s in enumerate(AUDIT_STATUSES) if s[0] == current_entity.get("status", "draft")),
                    key="entity_status",
                )
            with col3:
                if st.button("Update status"):
                    db.update_reporting_entity_status(entity_id, new_status)
                    st.success("Status updated.")
                    st.rerun()

    # ---- 3. Double Materiality Assessment (IROs) ----
    st.header("3. Double Materiality Assessment (IROs)")
    st.markdown("Document **Impacts, Risks, and Opportunities** and map each to a disclosure standard (e.g. ESRS E1, GRI 305).")

    disclosures = get_disclosures_for_framework(selected_fw_name if frameworks else "")
    disclosure_options = ["—"] + [f"{c} – {d}" for c, d in disclosures]

    st.subheader("Add IRO")
    st.caption("Topic (click a suggestion):")
    t_cols = st.columns(min(3, len(TOPIC_SUGGESTIONS)))
    for i, t in enumerate(TOPIC_SUGGESTIONS):
        with t_cols[i % 3]:
            if st.button(t, key=f"topic_sugg_{i}"):
                st.session_state.iro_topic_sugg = t
                st.rerun()
    st.caption("Sub-topic (click a suggestion):")
    st_cols = st.columns(min(3, len(SUB_TOPIC_SUGGESTIONS)))
    for i, s in enumerate(SUB_TOPIC_SUGGESTIONS):
        with st_cols[i % 3]:
            if st.button(s, key=f"sub_sugg_{i}"):
                st.session_state.iro_sub_sugg = s
                st.rerun()
    st.caption("Stakeholder groups (click to add):")
    sg_cols = st.columns(min(3, len(STAKEHOLDER_GROUPS_SUGGESTIONS)))
    for i, sg in enumerate(STAKEHOLDER_GROUPS_SUGGESTIONS):
        with sg_cols[i % 3]:
            if st.button(sg, key=f"sg_sugg_{i}"):
                prev = st.session_state.get("iro_stakeholder_prefill", "")
                st.session_state.iro_stakeholder_prefill = (prev + ", " + sg).strip(", ") if prev else sg
                st.rerun()

    with st.form("add_iro"):
        topic = st.text_input(
            "Topic",
            value=st.session_state.pop("iro_topic_sugg", "") or "",
            placeholder="e.g. Climate Change, Water, Workforce",
        )
        sub_topic = st.text_input(
            "Sub-topic",
            value=st.session_state.pop("iro_sub_sugg", "") or "",
            placeholder="e.g. Energy Consumption, GHG Emissions, Scope 1",
        )
        disclosure_choice = st.selectbox(
            "Disclosure standard (ESRS/GRI)",
            options=disclosure_options,
            key="disclosure_sel",
        )
        disclosure_code = disclosure_choice.split(" – ")[0] if disclosure_choice != "—" else ""
        materiality_scope = st.selectbox(
            "Materiality scope",
            options=["Financial", "Impact", "Both"],
            index=2,
            help="Impact = inside-out; Financial = outside-in; Both = double materiality",
        )
        stakeholder_groups = st.text_input(
            "Stakeholder groups",
            value=st.session_state.pop("iro_stakeholder_prefill", "") or "",
            placeholder="e.g. Employees, Investors, Communities",
        )
        description = st.text_area("Description / notes", placeholder="Optional")
        if st.form_submit_button("Add IRO"):
            if topic and sub_topic:
                db.create_iro(
                    framework_id=framework_id,
                    topic=topic.strip(),
                    sub_topic=sub_topic.strip(),
                    materiality_scope=materiality_scope,
                    stakeholder_groups=stakeholder_groups.strip() if stakeholder_groups else "",
                    description=description.strip() if description else "",
                    disclosure_code=disclosure_code or None,
                )
                st.success(f"Added IRO: {topic} — {sub_topic}" + (f" ({disclosure_code})" if disclosure_code else ""))
                st.rerun()
            else:
                st.warning("Topic and Sub-topic are required.")

    # List IROs for selected framework
    st.subheader("Recorded IROs")
    iros = db.list_iros(framework_id=framework_id)
    if not iros:
        st.info("No IROs yet. Add one above.")
    else:
        for iro in iros:
            label = f"{iro['topic']} — {iro['sub_topic']} ({iro['materiality_scope']})"
            if iro.get("disclosure_code"):
                label += f" · {iro['disclosure_code']}"
            with st.expander(label):
                st.write("**Stakeholder groups:**", iro["stakeholder_groups"] or "—")
                if iro.get("disclosure_code"):
                    st.write("**Disclosure:**", iro["disclosure_code"])
                if iro["description"]:
                    st.write("**Description:**", iro["description"])
                st.caption(f"ID: {iro['id']}")
