"""Module A: Materiality & Scope Definition — Framework selection and Double Materiality Assessment (DMA)."""

import streamlit as st

import config
import db

st.set_page_config(page_title="Materiality & Scope", layout="wide")
st.title("Materiality & Scope Definition")
st.markdown("Define your reporting context: select a framework and conduct a Double Materiality Assessment (DMA) to identify Impacts, Risks, and Opportunities (IROs).")

# Framework selection
st.header("1. Reporting framework")
frameworks = db.list_frameworks()
framework_options = {f["name"]: f["id"] for f in frameworks}
framework_id = None

if not frameworks:
    with st.form("new_framework"):
        st.subheader("Create a framework")
        fw_name = st.text_input("Framework name", placeholder="e.g. CSRD, GRI, custom")
        fw_desc = st.text_area("Description", placeholder="Optional")
        if st.form_submit_button("Create"):
            if fw_name and fw_name.strip():
                db.create_framework(fw_name.strip(), fw_desc or "")
                st.success(f"Created framework: {fw_name}")
                st.rerun()
            else:
                st.warning("Enter a framework name.")
else:
    selected_name = st.selectbox(
        "Select reporting framework",
        options=list(framework_options.keys()),
        index=0,
    )
    framework_id = framework_options[selected_name]

    # Optional: create another framework
    with st.expander("Create another framework"):
        with st.form("add_framework"):
            fw_name = st.text_input("Framework name", key="fw_name")
            fw_desc = st.text_area("Description", key="fw_desc")
            if st.form_submit_button("Add framework"):
                if fw_name and fw_name.strip():
                    db.create_framework(fw_name.strip(), fw_desc or "")
                    st.success(f"Created: {fw_name}")
                    st.rerun()

# Double Materiality Assessment — IROs (only when a framework is selected)
if framework_id is not None:
    st.header("2. Double Materiality Assessment (IROs)")
    st.markdown("Document your **Impacts, Risks, and Opportunities** for each material topic.")

    with st.form("add_iro"):
        st.subheader("Add IRO")
        topic = st.text_input("Topic", placeholder="e.g. Climate Change, Water, Workforce")
        sub_topic = st.text_input("Sub-topic", placeholder="e.g. Energy Consumption, GHG Emissions, Scope 1")
        materiality_scope = st.selectbox(
            "Materiality scope",
            options=["Financial", "Impact", "Both"],
            index=2,
        )
        stakeholder_groups = st.text_input(
            "Stakeholder groups",
            placeholder="e.g. Employees, Investors, Communities (comma-separated)",
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
                )
                st.success(f"Added IRO: {topic} — {sub_topic}")
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
            with st.expander(f"{iro['topic']} — {iro['sub_topic']} ({iro['materiality_scope']})"):
                st.write("**Stakeholder groups:**", iro["stakeholder_groups"] or "—")
                if iro["description"]:
                    st.write("**Description:**", iro["description"])
                st.caption(f"ID: {iro['id']}")
