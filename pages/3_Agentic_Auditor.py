"""Module C: Agentic Auditor — Select tag, pick sample file, generate code, run on all files when approved."""

import os
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

import config
import db
from ingestion import get_existing_vector_store
from llm import get_llm
from auditor_agent import (
    get_sample_content_from_file,
    generate_calculator_code,
    make_get_emission_factor,
    run_calculator_on_file,
    run_calculator_on_tag,
    run_auditor_agent,
)
from evidence_store import get_file_paths_for_tag

st.set_page_config(page_title="Agentic Auditor", layout="wide")
st.title("3. Agentic Auditor")
st.caption("Step 3 of 4 — Turn evidence into reported metrics.")
st.markdown(
    "Select a **tag**, choose a **sample file**. The auditor agent will: understand **double materiality** (IROs), "
    "use the sample to **prepare emission factor queries**, **retrieve** factors from the vector store, then **generate** "
    "the calculator code. Run on all files for that tag and save to the report. Every value is traceable to evidence and code."
)

# LLM config in sidebar
with st.sidebar:
    st.header("LLM for code generation")
    llm_choice = st.selectbox(
        "Model",
        options=list(config.LLM_PROVIDERS.keys()),
        index=0,
        key="auditor_llm",
    )
    provider_config = config.LLM_PROVIDERS[llm_choice]
    if provider_config["provider"] == "openai":
        api_key = st.text_input("OpenAI API Key", type="password", key="ak_openai")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    elif provider_config["provider"] == "google":
        api_key = st.text_input("Google API Key", type="password", key="ak_google")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    elif provider_config["provider"] == "groq":
        api_key = st.text_input("Groq API Key", type="password", key="ak_groq")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    elif provider_config["provider"] == "ollama":
        st.text_input("Ollama URL", value="http://localhost:11434", key="ollama_url")
    st.divider()
    st.caption("Reporting period (for saving to report)")
    entities = db.list_reporting_entities()
    entity_options = ["— None"] + [f"{e['entity_name']} ({e['reporting_year']})" for e in entities]
    if len(entities) > 0:
        sel = st.selectbox("Link saved metric to period", options=entity_options, key="auditor_entity")
        if sel != "— None":
            st.session_state["_auditor_entity_id"] = entities[entity_options.index(sel) - 1]["id"]
        else:
            st.session_state["_auditor_entity_id"] = None
    else:
        st.session_state["_auditor_entity_id"] = None
    st.divider()
    st.caption("Emission factors (Scope 3)")
    use_factors_db = st.checkbox(
        "Use emission factors from Chroma DB",
        value=True,
        help="When enabled, generated calculators can call get_emission_factor(material) to fetch factors from the Document Q&A index (esg_factors). Use for Scope 3: weight × emission factor.",
        key="use_factors_db",
    )

# Load factors DB (esg_factors) so calculators can fetch emission factors (e.g. Scope 3)
factors_vectorstore = None
if st.session_state.get("use_factors_db", True):
    factors_vectorstore = get_existing_vector_store(
        list(config.EMBEDDING_MODELS.keys())[0],
        persist_directory=str(config.PERSIST_DIRECTORY),
        collection_name=config.COLLECTION_NAME,
    )
# Always provide a callable (returns None when factors DB not available) so generated code can call get_emission_factor(material)
get_emission_factor = make_get_emission_factor(factors_vectorstore)

all_tags = db.list_all_evidence_tags()

if not all_tags:
    st.info("No evidence tags yet. Upload documents in **Evidence Vault** and assign a tag first.")
    st.stop()

selected_tag = st.selectbox("Evidence tag", options=all_tags, key="auditor_tag")

# List files for this tag; user picks one as sample for code generation
files_for_tag = get_file_paths_for_tag(selected_tag)
if not files_for_tag:
    st.warning(f"No files found for tag **{selected_tag}**. Upload files with this tag in Evidence Vault.")
    st.stop()

# Build options: show original name, or "name [2]" etc. when duplicates so each file is selectable
count_orig = {}
for _path, orig in files_for_tag:
    count_orig[orig] = count_orig.get(orig, 0) + 1
file_options = []
path_by_option = {}
for i, (path, orig) in enumerate(files_for_tag):
    label = f"{orig} [{i+1}]" if count_orig.get(orig, 0) > 1 else orig
    file_options.append(label)
    path_by_option[label] = path
st.subheader("Sample file for code generation")
sample_file_display = st.selectbox(
    "Choose one file as sample (used to generate the calculator code)",
    options=file_options,
    key="sample_file_select",
    help="The selected file's structure and content will be used to generate the Python calculator.",
)
sample_file_path = path_by_option.get(sample_file_display) if sample_file_display else None

# Optional AI suggestion for metric description from sample file structure
metric_prefill = st.session_state.pop("metric_description_suggestion", None)
if sample_file_path and not metric_prefill:
    if st.button("Suggest metric description from file structure"):
        try:
            from suggestions import suggest_metric_description_llm
            from auditor_agent import get_sample_content_from_file
            import ast
            sample_content = get_sample_content_from_file(sample_file_path)
            col_list = []
            if sample_content:
                for line in sample_content.split("\n"):
                    if line.strip().startswith("Columns:"):
                        rest = line.split("Columns:", 1)[-1].strip()
                        try:
                            col_list = ast.literal_eval(rest)
                        except Exception:
                            pass
                        break
                if not isinstance(col_list, list):
                    col_list = []
                suggestion = suggest_metric_description_llm(
                    selected_tag,
                    col_list,
                    sample_content,
                    get_llm,
                    provider_config,
                )
                if suggestion:
                    st.session_state.metric_description_suggestion = suggestion
                    st.rerun()
                else:
                    st.info("Could not generate suggestion. Enter a description manually.")
            else:
                st.info("No content in sample file to suggest from.")
        except Exception as e:
            st.warning(f"Suggestion unavailable: {e}. Enter a description manually.")

metric_description = st.text_input(
    "Metric description (for the Reasoning Agent)",
    value=metric_prefill or "",
    placeholder="e.g. Total Scope 1 GHG emissions in tCO2e; or Total electricity consumption in kWh",
    key="metric_desc",
)

# Agentic flow: materiality → sample → query plan → retrieval → code gen
framework_id = db.get_framework_id_for_entity(st.session_state.get("_auditor_entity_id"))

# Auto-Generate Calculator (agentic pipeline)
if st.button("Auto-Generate Calculator (Agentic)"):
    if not sample_file_path:
        st.error("Select a sample file first.")
        st.stop()
    llm = get_llm(provider_config, temperature=0.2, streaming=False)
    with st.spinner("Running auditor agent: materiality → sample → query plan → retrieval → code gen..."):
        code_text, steps_log = run_auditor_agent(
            evidence_tag=selected_tag,
            sample_file_path=sample_file_path,
            metric_description=metric_description or f"Metric for {selected_tag}",
            llm=llm,
            framework_id=framework_id,
            factors_vectorstore=factors_vectorstore,
        )
    st.session_state["generated_code"] = code_text
    st.session_state["generated_tag"] = selected_tag
    st.session_state["auditor_steps_log"] = steps_log
    st.rerun()

# Show agent steps log when we have a fresh generation
if "auditor_steps_log" in st.session_state and st.session_state.get("generated_tag") == selected_tag:
    with st.expander("Agent steps (materiality → sample → query plan → retrieval → code)", expanded=True):
        for s in st.session_state["auditor_steps_log"]:
            st.markdown(f"**{s['step'].replace('_', ' ').title()}** — {s['summary']}")
            if s.get("detail"):
                st.text(s["detail"][:1200] + ("..." if len(s.get("detail", "")) > 1200 else ""))
            st.divider()

# Show generated or saved code and allow edit
code_to_run = None
if "generated_code" in st.session_state and st.session_state.get("generated_tag") == selected_tag:
    st.header("Generated calculator")
    code_to_run = st.text_area(
        "Python code (edit if needed)",
        value=st.session_state["generated_code"],
        height=280,
        key="code_editor",
    )
elif db.get_calculator_module(selected_tag):
    mod = db.get_calculator_module(selected_tag)
    st.header("Saved calculator for this tag")
    code_to_run = st.text_area(
        "Python code (edit to update)",
        value=mod["code_text"],
        height=280,
        key="code_editor_saved",
    )
    if st.button("Update saved module"):
        db.save_calculator_module(selected_tag, code_to_run, mod.get("metric_description") or "")
        st.success("Updated.")
        st.rerun()

if code_to_run:
    # Preview on selected sample file
    if sample_file_path:
        st.subheader("Preview on sample file")
        st.caption(f"File: {sample_file_display}")
        if st.button("Run on sample"):
            try:
                val, _ = run_calculator_on_file(
                    code_to_run, sample_file_path, get_emission_factor=get_emission_factor
                )
                st.success(f"Result: **{val}**")
            except Exception as e:
                st.error(str(e))

    # Accept → then run on all files of this tag and store
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Accept and save module"):
            db.save_calculator_module(selected_tag, code_to_run, metric_description or "")
            st.success("Calculator saved for this tag.")
            if "generated_code" in st.session_state:
                del st.session_state["generated_code"]
                del st.session_state["generated_tag"]
            st.rerun()
    with col2:
        if st.button("Run on all files of this tag and save to report"):
            try:
                aggregated, per_file = run_calculator_on_tag(
                    code_to_run, selected_tag, get_emission_factor=get_emission_factor
                )
                if aggregated is not None:
                    mod = db.get_calculator_module(selected_tag)
                    mid = mod["id"] if mod else None
                    evidence_ids = [r["id"] for r in db.list_evidence_by_tag(selected_tag)]
                    db.save_report_value(
                        evidence_tag=selected_tag,
                        metric_name=metric_description or selected_tag,
                        calculated_value=aggregated,
                        unit="",
                        calculator_module_id=mid,
                        evidence_file_ids=evidence_ids,
                        reporting_entity_id=st.session_state.get("_auditor_entity_id") or None,
                    )
                    st.success(f"Aggregated result saved: **{aggregated}** (run over {len(per_file)} file(s)).")
                else:
                    st.warning("No values computed from files.")
                with st.expander("Per-file results"):
                    for path, v in per_file:
                        st.write(f"- {path}: {v}")
            except Exception as e:
                st.error(str(e))
    with col3:
        if st.button("Reject (clear)"):
            if "generated_code" in st.session_state:
                del st.session_state["generated_code"]
                del st.session_state["generated_tag"]
            if "auditor_steps_log" in st.session_state:
                del st.session_state["auditor_steps_log"]
            st.rerun()

# List saved calculators
st.header("Saved calculators")
modules = db.list_calculator_modules()
if not modules:
    st.info("No saved calculators yet.")
else:
    for m in modules:
        st.write(f"- **{m['evidence_tag']}** — {m.get('metric_description') or '(no description)'}")
