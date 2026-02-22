"""Main Streamlit application for ESG Factor RAG."""

import os
import streamlit as st
from typing import List
# `langchain.schema` may not be present in some langchain distributions; prefer
# the public `langchain.schema` but fall back to `langchain_core.documents` when
# necessary for compatibility with the project's virtualenv.
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document
# PromptTemplate may be under `langchain.prompts` or `langchain_core.prompts`.
try:
    from langchain.prompts import PromptTemplate
except Exception:
    from langchain_core.prompts import PromptTemplate

# RetrievalQA historically lived under `langchain.chains`, but newer/older
# installs may expose it under `langchain_classic.chains.retrieval_qa.base`.
try:
    from langchain.chains import RetrievalQA
except Exception:
    from langchain_classic.chains.retrieval_qa.base import RetrievalQA

import config
from ingestion import load_document_from_file, split_documents, create_vector_store
from retrieval import create_hybrid_retriever
from llm import get_llm
from logger import get_logger

log = get_logger()

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()

def process_uploaded_files(uploaded_files, embedding_choice):
    """Process new files: load, split, and update vector store."""
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        return

    all_docs = st.session_state.documents.copy()
    for f in new_files:
        try:
            docs = load_document_from_file(f)
            all_docs.extend(docs)
            st.session_state.processed_files.add(f.name)
        except Exception as e:
            st.error(f"Error processing {f.name}: {e}")

    # Split all documents (including previous) to ensure consistent chunks
    split_all = split_documents(all_docs)
    st.session_state.documents = split_all

    # Recreate vector store with all documents
    vectorstore, rebuilt = create_vector_store(
        st.session_state.documents,
        embedding_choice,
        persist_directory=config.PERSIST_DIRECTORY
    )
    st.session_state.vectorstore = vectorstore
    log.info("ingestion | processed_files=%s new_count=%s total_chunks=%s rebuilt=%s", list(st.session_state.processed_files), len(new_files), len(split_all), rebuilt)
    st.success(f"Index updated with {len(new_files)} new file(s). Total chunks: {len(split_all)}")

def main():
    st.set_page_config(page_title="ESG Audit Platform", layout="wide", initial_sidebar_state="expanded")

    # ----- Hero: ESG Audit & Assurance Platform -----
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
            <h1 style="margin: 0; font-size: 2rem;">ESG Audit & Assurance Platform</h1>
            <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 1.05rem;">
                Double materiality · Evidence-backed metrics · Audit-ready trail
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Audit readiness from DB -----
    try:
        import db as _db
        stats = _db.get_audit_readiness_stats()
    except Exception:
        stats = {
            "framework_count": 0, "entity_count": 0, "iro_count": 0,
            "evidence_count": 0, "report_value_count": 0,
            "readiness_pct": 0, "steps_completed": 0, "steps_total": 5,
        }

    st.markdown("### Audit readiness")
    r1, r2, r3 = st.columns([2, 1, 2])
    with r1:
        pct = stats["readiness_pct"]
        st.progress(pct / 100.0)
        st.caption(f"**{pct}%** — {stats['steps_completed']} of {stats['steps_total']} steps: Framework → Entity → Material topics → Evidence → Reported metrics")
    with r2:
        st.metric("Reported metrics", stats["report_value_count"])
    with r3:
        st.metric("Evidence files", stats["evidence_count"])

    if stats["readiness_pct"] < 100:
        next_steps = []
        if stats["framework_count"] == 0:
            next_steps.append("**Materiality & Scope** → Add a reporting framework (e.g. CSRD, GRI).")
        if stats["entity_count"] == 0 and stats["framework_count"] > 0:
            next_steps.append("**Materiality & Scope** → Create a reporting entity and period.")
        if stats["iro_count"] == 0 and stats["entity_count"] > 0:
            next_steps.append("**Materiality & Scope** → Add material topics (IROs) and map to disclosures.")
        if stats["evidence_count"] == 0 and stats["iro_count"] > 0:
            next_steps.append("**Evidence Vault** → Upload source documents and link to material topics.")
        if stats["report_value_count"] == 0 and stats["evidence_count"] > 0:
            next_steps.append("**Agentic Auditor** → Generate a calculator, run on evidence, and save to report.")
        if next_steps:
            with st.expander("What to do next"):
                for s in next_steps:
                    st.markdown(s)

    # ----- Four-step workflow (clear for presentations) -----
    st.markdown("### Workflow")
    st.markdown(
        "Complete each step to build an audit-ready report. Every metric is traceable to **source evidence** and **calculation logic**."
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.page_link("pages/1_Materiality_Scope.py", label="1. Materiality & Scope", icon="📋")
        st.caption("Set reporting entity, period & framework. Run double materiality (IROs) and map to ESRS/GRI.")
    with col2:
        st.page_link("pages/2_Evidence_Vault.py", label="2. Evidence Vault", icon="📁")
        st.caption("Upload source documents and link them to material topics for traceability.")
    with col3:
        st.page_link("pages/3_Agentic_Auditor.py", label="3. Agentic Auditor", icon="🤖")
        st.caption("Generate metric calculators from a sample file; run on all evidence and save to report.")
    with col4:
        st.page_link("pages/4_Report_Audit.py", label="4. Report & Audit", icon="📊")
        st.caption("View disclosure coverage, metrics, evidence, and code — ready for assurance.")

    st.markdown(
        "_Advanced (beta): Use the **Standardised Data Lake** page to normalise messy files to JSON, "
        "tag them, and then run the Agentic Auditor over those JSON datasets._"
    )

    st.divider()
    st.markdown("### Document Q&A (evidence lookup)")
    st.markdown("Index emission factor databases, ESG reports, or policies. Ask questions and get answers **with cited sources** (file, page/sheet).")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        embedding_choice = st.selectbox(
            "Embedding Model",
            options=list(config.EMBEDDING_MODELS.keys()),
            index=0
        )
        llm_choice = st.selectbox(
            "LLM Model",
            options=list(config.LLM_PROVIDERS.keys()),
            index=0
        )
        provider_config = config.LLM_PROVIDERS[llm_choice]

        k_documents = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=4)
        use_hybrid = st.checkbox("Use hybrid search (vector + BM25)", value=True)

        # API keys input
        if provider_config["provider"] == "openai":
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        elif provider_config["provider"] == "google":
            api_key = st.text_input("Google API Key", type="password")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
        elif provider_config["provider"] == "groq":
            api_key = st.text_input("Groq API Key", type="password")
            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
        elif provider_config["provider"] == "ollama":
            ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
            os.environ["OLLAMA_BASE_URL"] = ollama_url

        if st.button("Clear Indexed Data"):
            import shutil
            if os.path.exists(config.PERSIST_DIRECTORY):
                shutil.rmtree(config.PERSIST_DIRECTORY)
            st.session_state.documents = []
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.processed_files = set()
            st.session_state.processed_hashes = set()
            st.success("Cleared all indexed data.")
            st.rerun()

    # Main area: file upload
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, Excel, CSV, TXT)",
        type=["pdf", "xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True
    )

    # Process new uploads
    if uploaded_files:
        process_uploaded_files(uploaded_files, embedding_choice)

    # Display indexed files
    if st.session_state.processed_files:
        st.write("**Indexed files:**")
        for fname in st.session_state.processed_files:
            st.write(f"- {fname}")

    # Query section
    if st.session_state.vectorstore:
        # Create retriever if not exists or if settings changed
        if (st.session_state.retriever is None or
            st.session_state.get("last_k") != k_documents or
            st.session_state.get("last_hybrid") != use_hybrid):
            if use_hybrid:
                st.session_state.retriever = create_hybrid_retriever(
                    st.session_state.vectorstore,
                    st.session_state.documents,
                    k=k_documents
                )
            else:
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": k_documents}
                )
            st.session_state.last_k = k_documents
            st.session_state.last_hybrid = use_hybrid

        # Suggested questions so users know what to ask
        from suggestions import DEFAULT_QA_SUGGESTIONS
        if DEFAULT_QA_SUGGESTIONS:
            st.caption("Suggested questions (click to use):")
            cols = st.columns(min(len(DEFAULT_QA_SUGGESTIONS), 3))
            for i, q in enumerate(DEFAULT_QA_SUGGESTIONS):
                with cols[i % len(cols)]:
                    if st.button(q[:55] + ("..." if len(q) > 55 else ""), key=f"qa_sugg_{i}"):
                        st.session_state.qa_query_prefill = q
                        st.rerun()

        query = st.text_input(
            "Ask a question about your ESG factors:",
            value=st.session_state.get("qa_query_prefill", ""),
            key="qa_query_input",
        )
        if st.session_state.get("qa_query_prefill"):
            st.session_state.qa_query_prefill = None
        if query:
            with st.spinner("Searching and generating answer..."):
                llm = get_llm(provider_config, temperature=0.0, streaming=True)

                # Custom prompt for accuracy and source citation
                template = """You are an ESG and emissions reporting expert. Use ONLY the context below to answer the question.

Rules:
- If the answer is not clearly supported by the context, say "I don't know" or "The provided documents do not contain this information." Do not guess or infer beyond the context.
- Always cite the exact source: file name, and when available sheet name, page number, or row from the context.
- For numbers (e.g. emission factors, totals), quote the value and its unit from the context.
- Keep the answer concise but complete; include all relevant figures and their sources.

Context:
{context}

Question: {question}

Answer (with exact sources):"""
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.retriever,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                    return_source_documents=True,
                    verbose=True
                )

                result = qa_chain({"query": query})

                answer = result.get("result", "")
                sources = result.get("source_documents", [])
                source_files = list({d.metadata.get("source_file", "Unknown") for d in sources})
                log.info("retrieval_qa | query=%s | answer_len=%s | num_sources=%s | source_files=%s", query[:200], len(answer), len(sources), source_files)

                st.markdown("### Answer")
                st.write(result["result"])

                st.markdown("### Sources")
                seen_sources = set()
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source_file", "Unknown")
                    if source not in seen_sources:
                        seen_sources.add(source)
                        st.write(f"- **{source}**")
                        with st.expander(f"Preview from {source}"):
                            st.write(doc.page_content)
    else:
        st.info("Upload documents to start.")

if __name__ == "__main__":
    main()
