"""Retrieval: hybrid retriever with ensemble (vector + BM25)."""

from typing import List

from logger import get_logger

log = get_logger()

# ---- Safe Imports for LangChain Version Compatibility ----

# Prefer langchain_community (recommended); fall back to langchain or langchain_classic
try:
    from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
except Exception:
    try:
        from langchain.retrievers import BM25Retriever, EnsembleRetriever
    except Exception:
        from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever

# Document may live in different locations
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document


# ---- Hybrid Retriever ----

def create_hybrid_retriever(
    vectorstore,
    documents: List[Document],
    k: int = 4,
    weights: List[float] = None
):
    """
    Create an ensemble retriever combining:
    - Vector similarity search (semantic search)
    - BM25 keyword search

    Args:
        vectorstore: Chroma (or compatible) vector store instance
        documents: List of LangChain Document objects
        k: number of documents to retrieve
        weights: weights for ensemble (vector_weight, bm25_weight)

    Returns:
        EnsembleRetriever
    """

    if weights is None:
        weights = [0.5, 0.5]

    # --- Vector Retriever (Semantic Search) ---
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # --- BM25 Retriever (Keyword Search) ---
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # --- Ensemble Retriever ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights
    )
    log.info("create_hybrid_retriever | k=%s weights=%s", k, weights)
    return ensemble_retriever


def get_retriever_by_tag(vectorstore, evidence_tag: str, k: int = 4):
    """
    Return a retriever that only returns documents with the given evidence_tag in metadata.
    Use for Evidence Vault / Agentic Auditor to fetch sample documents by tag.
    """
    log.info("get_retriever_by_tag | tag=%s k=%s", evidence_tag, k)
    return vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"evidence_tag": evidence_tag},
        }
    )