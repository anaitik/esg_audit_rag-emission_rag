"""Document ingestion: load, split, embed, and index."""

import tempfile
import os
from typing import List, Optional
import pandas as pd
import openpyxl

try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document

import config
from utils import compute_documents_hash
from logger import get_logger

log = get_logger()


def load_document_from_file(uploaded_file, extra_metadata: Optional[dict] = None) -> List[Document]:
    """Load a single uploaded file and convert to LangChain Documents.
    extra_metadata (e.g. {'evidence_tag': 'CSRD_ClimateChange_Scope1_2025'}) is added to every document.
    """
    extra_metadata = extra_metadata or {}
    docs = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Import document loaders lazily to avoid pulling heavy deps during module import
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
        except Exception:
            try:
                from langchain_classic.document_loaders import PyPDFLoader, TextLoader, CSVLoader
            except Exception:
                from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
        elif uploaded_file.name.endswith(".csv"):
            loader = CSVLoader(tmp_path)
            docs = loader.load()
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            # Custom Excel handling
            wb = openpyxl.load_workbook(tmp_path, data_only=True)
            for sheet_name in wb.sheetnames:
                df = pd.read_excel(tmp_path, sheet_name=sheet_name, engine="openpyxl")
                for idx, row in df.iterrows():
                    content = f"Sheet: {sheet_name}, Row {idx+2}: " + ", ".join(
                        [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                    )
                    metadata = {
                        "source_file": uploaded_file.name,
                        "sheet": sheet_name,
                        "row": idx+2,
                        "type": "excel"
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Add source file metadata and any extra (e.g. evidence_tag) to every document
    for doc in docs:
        if "source_file" not in doc.metadata:
            doc.metadata["source_file"] = uploaded_file.name
        for k, v in extra_metadata.items():
            if v is not None:
                doc.metadata[k] = v
    log.info("load_document_from_file | file=%s doc_count=%s extra_metadata=%s", uploaded_file.name, len(docs), extra_metadata)
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    try:
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except Exception:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    split = text_splitter.split_documents(documents)
    log.info("split_documents | input_docs=%s output_chunks=%s", len(documents), len(split))
    return split


def get_embeddings(embedding_choice: str):
    """Create embeddings instance based on user choice."""
    model_name = config.EMBEDDING_MODELS[embedding_choice]
    # Instantiate embeddings lazily to avoid importing heavy libraries at module import
    if model_name == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()
        except Exception:
            try:
                from langchain_community.embeddings.openai import OpenAIEmbeddings
                return OpenAIEmbeddings()
            except Exception:
                from langchain_core.embeddings import FakeEmbeddings
                return FakeEmbeddings()
    else:
        try:
            from langchain_classic.embeddings.huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception:
            try:
                from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=model_name)
            except Exception:
                from langchain_core.embeddings import FakeEmbeddings
                return FakeEmbeddings()


def create_vector_store(
    documents: List[Document],
    embedding_choice: str,
    persist_directory: str = str(config.PERSIST_DIRECTORY),
    collection_name: str = None,
) -> object:
    """Create or update Chroma vector store. Returns vectorstore and whether it was rebuilt."""
    collection_name = collection_name or config.COLLECTION_NAME
    embeddings = get_embeddings(embedding_choice)
    doc_hash = compute_documents_hash(documents)

    # Check if we have a previously saved hash (only for default collection to avoid breaking main app)
    hash_file = os.path.join(str(persist_directory), f"doc_hash_{collection_name}.txt")
    if os.path.exists(persist_directory) and os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            old_hash = f.read()
        if old_hash == doc_hash:
            # Load existing store
            try:
                from langchain_community.vectorstores import Chroma
            except Exception:
                try:
                    from langchain_core.vectorstores import Chroma
                except Exception:
                    from langchain.vectorstores import Chroma

            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_directory)
            )
            log.info("create_vector_store | collection=%s rebuilt=False (loaded existing)", collection_name)
            return vectorstore, False

    # Create new store
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:
        try:
            from langchain_core.vectorstores import Chroma
        except Exception:
            from langchain.vectorstores import Chroma

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory)
    )
    vectorstore.persist()
    os.makedirs(str(persist_directory), exist_ok=True)
    with open(hash_file, "w") as f:
        f.write(doc_hash)
    log.info("create_vector_store | collection=%s docs=%s rebuilt=True", collection_name, len(documents))
    return vectorstore, True


def add_documents_to_vector_store(
    vectorstore,
    documents: List[Document],
) -> None:
    """Add new documents to an existing Chroma vector store (e.g. for Evidence Vault tag-based uploads)."""
    if not documents:
        return
    vectorstore.add_documents(documents)
    log.info("add_documents_to_vector_store | doc_count=%s", len(documents))
    try:
        vectorstore.persist()
    except Exception:
        pass


def get_existing_vector_store(
    embedding_choice: str,
    persist_directory: str = str(config.PERSIST_DIRECTORY),
    collection_name: str = None,
):
    """Load existing Chroma from disk if it exists; otherwise return None."""
    collection_name = collection_name or config.COLLECTION_NAME
    if not os.path.exists(persist_directory):
        return None
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:
        try:
            from langchain.vectorstores import Chroma
        except Exception:
            from langchain_core.vectorstores import Chroma
    embeddings = get_embeddings(embedding_choice)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    return vectorstore
