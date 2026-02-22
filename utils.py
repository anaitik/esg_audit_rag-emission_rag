"""Utility functions for hashing, caching, etc."""

import hashlib
import pickle
from typing import List
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document


def compute_documents_hash(documents: List[Document]) -> str:
    """Compute a hash of the documents to detect changes."""
    # Extract content and metadata
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    # Serialize and hash
    serialized = pickle.dumps((texts, metadatas))
    return hashlib.md5(serialized).hexdigest()
