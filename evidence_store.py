"""Evidence Vault: persist uploaded files to disk by tag (no vectorization)."""

import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import config
import db
from logger import get_logger

log = get_logger()

config.EVIDENCE_STORE_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, evidence_tag: str, iro_id: Optional[int] = None) -> str:
    """
    Save an uploaded file under evidence_store/<tag>/ with a unique name.
    Files are stored in a folder per tag (no vectorization). Register in evidence_files table.
    Optionally link to an IRO (material topic) for audit traceability.
    Returns the stored file path (absolute).
    """
    safe_tag = evidence_tag.replace(" ", "_").strip() or "untagged"
    unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    dest_dir = config.EVIDENCE_STORE_DIR / safe_tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / unique_name
    dest_path.write_bytes(uploaded_file.getvalue())
    path_str = str(dest_path)
    db.add_evidence_file(evidence_tag=safe_tag, file_path=path_str, original_name=uploaded_file.name, iro_id=iro_id)
    log.info("evidence_upload | tag=%s | file=%s | path=%s", safe_tag, uploaded_file.name, path_str)
    return path_str


def get_file_paths_for_tag(evidence_tag: str) -> List[Tuple[str, str]]:
    """Return list of (file_path, original_name) for the given tag."""
    rows = db.list_evidence_by_tag(evidence_tag)
    return [(r["file_path"], r["original_name"]) for r in rows]
