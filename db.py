"""Database layer for reporting frameworks, materiality, IROs, evidence, and audit trail."""

import sqlite3
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime, timezone

import config
from logger import get_logger

log = get_logger()

DB_PATH = config.BASE_DIR / "esg_platform.db"


def get_connection():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reporting_frameworks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS material_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            framework_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (framework_id) REFERENCES reporting_frameworks(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS iros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            framework_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            sub_topic TEXT NOT NULL,
            materiality_scope TEXT NOT NULL CHECK (materiality_scope IN ('Financial', 'Impact', 'Both')),
            stakeholder_groups TEXT,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (framework_id) REFERENCES reporting_frameworks(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evidence_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_tag TEXT NOT NULL,
            file_path TEXT NOT NULL,
            original_name TEXT NOT NULL,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(file_path)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS calculator_modules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_tag TEXT NOT NULL UNIQUE,
            code_text TEXT NOT NULL,
            metric_description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reporting_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT NOT NULL,
            reporting_year INTEGER NOT NULL,
            framework_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'in_progress', 'under_review', 'assured')),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (framework_id) REFERENCES reporting_frameworks(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS report_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_tag TEXT NOT NULL,
            metric_name TEXT,
            calculated_value TEXT,
            unit TEXT,
            calculator_module_id INTEGER,
            evidence_file_ids TEXT,
            computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            reporting_entity_id INTEGER,
            FOREIGN KEY (calculator_module_id) REFERENCES calculator_modules(id),
            FOREIGN KEY (reporting_entity_id) REFERENCES reporting_entities(id)
        )
    """)

    # Add new columns if missing (migration-safe)
    def _has_column(table: str, column: str) -> bool:
        cur.execute("PRAGMA table_info(%s)" % table)
        return any(row[1] == column for row in cur.fetchall())

    if not _has_column("iros", "disclosure_code"):
        cur.execute("ALTER TABLE iros ADD COLUMN disclosure_code TEXT")
    if not _has_column("evidence_files", "iro_id"):
        cur.execute("ALTER TABLE evidence_files ADD COLUMN iro_id INTEGER")
    if not _has_column("report_values", "reporting_entity_id"):
        cur.execute("ALTER TABLE report_values ADD COLUMN reporting_entity_id INTEGER")

    conn.commit()
    conn.close()


# ---- Reporting frameworks ----

def create_framework(name: str, description: str = "") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO reporting_frameworks (name, description) VALUES (?, ?)",
        (name, description)
    )
    fid = cur.lastrowid
    conn.commit()
    conn.close()
    log.info("create_framework | name=%s | id=%s", name, fid)
    return fid


def list_frameworks() -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, description, created_at FROM reporting_frameworks ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Material topics (optional sub-entities; IROs are the main DMA output) ----

def create_material_topic(framework_id: int, name: str, description: str = "") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO material_topics (framework_id, name, description) VALUES (?, ?, ?)",
        (framework_id, name, description)
    )
    tid = cur.lastrowid
    conn.commit()
    conn.close()
    return tid


# ---- IROs (Double Materiality Assessment) ----

# ---- Reporting entities (audit period: entity + year + framework + status) ----

def create_reporting_entity(entity_name: str, reporting_year: int, framework_id: int, status: str = "draft") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO reporting_entities (entity_name, reporting_year, framework_id, status)
           VALUES (?, ?, ?, ?)""",
        (entity_name.strip(), int(reporting_year), framework_id, status or "draft")
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    log.info("create_reporting_entity | name=%s year=%s | id=%s", entity_name, reporting_year, eid)
    return eid


def list_reporting_entities(framework_id: Optional[int] = None) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()
    if framework_id is not None:
        cur.execute(
            """SELECT e.id, e.entity_name, e.reporting_year, e.framework_id, e.status, e.created_at, e.updated_at, f.name as framework_name
               FROM reporting_entities e JOIN reporting_frameworks f ON e.framework_id = f.id
               WHERE e.framework_id = ? ORDER BY e.reporting_year DESC, e.created_at DESC""",
            (framework_id,)
        )
    else:
        cur.execute(
            """SELECT e.id, e.entity_name, e.reporting_year, e.framework_id, e.status, e.created_at, e.updated_at, f.name as framework_name
               FROM reporting_entities e JOIN reporting_frameworks f ON e.framework_id = f.id
               ORDER BY e.reporting_year DESC, e.created_at DESC"""
        )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_reporting_entity(entity_id: int) -> Optional[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT e.id, e.entity_name, e.reporting_year, e.framework_id, e.status, e.created_at, e.updated_at, f.name as framework_name
           FROM reporting_entities e JOIN reporting_frameworks f ON e.framework_id = f.id WHERE e.id = ?""",
        (entity_id,)
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def update_reporting_entity_status(entity_id: int, status: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "UPDATE reporting_entities SET status = ?, updated_at = ? WHERE id = ?",
        (status, now, entity_id)
    )
    conn.commit()
    conn.close()
    log.info("update_reporting_entity_status | id=%s status=%s", entity_id, status)


# ---- IROs (Double Materiality Assessment) ----

def create_iro(
    framework_id: int,
    topic: str,
    sub_topic: str,
    materiality_scope: str,
    stakeholder_groups: str = "",
    description: str = "",
    disclosure_code: str = "",
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO iros (framework_id, topic, sub_topic, materiality_scope, stakeholder_groups, description, disclosure_code)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (framework_id, topic, sub_topic, materiality_scope, stakeholder_groups, description, disclosure_code or None)
    )
    iid = cur.lastrowid
    conn.commit()
    conn.close()
    log.info("create_iro | framework_id=%s topic=%s sub_topic=%s scope=%s | id=%s", framework_id, topic, sub_topic, materiality_scope, iid)
    return iid


def list_iros(framework_id: Optional[int] = None) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()
    if framework_id is not None:
        cur.execute(
            """SELECT id, framework_id, topic, sub_topic, materiality_scope, stakeholder_groups, description, created_at, disclosure_code
               FROM iros WHERE framework_id = ? ORDER BY created_at DESC""",
            (framework_id,)
        )
    else:
        cur.execute(
            """SELECT id, framework_id, topic, sub_topic, materiality_scope, stakeholder_groups, description, created_at, disclosure_code
               FROM iros ORDER BY created_at DESC"""
        )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Evidence files ----

def add_evidence_file(evidence_tag: str, file_path: str, original_name: str, iro_id: Optional[int] = None) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO evidence_files (evidence_tag, file_path, original_name, iro_id) VALUES (?, ?, ?, ?)",
        (evidence_tag, file_path, original_name, iro_id)
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid


def list_evidence_by_tag(evidence_tag: str) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, evidence_tag, file_path, original_name, uploaded_at FROM evidence_files WHERE evidence_tag = ? ORDER BY uploaded_at",
        (evidence_tag,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_all_evidence_tags() -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT evidence_tag FROM evidence_files ORDER BY evidence_tag")
    rows = cur.fetchall()
    conn.close()
    return [r["evidence_tag"] for r in rows]


def get_evidence_tags_for_iro(iro_id: int) -> List[str]:
    """Return distinct evidence tags for files linked to this IRO."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT evidence_tag FROM evidence_files WHERE iro_id = ?", (iro_id,))
    rows = cur.fetchall()
    conn.close()
    return [r["evidence_tag"] for r in rows]


# ---- Calculator modules ----

def save_calculator_module(evidence_tag: str, code_text: str, metric_description: str = "") -> int:
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """INSERT INTO calculator_modules (evidence_tag, code_text, metric_description, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(evidence_tag) DO UPDATE SET
             code_text = excluded.code_text,
             metric_description = excluded.metric_description,
             updated_at = excluded.updated_at""",
        (evidence_tag, code_text, metric_description, now)
    )
    cur.execute("SELECT id FROM calculator_modules WHERE evidence_tag = ?", (evidence_tag,))
    row = cur.fetchone()
    mid = row["id"]
    conn.commit()
    conn.close()
    log.info("save_calculator_module | tag=%s | module_id=%s", evidence_tag, mid)
    return mid


def get_calculator_module(evidence_tag: str) -> Optional[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, evidence_tag, code_text, metric_description, created_at, updated_at FROM calculator_modules WHERE evidence_tag = ?",
        (evidence_tag,)
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_calculator_module_by_id(module_id: int) -> Optional[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, evidence_tag, code_text, metric_description, created_at, updated_at FROM calculator_modules WHERE id = ?",
        (module_id,)
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_calculator_modules() -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, evidence_tag, metric_description, created_at, updated_at FROM calculator_modules ORDER BY updated_at DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Report values (audit trail) ----

def save_report_value(
    evidence_tag: str,
    metric_name: str,
    calculated_value: Any,
    unit: str = "",
    calculator_module_id: Optional[int] = None,
    evidence_file_ids: Optional[List[int]] = None,
    reporting_entity_id: Optional[int] = None,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    eids_str = ",".join(map(str, evidence_file_ids or []))
    cur.execute(
        """INSERT INTO report_values (evidence_tag, metric_name, calculated_value, unit, calculator_module_id, evidence_file_ids, reporting_entity_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (evidence_tag, metric_name, str(calculated_value), unit, calculator_module_id, eids_str, reporting_entity_id)
    )
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    log.info("save_report_value | tag=%s | metric=%s | value=%s | report_id=%s", evidence_tag, metric_name, calculated_value, rid)
    return rid


def list_report_values(
    evidence_tag: Optional[str] = None,
    reporting_entity_id: Optional[int] = None,
) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()
    if evidence_tag:
        cur.execute(
            """SELECT r.id, r.evidence_tag, r.metric_name, r.calculated_value, r.unit, r.calculator_module_id, r.evidence_file_ids, r.computed_at, r.reporting_entity_id
               FROM report_values r WHERE r.evidence_tag = ? ORDER BY r.computed_at DESC""",
            (evidence_tag,)
        )
    elif reporting_entity_id is not None:
        cur.execute(
            """SELECT r.id, r.evidence_tag, r.metric_name, r.calculated_value, r.unit, r.calculator_module_id, r.evidence_file_ids, r.computed_at, r.reporting_entity_id
               FROM report_values r WHERE r.reporting_entity_id = ? ORDER BY r.computed_at DESC""",
            (reporting_entity_id,)
        )
    else:
        cur.execute(
            """SELECT r.id, r.evidence_tag, r.metric_name, r.calculated_value, r.unit, r.calculator_module_id, r.evidence_file_ids, r.computed_at, r.reporting_entity_id
               FROM report_values r ORDER BY r.computed_at DESC"""
        )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Audit readiness & disclosure coverage ----

def get_audit_readiness_stats() -> dict:
    """
    Return counts and a simple readiness score (0–100) for dashboard.
    Score: each of (framework, entity, IROs, evidence, reported metrics) contributes 20%.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM reporting_frameworks")
    framework_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM reporting_entities")
    entity_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM iros")
    iro_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM evidence_files")
    evidence_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM report_values")
    report_value_count = cur.fetchone()[0]
    conn.close()
    steps = [
        framework_count > 0,
        entity_count > 0,
        iro_count > 0,
        evidence_count > 0,
        report_value_count > 0,
    ]
    readiness_pct = round(100 * sum(steps) / 5) if steps else 0
    return {
        "framework_count": framework_count,
        "entity_count": entity_count,
        "iro_count": iro_count,
        "evidence_count": evidence_count,
        "report_value_count": report_value_count,
        "readiness_pct": readiness_pct,
        "steps_completed": sum(steps),
        "steps_total": 5,
    }


def list_iros_with_evidence_count(framework_id: Optional[int] = None) -> List[dict]:
    """List IROs with evidence_count (number of evidence files linked to that IRO)."""
    conn = get_connection()
    cur = conn.cursor()
    if framework_id is not None:
        cur.execute(
            """SELECT i.id, i.framework_id, i.topic, i.sub_topic, i.materiality_scope, i.disclosure_code,
                      COALESCE(COUNT(e.id), 0) AS evidence_count
               FROM iros i
               LEFT JOIN evidence_files e ON e.iro_id = i.id
               WHERE i.framework_id = ?
               GROUP BY i.id
               ORDER BY i.created_at DESC""",
            (framework_id,),
        )
    else:
        cur.execute(
            """SELECT i.id, i.framework_id, i.topic, i.sub_topic, i.materiality_scope, i.disclosure_code,
                      COALESCE(COUNT(e.id), 0) AS evidence_count
               FROM iros i
               LEFT JOIN evidence_files e ON e.iro_id = i.id
               GROUP BY i.id
               ORDER BY i.created_at DESC"""
        )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize on import
init_db()
